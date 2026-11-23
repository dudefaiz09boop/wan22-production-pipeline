const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const {
  KaggleProvider,
  buildImageNotebook,
  buildQwenImageNotebook,
  buildGenericImageNotebook,
  buildNotebook,
  buildVideoNotebook,
  buildQwenWorkerNotebook,
  buildVideoWorkerNotebook,
  kernelStatus,
  readModelMap,
  resolveKaggleCli,
} = require("../gateway/kaggle");

test("Kaggle CLI configuration supports executable and Python-module forms", () => {
  assert.deepEqual(resolveKaggleCli({ KAGGLE_CLI: "kaggle.exe" }), {
    executable: "kaggle.exe",
    prefixArgs: [],
  });
  assert.deepEqual(resolveKaggleCli({ KAGGLE_CLI: "python -m kaggle" }), {
    executable: "python",
    prefixArgs: ["-m", "kaggle"],
  });
  assert.deepEqual(resolveKaggleCli({ KAGGLE_CLI: '"C:\\Python 314\\python.exe" -m kaggle' }), {
    executable: "C:\\Python 314\\python.exe",
    prefixArgs: ["-m", "kaggle"],
  });
});

test("Kaggle routing selects only configured models above three billion parameters", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-route-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: { KAGGLE_ENABLED: "false" },
  });
  assert.equal(provider.shouldUseKaggle("qwen2.5:3b"), false);
  assert.equal(provider.shouldUseKaggle("phi3:mini"), true);
  assert.equal(provider.shouldUseKaggle("not-configured"), false);
  provider.stop();
  db.close();
  fs.rmSync(root, { recursive: true, force: true });
});

test("remote submission is blocked unless explicit consent and Kaggle configuration are present", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-consent-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "false",
      KAGGLE_USERNAME: "test-user",
    },
  });
  await assert.rejects(
    provider.submitChat({
      model: "phi3:mini",
      messages: [{ role: "user", content: "private prompt" }],
    }),
    /explicitly set KAGGLE_REMOTE_DATA_CONSENT=true/,
  );
  assert.equal(db.prepare("SELECT COUNT(*) AS count FROM compute_jobs").get().count, 0);
  provider.stop();
  db.close();
  fs.rmSync(root, { recursive: true, force: true });
});

test("GGUF Model Hub checkpoints cannot be sent to the generic image workflow", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-gguf-image-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    modelHub: {
      folderKeyForFilename: () => "civitai-1-2",
    },
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
    },
  });
  try {
    await assert.rejects(
      provider.submitModelHubImage({ prompt: "a test image", model: "wan-high.gguf" }),
      /GGUF transformer checkpoints cannot be loaded by the generic image workflow/,
    );
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM compute_jobs").get().count, 0);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("Model Hub generation rejects a LoRA whose base model does not match the checkpoint", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-hub-lora-mismatch-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    modelHub: {
      folderKeyForFilename: () => "civitai-1-2",
      isAllowlistedLora: () => true,
      isCompatibleLora: () => false,
    },
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
    },
  });
  provider.pump = () => {};
  try {
    await assert.rejects(
      provider.submitModelHubImage({
        prompt: "A still-life of a glass vase",
        model: "checkpoint.safetensors",
        loras: ["incompatible.safetensors"],
        loraStrengths: [0.7],
      }),
      /exact SD 1.5\/SDXL-family base-model match/,
    );
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM compute_jobs").get().count, 0);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("Kaggle notebook embeds request data and keeps kernel private", () => {
  const notebook = buildNotebook("Qwen/Qwen2.5-3B-Instruct", {
    messages: [{ role: "user", content: "test prompt" }],
    maxTokens: 32,
  });
  const metadata = readModelMap();
  assert.equal(metadata["qwen2.5:3b"].repo, "Qwen/Qwen2.5-3B-Instruct");
  assert.equal(notebook.nbformat, 4);
  assert.equal(notebook.cells[0].id, "local-ai-chat");
  assert.match(notebook.cells[0].source.join(""), /result\.json/);
  assert.match(notebook.cells[0].source.join(""), /torch\.cuda\.is_available/);
});

test("image notebook creates a 512px GPU image and leaves the safety checker enabled", () => {
  const notebook = buildImageNotebook({
    prompt: "a unicorn dragon with red eyes over flaming lava",
    negativePrompt: "blurry",
    model: "stable-diffusion-v1-5/stable-diffusion-v1-5",
    steps: 24,
    seed: 42,
  });
  const source = notebook.cells[0].source.join("");
  assert.equal(notebook.cells[0].id, "local-ai-image");
  assert.match(source, /output_size = 1024 if payload\.get\("refine"\) else 512/);
  assert.match(source, /"height": output_size/);
  assert.match(source, /DPMSolverMultistepScheduler\.from_config/);
  assert.doesNotMatch(source, /enable_attention_slicing/);
  assert.match(source, /candidate-\{index \+ 1\}\.png/);
  assert.match(source, /StableDiffusionImg2ImgPipeline/);
  assert.match(source, /ImageOps\.fit/);
  assert.match(source, /payload\["strength"\]/);
  assert.doesNotMatch(source, /safety_checker=None/);
  assert.match(source, /notebookStartupMs/);
  assert.match(source, /modelDownloadLoadMs/);
  assert.match(source, /inferenceMs/);
  const encoded = source.match(/payload = json\.loads\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/)?.[1];
  assert.equal(JSON.parse(Buffer.from(encoded, "base64").toString("utf8")).candidates, 1);
});

test("image notebook supports timed sequential candidates and optional 2x refinement", () => {
  const notebook = buildImageNotebook({
    prompt: "a bright mountain lake",
    negativePrompt: "",
    model: "Lykon/dreamshaper-8",
    steps: 20,
    candidates: 3,
    seed: 41,
  });
  const source = notebook.cells[0].source.join("");
  assert.match(source, /for index in range\(payload\["candidates"\]\)/);
  assert.match(source, /candidate_seed = \(payload\["seed"\] \+ index\) % 2147483648/);
  assert.match(source, /"outputWriteMs": output_write_ms/);
  assert.match(source, /output_size = 1024 if payload\.get\("refine"\) else 512/);
  assert.match(source, /pipe\.enable_vae_tiling\(\)/);
  assert.match(source, /notebookStartedAt/);
});

function pythonCommand() {
  for (const candidate of ["python", "py"]) {
    try {
      execFileSync(candidate, ["-c", "import sys; sys.exit(0)"], { stdio: "ignore" });
      return candidate;
    } catch {
      // Try the next Python launcher.
    }
  }
  return null;
}

test("generated Kaggle notebooks are syntactically valid Python", (t) => {
  const python = pythonCommand();
  if (!python) return t.skip("Python is unavailable, so generated notebooks were not parsed");
  const notebooks = {
    image: buildImageNotebook({
      prompt: "syntax check",
      negativePrompt: "",
      model: "stable-diffusion-v1-5/stable-diffusion-v1-5",
      steps: 20,
      candidates: 2,
      seed: 7,
    }),
    video: buildVideoNotebook({
      prompt: "syntax check",
      negativePrompt: "",
      frames: 17,
      fps: 8,
      steps: 8,
      seed: 7,
      modelRepo: "Comfy-Org/Wan-2.1",
      modelRevision: "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
      modelBytes: 10_500_000_000,
    }),
    qwenImage: buildQwenImageNotebook({
      prompt: "syntax check",
      negativePrompt: "",
      model: "qwen-image-21-uncensored-omni-quantized.safetensors",
      steps: 30,
      width: 1344,
      height: 1344,
      seed: 7,
      cfg: 3.5,
      modelSource: {
        baseUrl: "https://models.example.test",
        key: "qwenomni",
        expiresAt: 1234567890,
        signature: "a".repeat(64),
        maxBytes: 1024,
        totalBytes: 512,
      },
    }),
    genericImage: buildGenericImageNotebook({
      prompt: "syntax check",
      checkpointFilename: "model.safetensors",
    }),
    imageWorker: buildQwenWorkerNotebook({
      sessionId: "syntax-check",
      token: "a".repeat(64),
      model: "qwen-image-2.1-step90000-FP8-comfy.safetensors",
      modelSource: {
        baseUrl: "https://models.example.test",
        key: "qwenomni",
        expiresAt: 1234567890,
        signature: "a".repeat(64),
      },
      bridgeBaseUrl: "https://models.example.test",
      idleTimeoutSeconds: 900,
      maxRuntimeSeconds: 3600,
    }),
    videoWorker: buildVideoWorkerNotebook({
      sessionId: "syntax-check",
      token: "a".repeat(64),
      bridgeBaseUrl: "https://models.example.test",
      profile: {
        id: "wan-t2v-1.3b",
        repo: "Comfy-Org/Wan-2.1",
        revision: "123acf1cc74bccbb9fff8ac1ee72edc08c2341d",
        diffusion: "wan2.1_t2v_1.3B_fp16.safetensors",
        textEncoder: "umt5.safetensors",
        vae: "wan_2.1_vae.safetensors",
      },
    }),
    cogVideoXWorker: buildVideoWorkerNotebook({
      sessionId: "syntax-check",
      token: "a".repeat(64),
      bridgeBaseUrl: "https://models.example.test",
      profile: {
        id: "cogvideox-2b",
        engine: "diffusers",
        repo: "zai-org/CogVideoX-2b",
        revision: "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01",
      },
    }),
  };
  for (const [name, notebook] of Object.entries(notebooks)) {
    const source = notebook.cells.find((cell) => cell.id === "local-ai-video-worker")?.source.join("")
      || notebook.cells[0].source.join("");
    assert.ok(Buffer.byteLength(JSON.stringify(notebook, null, 2), "utf8") < 750 * 1024, `${name} notebook should fit Kaggle's upload limit`);
    if (name !== "image" && name !== "cogVideoXWorker") {
      assert.match(source, /--require-hashes/, `${name} notebook must install the pinned ComfyUI lock`);
      assert.match(source, /--no-deps/, `${name} notebook must preserve Kaggle's preinstalled CUDA stack`);
    }
    if (name === "cogVideoXWorker") {
      assert.match(source, /CogVideoXPipeline\.from_pretrained/);
      assert.match(source, /pipe\.enable_sequential_cpu_offload\(\)/);
      assert.match(source, /pipe\.vae\.enable_tiling\(\)/);
      assert.match(source, /num_frames=request\["frames"\] - 1/);
      assert.match(source, /video_path\.unlink\(missing_ok=True\)/);
      assert.match(source, /server = None/);
      assert.doesNotMatch(source, /server_log = open\("\/tmp\/comfyui\.log"/);
    }
    for (const [index, cell] of notebook.cells.entries()) {
      if (cell.cell_type !== "code") continue;
      const file = path.join(os.tmpdir(), `local-ai-${name}-notebook-${index}-${process.pid}-${Date.now()}.py`);
      fs.writeFileSync(file, cell.source.join(""), "utf8");
      try {
        execFileSync(python, ["-c", "import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())", file], {
          stdio: "pipe",
          timeout: 30000,
        });
      } catch (error) {
        assert.fail(`${name} notebook cell ${index} is not valid Python: ${String(error.stderr || error.message).trim()}`);
      } finally {
        fs.rmSync(file, { force: true });
      }
    }
  }
});

test("local studio exposes unified image generation, character prompts and Hub imports", () => {
  const html = fs.readFileSync(path.join(__dirname, "..", "gateway", "public", "index.html"), "utf8");
  const script = fs.readFileSync(path.join(__dirname, "..", "gateway", "public", "studio.js"), "utf8");
  const imageModelSelect = html.match(/<select id="image-model">([\s\S]*?)<\/select>/);
  assert.ok(imageModelSelect);
  assert.doesNotMatch(imageModelSelect[1], /qwen-image-21-uncensored-omni-quantized\.safetensors/);
  assert.ok(/data-engine="local"/.test(imageModelSelect[1]));
  assert.match(html, /optgroup label="Model Hub imports" id="hub-imports-optgroup"/);
  assert.match(script, /setMode\("image"\)/);
  assert.match(script, /function applyImageEngineDefaults/) && assert.match(script, /function updateImageEstimate/);
  assert.match(html, /Krea-2 LoRA/);
  assert.match(html, /Image reference/);
  assert.match(html, /OPTIONAL/);
  assert.match(html, /<select id="character-select">/);
  assert.match(html, /id="characters-button"/);
  assert.match(script, /\/v1\/characters\/[^/]+\/prompt/);
  assert.match(html, /model-hub-import/);
  assert.match(script, /\/v1\/model-library/);
  // Unified flow: Qwen-Image-2.1 (DiT) goes through /v1/qwen-image/generations,
  // Model Hub checkpoints through /v1/model-hub/generations, no SD 1.5 path.
  assert.match(script, /\/v1\/qwen-image\/generations/);
  assert.match(script, /\/v1\/model-hub\/generations/);
  assert.match(script, /Reference images require a Krea-2 checkpoint with remote Kaggle enabled/);
  assert.ok(script.includes('if (!["local-dit", "hub-krea2"].includes(entry.engine)) continue;'));
  assert.ok(script.includes('option.dataset.workerSupported = "true"'));
  assert.ok(script.includes("workerStartPending = true"));
  assert.ok(script.includes("ensureImageWorkerReady"));
  assert.ok(script.includes("Worker preparation failed:"));
  assert.doesNotMatch(script, /\/v1\/images\/generations/);
  assert.doesNotMatch(script, /\/v1\/videos\/generations/);
  assert.doesNotMatch(script, /applyVideoPreset/);
  assert.doesNotMatch(script, /Camera movement:/);
  assert.doesNotMatch(script, /Math\.ceil\(8 \/ fps\)/);
  assert.match(script, /function optimizeReferenceImage/);
  assert.match(script, /fileAsBase64/);
  assert.match(script, /openCharactersDialog/);
  assert.match(html, /id="character-create"/);
  assert.match(html, /id="character-refresh"/);
});

test("image submission accepts SSD-mapped non-default checkpoints through Kaggle", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-image-model-map-"));
  const db = new DatabaseSync(":memory:");
  const modelDir = path.join(root, "models", "qwen-omni");
  fs.mkdirSync(modelDir, { recursive: true });
  fs.writeFileSync(path.join(modelDir, "qwen-image-21-uncensored-omni-quantized.safetensors"), "weights");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
      KAGGLE_MODEL_TRANSFER_CONSENT: "true",
      KAGGLE_MODEL_STORE_URL: "https://models.example.test",
      KAGGLE_MODEL_STORE_SECRET: "test-secret-012345678901234567890123456789",
      KAGGLE_MODEL_STORE_MAP: JSON.stringify({ qwenomni: modelDir }),
      KAGGLE_IMAGE_MODEL_KEY_MAP: JSON.stringify({
        "qwen-image-21-uncensored-omni-quantized.safetensors": "qwenomni",
      }),
    },
  });
  provider.pump = () => {};
  try {
    const task = await provider.submitImage({
      prompt: "portrait",
      steps: 20,
      model: "qwen-image-21-uncensored-omni-quantized.safetensors",
      seed: 42,
    });
    const notebook = JSON.parse(fs.readFileSync(path.join(root, "kaggle", "jobs", task.id, "code.ipynb"), "utf8"));
    const payloadEncoded = notebook.cells[0].source.join("").match(/payload = json\.loads\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/)?.[1];
    assert.ok(payloadEncoded);
    const payload = JSON.parse(Buffer.from(payloadEncoded, "base64").toString("utf8"));
    assert.equal(payload.model, "qwen-image-21-uncensored-omni-quantized.safetensors");
    assert.equal(payload.modelSource.key, "qwenomni");
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("qwen kaggle image notebook builds a real ComfyUI graph fed from SSD-backed storage", () => {
  const notebook = buildQwenImageNotebook({
    prompt: "portrait",
    negativePrompt: "blurry",
    model: "qwen-image-21-uncensored-omni-quantized.safetensors",
    steps: 30,
    width: 1344,
    height: 1344,
    seed: 42,
    cfg: 3.5,
    modelSource: {
      baseUrl: "https://models.example.test",
      key: "qwenomni",
      expiresAt: 1234567890,
      signature: "a".repeat(64),
      maxBytes: 1024,
      totalBytes: 512,
    },
  });
  const source = notebook.cells[0].source.join("");
  assert.equal(notebook.cells[0].id, "local-ai-qwen-image");
  assert.match(source, /"git", "clone", "--depth", "1"/);
  assert.match(source, /Comfy-Org\/ComfyUI\.git/);
  assert.match(source, /2c75dc1b379a59e0880619d029607af5169a62ef/);
  assert.match(source, /"UNETLoader"/);
  assert.match(source, /"CLIPLoader"/);
  assert.match(source, /"type":"qwen_image"/);
  assert.match(source, /"VAELoader"/);
  assert.match(source, /"TextEncodeQwenImage21"/);
  assert.doesNotMatch(source, /LoadCheckpoint/);
  assert.doesNotMatch(source, /DiffusionPipeline/);
  assert.match(source, /engine": "qwen-kaggle/);
  assert.match(source, /candidate-1\.png/);
});

test("qwen kaggle image notebook wires MOSS-Image checkpoints to the moss_image CLIP type", () => {
  const notebook = buildQwenImageNotebook({
    prompt: "portrait",
    negativePrompt: "",
    model: "jib-mix-moss-realistic.safetensors",
    steps: 20,
    width: 1024,
    height: 1344,
    seed: 7,
    cfg: 1,
    modelSource: {
      baseUrl: "https://models.example.test",
      key: "jibmoss",
      expiresAt: 1234567890,
      signature: "b".repeat(64),
      maxBytes: 1024,
      totalBytes: 512,
    },
  });
  const source = notebook.cells[0].source.join("");
  assert.match(source, /"type":"moss_image"/);
  assert.match(source, /"EmptyFlux2LatentImage"/);
  assert.match(source, /"CLIPTextEncode"/);
});

test("qwen kaggle image notebook wires Krea-2 LoRA and reference-image conditioning", () => {
  const notebook = buildQwenImageNotebook({
    prompt: "portrait",
    negativePrompt: "",
    model: "Ultra Realistic by Stable Yogi Krea2 V1.0 Civ INT8-ConvRot C89.safetensors",
    steps: 12,
    width: 1024,
    height: 1536,
    seed: 7,
    cfg: 1.5,
    lora: "krea2_style_reference.safetensors",
    loraStrength: 1,
    referenceImage: Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).toString("base64"),
    modelSource: {
      baseUrl: "https://models.example.test",
      key: "stable-yogi-krea2",
      expiresAt: 1234567890,
      signature: "c".repeat(64),
      maxBytes: 20 * 1024 ** 3,
      totalBytes: 18 * 1024 ** 3,
    },
  });
  const source = notebook.cells[0].source.join("");
  assert.match(source, /"type":"krea2"/);
  assert.match(source, /LoraLoaderModelOnly/);
  assert.match(source, /TextEncodeQwenImageEditPlus/);
  assert.match(source, /FluxKontextMultiReferenceLatentMethod/);
  assert.match(source, /krea2_reference_image\.png/);
});

test("qwen image submission requires a mapped SSD checkpoint for Kaggle remote mode", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-qwen-kaggle-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
    },
  });
  provider.pump = () => {};
  try {
    await assert.rejects(
      provider.submitQwenImage({
        prompt: "portrait",
        model: "qwen-image-21-uncensored-omni-quantized.safetensors",
      }),
      /Map the selected Qwen checkpoint in KAGGLE_IMAGE_MODEL_KEY_MAP|Set KAGGLE_MODEL_TRANSFER_CONSENT=true/,
    );
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("video notebook downloads only the pinned Wan files directly and waits for ComfyUI completion", () => {
  const notebook = buildVideoNotebook({
    prompt: "A short cinematic desert scene",
    negativePrompt: "text, watermark",
    frames: 17,
    fps: 8,
    steps: 8,
    seed: 42,
    modelRepo: "Comfy-Org/Wan-2.1",
    modelRevision: "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
    modelBytes: 10_500_000_000,
  });
  const source = notebook.cells[0].source.join("");
  assert.equal(notebook.cells[0].id, "local-ai-video");
  assert.match(source, /wan2\.1_t2v_1\.3B_fp16\.safetensors/);
  assert.match(source, /2c75dc1b379a59e0880619d029607af5169a62ef/);
  assert.match(source, /umt5_xxl_fp8_e4m3fn_scaled\.safetensors/);
  assert.match(source, /SaveWEBM/);
  assert.match(source, /maestro-video\.webm/);
  assert.match(source, /--lowvram/);
  assert.match(source, /huggingface_hub import hf_hub_download/);
  assert.match(source, /local_dir="\/kaggle\/working\/\.hf-download"/);
  assert.match(source, /modelRevision/);
  assert.doesNotMatch(source, /laptop-ssd/);
  assert.match(source, /shutil\.rmtree\(comfyui_dir/);
  assert.match(source, /report_phase\("Wan model download"/);
  assert.match(source, /report_phase\("Wan video inference"/);
  assert.match(source, /phase_seconds\[name\] = round/);
  assert.match(source, /"timings": timings/);
  assert.match(source, /prompt_id = response\.json\(\)\.get\("prompt_id"\)[\s\S]*?while True:/);
  assert.doesNotMatch(source, /Wan video generation exceeded its one-hour execution limit/);
  assert.doesNotMatch(source, /deadline = time\.time\(\) \+ 60 \* 60/);
});

test("kernel status parser distinguishes terminal and pending states", () => {
  assert.equal(kernelStatus("KernelWorkerStatus.COMPLETE"), "complete");
  assert.equal(kernelStatus("KernelWorkerStatus.ERROR"), "failed");
  assert.equal(kernelStatus("KernelWorkerStatus.RUNNING"), "running");
  assert.equal(kernelStatus("KernelWorkerStatus.QUEUED"), "queued");
  assert.equal(kernelStatus("KernelWorkerStatus.CANCEL_ACKNOWLEDGED"), "failed");
  assert.equal(kernelStatus("warning: error log rotated\nKernelWorkerStatus.QUEUED"), "queued");
  assert.equal(kernelStatus("KernelWorkerStatus.CANCELLED"), "failed");
  assert.equal(kernelStatus("not complete yet"), "unknown");
});

test("consented remote chat queues, polls, and persists the Kaggle result", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-kaggle-job-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
    },
  });
  provider.runCli = async (args) => {
    if (args[1] === "push") {
      assert.equal(args.includes("--accelerator"), true);
      return { stdout: "Notebook pushed", stderr: "" };
    }
    if (args[1] === "status") return { stdout: "KernelWorkerStatus.COMPLETE", stderr: "" };
    if (args[1] === "output") {
      const outputDir = args[4];
      fs.mkdirSync(outputDir, { recursive: true });
      fs.writeFileSync(path.join(outputDir, "result.json"), JSON.stringify({ text: "remote result" }));
      return { stdout: "Downloaded", stderr: "" };
    }
    throw new Error(`Unexpected mocked Kaggle command: ${args.join(" ")}`);
  };
  try {
    const task = await provider.submitChat({
      model: "phi3:mini",
      messages: [{ role: "user", content: "test prompt" }],
    });
    for (let attempt = 0; attempt < 50 && provider.getJob(task.id).status !== "completed"; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    const completed = provider.getJob(task.id);
    assert.equal(completed.status, "completed");
    assert.deepEqual(completed.result, { text: "remote result" });
    const metadata = JSON.parse(fs.readFileSync(path.join(root, "kaggle", "jobs", task.id, "kernel-metadata.json"), "utf8"));
    assert.equal(metadata.is_private, "true");
    assert.equal(metadata.enable_gpu, "true");
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("queued Kaggle jobs are all submitted without waiting for earlier GPU jobs", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-parallel-submit-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
      KAGGLE_POLL_SECONDS: "0.01",
    },
  });
  const pushed = [];
  provider.runCli = async (args) => {
    if (args[1] === "push") {
      pushed.push(args[args.indexOf("-p") + 1]);
      return { stdout: "Notebook pushed", stderr: "" };
    }
    if (args[1] === "status") return { stdout: "KernelWorkerStatus.QUEUED", stderr: "" };
    throw new Error(`Unexpected mocked Kaggle command: ${args.join(" ")}`);
  };
  provider.pollMs = 5;
  try {
    const first = await provider.submitChat({
      model: "phi3:mini",
      messages: [{ role: "user", content: "first prompt" }],
    });
    const second = await provider.submitChat({
      model: "phi3:mini",
      messages: [{ role: "user", content: "second prompt" }],
    });
    for (let attempt = 0; attempt < 100
      && (pushed.length < 2 || provider.getJob(first.id).status !== "submitted" || provider.getJob(second.id).status !== "submitted");
      attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.equal(pushed.length, 2);
    assert.equal(provider.getJob(first.id).status, "submitted");
    assert.equal(provider.getJob(second.id).status, "submitted");
  } finally {
    provider.stop();
    for (let attempt = 0; attempt < 20 && provider.pumping; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("private Kaggle status visibility errors do not falsely fail a submitted notebook", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-kaggle-private-status-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
      KAGGLE_TIMEOUT_MINUTES: "120",
    },
  });
  provider.pollMs = 5;
  provider.visibilityTimeoutMs = 0;
  provider.timeoutMs = 5;
  provider.runCli = async (args) => {
    if (args[1] === "push") return { stdout: "Notebook pushed", stderr: "" };
    if (args[1] === "status") throw new Error("Permission 'kernels.get' was denied for private kernel");
    throw new Error(`Unexpected mocked Kaggle command: ${args.join(" ")}`);
  };
  try {
    const task = await provider.submitChat({
      model: "phi3:mini",
      messages: [{ role: "user", content: "private status check" }],
    });
    db.prepare("UPDATE compute_jobs SET created_at = datetime('now', '-10 minutes') WHERE id = ?").run(task.id);
    for (let attempt = 0; attempt < 100 && !provider.getJob(task.id).statusMessage; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    const current = provider.getJob(task.id);
    assert.equal(current.status, "submitted");
    assert.match(current.statusMessage, /CLI status is unavailable/);
    assert.equal(current.error, null);
    const confirmed = provider.confirmRemoteCompletion(task.id);
    assert.equal(confirmed.status, "remote_completed");
    assert.match(confirmed.statusMessage, /confirmed complete/);
    assert.equal(provider.stats().queued.some((item) => item.status === "remote_completed"), true);
  } finally {
    provider.stop();
    for (let attempt = 0; attempt < 20 && provider.pumping; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("a transient CLI status error keeps an accepted notebook submitted", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-kaggle-transient-status-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
    },
  });
  provider.pollMs = 5;
  provider.visibilityTimeoutMs = 60000;
  provider.timeoutMs = 60000;
  provider.runCli = async (args) => {
    if (args[1] === "push") return { stdout: "Notebook pushed", stderr: "" };
    if (args[1] === "status") throw new Error("Command failed: kaggle kernels status (socket hang up)");
    throw new Error(`Unexpected mocked Kaggle command: ${args.join(" ")}`);
  };
  try {
    const task = await provider.submitChat({
      model: "phi3:mini",
      messages: [{ role: "user", content: "transient status check" }],
    });
    for (let attempt = 0; attempt < 100 && !provider.getJob(task.id).statusMessage; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    const current = provider.getJob(task.id);
    assert.equal(current.status, "submitted");
    assert.equal(current.error, null);
    assert.match(current.statusMessage, /last status check failed/);
  } finally {
    provider.stop();
    for (let attempt = 0; attempt < 20 && provider.pumping; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("manual remote completion survives an in-flight private status failure", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-kaggle-completion-race-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
    },
  });
  let markStatusStarted;
  const statusStarted = new Promise((resolve) => { markStatusStarted = resolve; });
  let rejectStatus;
  const pendingStatus = new Promise((_resolve, reject) => { rejectStatus = reject; });
  provider.runCli = async (args) => {
    if (args[1] === "push") return { stdout: "Notebook pushed", stderr: "" };
    if (args[1] === "status") {
      markStatusStarted();
      return pendingStatus;
    }
    throw new Error(`Unexpected mocked Kaggle command: ${args.join(" ")}`);
  };
  try {
    const task = await provider.submitChat({
      model: "phi3:mini",
      messages: [{ role: "user", content: "private completion race" }],
    });
    await statusStarted;
    const confirmed = provider.confirmRemoteCompletion(task.id);
    assert.equal(confirmed.status, "remote_completed");
    rejectStatus(new Error("Permission 'kernels.get' was denied for private kernel"));
    for (let attempt = 0; attempt < 100 && provider.pumping; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.equal(provider.getJob(task.id).status, "remote_completed");
    assert.equal(provider.getJob(task.id).error, null);
  } finally {
    provider.stop();
    for (let attempt = 0; attempt < 20 && provider.pumping; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("consented image job downloads and records its SSD image path", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-image-job-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "Faiz000009",
    },
  });
  provider.pollMs = 5;
  let statusAttempts = 0;
  provider.runCli = async (args) => {
    if (args[1] === "push") return { stdout: "Notebook pushed", stderr: "" };
    if (args[1] === "status") {
      statusAttempts += 1;
      if (statusAttempts === 1) {
        throw new Error("Cannot access kernel (Permission 'kernels.get' was denied)");
      }
      return { stdout: "KernelWorkerStatus.COMPLETE", stderr: "" };
    }
    if (args[1] === "output") {
      const outputDir = args[4];
      fs.mkdirSync(outputDir, { recursive: true });
      fs.writeFileSync(path.join(outputDir, "unicorn-dragon.png"), Buffer.from("png-test"));
      fs.writeFileSync(path.join(outputDir, "result.json"), JSON.stringify({
        image: "unicorn-dragon.png",
        width: 512,
        height: 512,
        seed: 42,
      }));
      return { stdout: "Downloaded", stderr: "" };
    }
    throw new Error(`Unexpected mocked Kaggle command: ${args.join(" ")}`);
  };
  try {
    await assert.rejects(
      provider.submitImage({
        prompt: "too large",
        referenceImage: Buffer.alloc(300_000).toString("base64"),
      }),
      /optimized to 256 KiB/,
    );
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM compute_jobs").get().count, 0);
    const task = await provider.submitImage({
      prompt: "a unicorn dragon with red eyes and flaming lava",
      steps: 24,
      model: "Lykon/dreamshaper-8",
      candidates: 2,
      seed: 42,
      referenceImage: Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).toString("base64"),
      strength: 0.8,
    });
    for (let attempt = 0; attempt < 50 && provider.getJob(task.id).status !== "completed"; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    const completed = provider.getJob(task.id);
    assert.equal(completed.status, "completed");
    assert.equal(statusAttempts, 2);
    assert.equal(completed.kind, "image");
    assert.equal(completed.result.width, 512);
    assert.equal(fs.readFileSync(completed.result.imagePath).toString(), "png-test");
    const notebook = JSON.parse(fs.readFileSync(path.join(root, "kaggle", "jobs", task.id, "code.ipynb"), "utf8"));
    const notebookSource = notebook.cells[0].source.join("");
    const payloadEncoded = notebookSource.match(/payload = json\.loads\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/)?.[1];
    assert.ok(payloadEncoded);
    const payload = JSON.parse(Buffer.from(payloadEncoded, "base64").toString("utf8"));
    assert.equal(payload.referenceImage, Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).toString("base64"));
    assert.equal(payload.strength, 0.8);
    assert.equal(payload.model, "Lykon/dreamshaper-8");
    assert.equal(payload.candidates, 2);
    assert.ok(fs.statSync(path.join(root, "kaggle", "jobs", task.id, "code.ipynb")).size < 750 * 1024);
    const metadata = JSON.parse(fs.readFileSync(path.join(root, "kaggle", "jobs", task.id, "kernel-metadata.json"), "utf8"));
    assert.equal(metadata.id, `faiz000009/${metadata.id.split("/")[1]}`);
    assert.equal(metadata.title.toLowerCase().replaceAll(" ", "-"), metadata.id.split("/")[1]);
    assert.equal(metadata.machine_shape, "NvidiaTeslaT4");
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("video job pins public Wan weights and creates a private GPU notebook", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-job-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
    },
  });
  provider.pump = () => {};
  try {
    await assert.rejects(
      provider.submitVideo({ prompt: "too short for the selected frame rate", videoLength: 1, fps: 4, steps: 4, seed: 42 }),
      /between 8 and 33 requested frames/,
    );
    const task = await provider.submitVideo({
      prompt: "A short cinematic desert scene",
      videoLength: 2,
      fps: 8,
      steps: 8,
      seed: 42,
    });
    const job = provider.getJob(task.id);
    const directory = path.join(root, "kaggle", "jobs", task.id);
    const metadata = JSON.parse(fs.readFileSync(path.join(directory, "kernel-metadata.json"), "utf8"));
    const notebook = JSON.parse(fs.readFileSync(path.join(directory, "code.ipynb"), "utf8"));
    const source = notebook.cells[0].source.join("");
    const encodedPayload = source.match(/payload = json\.loads\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/)?.[1];
    assert.ok(encodedPayload);
    const payload = JSON.parse(Buffer.from(encodedPayload, "base64").toString("utf8"));
    assert.equal(task.status, "queued");
    assert.equal(job.kind, "video");
    assert.equal(metadata.is_private, "true");
    assert.equal(metadata.enable_gpu, "true");
    assert.equal(payload.frames, 17);
    assert.equal(payload.modelRepo, "Comfy-Org/Wan-2.1");
    assert.equal(payload.modelRevision, "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d");
    assert.equal(payload.modelBytes, 10_500_000_000);
    assert.equal(db.prepare("SELECT kind FROM compute_jobs WHERE id = ?").get(task.id).kind, "video");

    const fastTask = await provider.submitVideo({
      prompt: "A fast cinematic preview",
      videoLength: 2,
      fps: 4,
      steps: 4,
      seed: 43,
    });
    const fastNotebook = JSON.parse(fs.readFileSync(path.join(root, "kaggle", "jobs", fastTask.id, "code.ipynb"), "utf8"));
    const fastSource = fastNotebook.cells[0].source.join("");
    const fastPayloadMatch = fastSource.match(/payload = json\.loads\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/);
    assert.ok(fastPayloadMatch);
    assert.equal(JSON.parse(Buffer.from(fastPayloadMatch[1], "base64").toString("utf8")).frames, 9);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }

test("cancelJob deletes the remote kernel and marks a submitted job cancelled", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-cancel-job-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
    },
  });
  const deletedKernels = [];
  provider.runCli = async (args) => {
    if (args[1] === "push") return { stdout: "Notebook pushed", stderr: "" };
    if (args[1] === "delete") {
      deletedKernels.push(args[3]);
      return { stdout: "deleted", stderr: "" };
    }
    throw new Error(`Unexpected mocked Kaggle command: ${args.join(" ")}`);
  };
  try {
    const task = await provider.submitChat({
      model: "phi3:mini",
      messages: [{ role: "user", content: "cancel me" }],
    });
    for (let attempt = 0; attempt < 50 && provider.getJob(task.id).status !== "submitted"; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    const submittedJob = provider.getJob(task.id);
    assert.equal(submittedJob.status, "submitted");
    const cancelled = await provider.cancelJob(task.id);
    assert.equal(cancelled.status, "cancelled");
    assert.match(cancelled.error, /Cancelled by user/);
    assert.deepEqual(deletedKernels, [submittedJob.remoteId]);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("cancelJob rejects an already-completed task", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-cancel-completed-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: { KAGGLE_ENABLED: "true", KAGGLE_REMOTE_DATA_CONSENT: "true", KAGGLE_USERNAME: "test-user" },
  });
  provider.pump = () => {};
  try {
    const task = await provider.submitVideo({ prompt: "a short clip", videoLength: 2, fps: 8, steps: 8, seed: 1 });
    db.prepare("UPDATE compute_jobs SET status = 'completed' WHERE id = ?").run(task.id);
    await assert.rejects(provider.cancelJob(task.id), /Cannot cancel a completed task/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("KaggleProvider emits job-updated events as a submitted job progresses", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-job-events-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    env: { KAGGLE_ENABLED: "true", KAGGLE_REMOTE_DATA_CONSENT: "true", KAGGLE_USERNAME: "test-user" },
  });
  provider.runCli = async (args) => {
    if (args[1] === "push") return { stdout: "pushed", stderr: "" };
    if (args[1] === "status") return { stdout: "KernelWorkerStatus.COMPLETE", stderr: "" };
    if (args[1] === "output") {
      const outputDir = args[4];
      fs.mkdirSync(outputDir, { recursive: true });
      fs.writeFileSync(path.join(outputDir, "result.json"), JSON.stringify({ text: "ok" }));
      return { stdout: "downloaded", stderr: "" };
    }
    throw new Error(`Unexpected mocked Kaggle command: ${args.join(" ")}`);
  };
  const seenStatuses = [];
  provider.on("job-updated", (job) => seenStatuses.push(job.status));
  try {
    const task = await provider.submitChat({ model: "phi3:mini", messages: [{ role: "user", content: "events" }] });
    for (let attempt = 0; attempt < 50 && provider.getJob(task.id).status !== "completed"; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.ok(seenStatuses.includes("submitting"));
    assert.ok(seenStatuses.includes("submitted"));
    assert.ok(seenStatuses.includes("completed"));
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("removeVideoJob deletes completed video copies but preserves other task types", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-delete-video-"));
  const db = new DatabaseSync(":memory:");
  const provider = new KaggleProvider({ db, dataDir: root, env: { KAGGLE_ENABLED: "false" } });
  const id = "video-delete-test";
  try {
    db.prepare(`
      INSERT INTO compute_jobs (id, provider, kind, model, status, result_json)
      VALUES (?, 'kaggle', 'video', 'wan-t2v-1.3b', 'completed', '{}')
    `).run(id);
    for (const directory of [provider.jobsDir, provider.outputDir]) {
      fs.mkdirSync(path.join(directory, id), { recursive: true });
      fs.writeFileSync(path.join(directory, id, "artifact"), "local data");
    }

    assert.deepEqual(provider.removeVideoJob(id), { id, removed: true, localOnly: true });
    assert.equal(provider.getJob(id), null);
    assert.equal(fs.existsSync(path.join(provider.jobsDir, id)), false);
    assert.equal(fs.existsSync(path.join(provider.outputDir, id)), false);
    assert.throws(() => provider.removeVideoJob(id), /not found/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

});
