"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const {
  KaggleProvider,
  buildQwenWorkerNotebook,
  buildVideoWorkerNotebook,
  readVideoAssetDatasetMap,
} = require("../gateway/kaggle");
const { resolveVideoProfile } = require("../gateway/video-workflow");
const queue = require("../gateway/worker-queue");

function makeVideoProvider(root, db, extraEnv = {}, modelHub = null) {
  const provider = new KaggleProvider({
    db,
    dataDir: root,
    modelHub,
    env: {
      KAGGLE_ENABLED: "true",
      KAGGLE_REMOTE_DATA_CONSENT: "true",
      KAGGLE_USERNAME: "test-user",
      KAGGLE_MODEL_STORE_URL: "https://models.example.test",
      ...extraEnv,
    },
  });
  provider.fetchImpl = async () => new Response(
    JSON.stringify({ service: "local-ai-model-store", status: "ok" }),
    { status: 200 },
  );
  provider.pump = () => {};
  provider.runCli = async () => ({ stdout: "", stderr: "" });
  return provider;
}

function videoWorkerSource(notebook) {
  const cell = notebook.cells.find((candidate) => candidate.id === "local-ai-video-worker");
  assert.ok(cell, "video worker notebook should contain its long-running worker cell");
  return cell.source.join("");
}

test("video worker skips latent previews, stages model files in parallel, and polls quickly", () => {
  const profile = resolveVideoProfile({}, "wan22-fm-lightning-q8");
  const notebook = buildVideoWorkerNotebook({
    sessionId: "session-video-perf",
    token: "p".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile,
  });
  const source = videoWorkerSource(notebook);
  assert.match(source, /comfy_perf_args = \[\s*"--preview-method", "none",\s*"--disable-auto-launch",\s*"--disable-smart-memory",\s*"--cache-none",\s*\]/);
  assert.equal((source.match(/\*comfy_perf_args/g) || []).length, 2);
  assert.match(source, /ThreadPoolExecutor\(max_workers=min\(4, len\(model_download_items\)\)\)/);
  assert.match(source, /HF_HUB_ENABLE_HF_TRANSFER/);
  const embedded = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(source);
  assert.ok(embedded);
  const payload = JSON.parse(Buffer.from(embedded[1], "base64").toString("utf8"));
  assert.equal(payload.pollSeconds, 2);
  assert.match(source, /time\.sleep\(2\)\n    check_job_cancelled/);
});

test("video worker uses the active tunnel URL and refuses to submit when its health check fails", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-tunnel-"));
  const db = new DatabaseSync(":memory:");
  const urlFile = path.join(root, "logs", "model-store-url.txt");
  fs.mkdirSync(path.dirname(urlFile), { recursive: true });
  fs.writeFileSync(urlFile, "https://active-model-store.example.test");
  const provider = makeVideoProvider(root, db, { KAGGLE_EXPECTED_GPU_COUNT: "2" });
  let healthUrl = "";
  let pushes = 0;
  provider.fetchImpl = async (url) => {
    healthUrl = String(url);
    return new Response(JSON.stringify({ service: "local-ai-model-store", status: "ok" }), { status: 200 });
  };
  provider.runCli = async () => {
    pushes += 1;
    return { stdout: "", stderr: "" };
  };
  try {
    const session = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    const notebook = JSON.parse(fs.readFileSync(
      path.join(root, "kaggle", "jobs", session.id, "code.ipynb"),
      "utf8",
    ));
    const source = videoWorkerSource(notebook);
    const embedded = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(source);
    assert.ok(embedded);
    const payload = JSON.parse(Buffer.from(embedded[1], "base64").toString("utf8"));
    assert.equal(healthUrl, "https://active-model-store.example.test/health");
    assert.equal(payload.bridgeBaseUrl, "https://active-model-store.example.test");
    assert.equal(payload.expectedGpuCount, 2);
    assert.equal(pushes, 1);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }

  const failedRoot = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-dead-tunnel-"));
  const failedDb = new DatabaseSync(":memory:");
  const failedProvider = makeVideoProvider(failedRoot, failedDb);
  let submitted = false;
  failedProvider.fetchImpl = async () => new Response("unavailable", { status: 502 });
  failedProvider.runCli = async () => {
    submitted = true;
    return { stdout: "", stderr: "" };
  };
  try {
    await assert.rejects(
      failedProvider.startWorker({ kind: "video", model: "wan-t2v-1.3b" }),
      /Kaggle model-store tunnel is not reachable/,
    );
    assert.equal(submitted, false);
    assert.equal(failedProvider.listWorkers().length, 0);
  } finally {
    failedProvider.stop();
    failedDb.close();
    fs.rmSync(failedRoot, { recursive: true, force: true });
  }
});

test("buildVideoWorkerNotebook embeds the bridge loop and the resolved profile", () => {
  const profile = {
    id: "wan-t2v-1.3b",
    label: "Wan 2.1 T2V 1.3B",
    conditioning: "text",
    repo: "Comfy-Org/Wan-2.1",
    revision: "abc",
    diffusion: "wan2.1_t2v_1.3B_fp16.safetensors",
    textEncoder: "umt5.safetensors",
    vae: "wan_2.1_vae.safetensors",
    clipVision: null,
  };
  const notebook = buildVideoWorkerNotebook({
    sessionId: "session-video",
    token: "f".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile,
  });
  assert.equal(notebook.cells[0].id, "local-ai-video-worker-bootstrap");
  const bootstrapSource = notebook.cells[0].source.join("");
  assert.match(bootstrapSource, /sending the first Studio heartbeat/);
  assert.match(bootstrapSource, /urllib\.request\.urlopen/);
  const bootstrapPayloadMatch = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(bootstrapSource);
  assert.ok(bootstrapPayloadMatch);
  const bootstrapPayload = JSON.parse(Buffer.from(bootstrapPayloadMatch[1], "base64").toString("utf8"));
  assert.equal(bootstrapPayload.sessionId, "session-video");
  assert.equal(bootstrapPayload.bridgeBaseUrl, "https://models.example.test");
  const source = videoWorkerSource(notebook);
  assert.match(source, /2c75dc1b379a59e0880619d029607af5169a62ef/);
  assert.match(source, /\/v1\/workers\/\{session_id\}\/claim/);
  assert.match(source, /\/v1\/workers\/\{session_id\}\/jobs\/\{job_id\}\/complete/);
  assert.match(source, /\/v1\/workers\/\{session_id\}\/jobs\/\{job_id\}\/fail/);
  assert.match(source, /polling for jobs/);
  assert.match(source, /idle timeout/);
  assert.match(source, /last_job_at = time\.monotonic\(\)\n    while True:/);
  assert.match(source, /--require-hashes/);
  assert.match(source, /--no-deps/);
  assert.match(source, /"videos": \[\{"data"/);
  assert.match(source, /torch\.cuda\.get_device_properties/);
  assert.match(source, /expected_gpu_count/);
  assert.match(source, /nvidia-smi/);
  assert.match(source, /peakMemoryUsedMiB/);
  assert.match(source, /averageUtilizationPercent/);
  assert.match(source, /MemAvailable/);
  assert.match(source, /"hostMemory": dict\(self\.host_memory\)/);
  assert.match(source, /def live_summary\(self\):/);
  assert.match(source, /still running after \{elapsed_seconds\}s/);
  assert.match(source, /def comfyui_failure_context\(reason\):/);
  assert.match(source, /ComfyUI log tail:/);
  assert.match(source, /ComfyUI history request failed:/);
  assert.match(source, /ComfyUI exited while the job was running/);
  assert.match(source, /ComfyUI exit code: \{exit_code\}/);
  assert.match(source, /def restart_comfyui\(\):/);
  assert.match(source, /The Kaggle worker will remain connected/);
  assert.match(source, /for _ in range\(180\):\s+if worker_stop_requested\.is_set\(\):\s+return/);
  assert.match(source, /failure_summary = gpu_sampler\.live_summary\(\)/);
  assert.match(source, /graph = request\["graph"\]/);
  assert.match(
    source,
    /report_job_progress\(\s*job_id,\s*f"ComfyUI job \{job_id\} still running after \{elapsed_seconds\}s; "\s*\+\s*gpu_sampler\.live_summary\(\)\s*\+\s*"\\nComfyUI log tail:\\n" \+ comfyui_log_tail\(\),\s*\)/,
  );
  assert.match(source, /startup_heartbeat_loop/);
  assert.match(source, /\/jobs\/\{job_id\}\/progress/);
  assert.match(source, /def bridge_request_until_available\(operation, method, url, \*\*kwargs\):/);
  assert.match(source, /retry_delay = min\(retry_delay \* 2, 60\)/);
  assert.match(source, /bridge_request_until_available\(\s*"Checking worker job status"/);
  assert.match(source, /bridge_request_until_available\(\s*"Uploading completed video"/);
  assert.match(source, /def check_job_cancelled\(job_id, prompt_id\):/);
  assert.match(source, /def interrupt_comfyui\(prompt_id, reason\):/);
  assert.match(source, /base_url \+ "\/interrupt"/);
  assert.match(source, /report_job_progress\(job_id, "ComfyUI accepted the prompt; " \+ gpu_sampler\.live_summary\(\)\)\s+last_progress_report_at = time\.monotonic\(\)\s+while True:/);
  assert.doesNotMatch(source, /Video generation exceeded its .* execution limit/);
  assert.doesNotMatch(source, /deadline = time\.(?:time|monotonic)\(\) \+/);
  assert.match(source, /retained_fraction = min\(source_ratio \/ target_ratio, target_ratio \/ source_ratio\)/);
  assert.match(source, /Provide a scene-matched start frame with a compatible aspect ratio/);
  assert.match(source, /filename_prefix"\] = "job-" \+ job_id/);
  assert.match(source, /rglob\("job-" \+ job_id \+ "\*\.webm"\)/);
  assert.match(source, /"gpuTelemetry": gpu_telemetry/);
  assert.match(source, /"outputSha256": hashlib\.sha256\(video_bytes\)/);
  assert.match(
    source,
    /complete_response\.raise_for_status\(\)\s+for output in outputs:\s+try:\s+output\.unlink\(missing_ok=True\)\s+except OSError as error:/,
  );
  assert.match(
    source,
    /keyframe_filename = request\.get\("keyframeFilename"\)\s+if keyframe_filename:\s+keyframe_path = comfyui_dir \/ "input" \/ keyframe_filename\s+try:\s+keyframe_path\.unlink\(missing_ok=True\)\s+except OSError as error:/,
  );
  const embedded = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(source);
  assert.ok(embedded, "the notebook should carry its payload as base64");
  const payload = JSON.parse(Buffer.from(embedded[1], "base64").toString("utf8"));
  assert.equal(payload.sessionId, "session-video");
  assert.equal(payload.expectedGpuCount, 1);
  assert.equal(payload.profile.id, "wan-t2v-1.3b");
  assert.equal(payload.profile.diffusion, "wan2.1_t2v_1.3B_fp16.safetensors");
  assert.throws(() => buildVideoWorkerNotebook({
    sessionId: "session-invalid-gpus",
    token: "f".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile,
    expectedGpuCount: 0,
  }), /expectedGpuCount must be a positive safe integer/);
});

test("VACE worker notebook validates the reference-image graph nodes", () => {
  const notebook = buildVideoWorkerNotebook({
    sessionId: "session-vace",
    token: "v".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile: {
      id: "wan-vace-1.3b",
      conditioning: "image",
      imageConditioning: "vace-reference",
      engine: "comfyui",
      repo: "Comfy-Org/Wan-2.1",
      revision: "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
      diffusion: "wan2.1_vace_1.3B_fp16.safetensors",
      textEncoder: "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
      vae: "wan_2.1_vae.safetensors",
      clipVision: null,
      lora: {
        fileName: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
        downloadUrl: "https://huggingface.co/Evados/DiffSynth-Studio-Lora-Wan2.1-ComfyUI/resolve/7d504148050a87baa53fb1b6cccb6c894be149d/Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
        sha256: "BC5F39B3A6E55FCBF4F3C84806CB37B324996ADB0D2A6EE9E9B9789E23948515",
        sizeBytes: 175054688,
        strength: 3,
      },
    },
  });
  const source = videoWorkerSource(notebook);
  assert.match(source, /profile\.get\("imageConditioning"\) == "vace-reference"/);
  assert.match(source, /"WanVaceToVideo", "TrimVideoLatent"/);
  assert.match(source, /required_nodes\.add\("LoraLoaderModelOnly"\)/);
  assert.match(source, /hashlib\.sha256\(\)/);
  assert.match(source, /LoRA SHA-256 does not match the pinned Civitai model version/);
});

test("video worker stages the pinned Civitai Wan checkpoint through signed model-store ranges", () => {
  const notebook = buildVideoWorkerNotebook({
    sessionId: "session-civitai-video",
    token: "c".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile: {
      id: "wan-nsfw-t2v-1.3b",
      conditioning: "text",
      engine: "comfyui",
      repo: "Comfy-Org/Wan-2.1",
      revision: "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
      diffusion: "wan_1.3B_exp_e14.safetensors",
      textEncoder: "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
      vae: "wan_2.1_vae.safetensors",
      modelHubCheckpoint: {
        modelId: 1697081,
        versionId: 1936498,
        fileName: "wan_1.3B_exp_e14.safetensors",
        sizeBytes: 2838077328,
        sha256: "86DAED5F43A68D7BE2CA1F0E85D2AC51F3BA7FB735C76FD9574323E18FE40C6C",
      },
      modelSource: {
        baseUrl: "https://models.example.test",
        key: "civitai-1697081-1936498",
        expiresAt: 1234567890,
        signature: "d".repeat(64),
        maxBytes: 12 * 1024 ** 3,
      },
    },
  });
  const source = videoWorkerSource(notebook);
  assert.match(source, /checkpoints\/" \+ checkpoint\["fileName"\]/);
  assert.match(source, /Range/);
  assert.match(source, /Content-Range/);
  assert.match(source, /Wan checkpoint SHA-256 does not match the pinned Civitai model version/);
  assert.match(source, /checkpoint\.get\("targetDirectory", "diffusion_models"\)/);
  assert.doesNotMatch(source, /CIVITAI_API_KEY/);
});

test("Wan 2.2 SVI worker stages both pinned GGUFs, LoRAs, and scratch-backed assets", () => {
  const profile = resolveVideoProfile({}, "wan22-svi-q4km");
  profile.assetDataset = "test-user/wan22-svi-assets";
  const notebook = buildVideoWorkerNotebook({
    sessionId: "session-wan22-svi",
    token: "s".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile: {
      ...profile,
      modelSources: profile.modelHubCheckpoints.map((checkpoint, index) => ({
        checkpoint,
        modelSource: {
          baseUrl: "https://models.example.test",
          key: `civitai-${checkpoint.modelId}-${checkpoint.versionId}`,
          expiresAt: 1234567890,
          signature: String(index).repeat(64),
          maxBytes: 12 * 1024 ** 3,
        },
      })),
      videoLoraSource: {
        baseUrl: "https://models.example.test",
        key: "wan22-video-loras",
        expiresAt: 1234567890,
        signature: "l".repeat(64),
        maxBytes: 2 * 1024 ** 3,
      },
    },
  });
  const source = videoWorkerSource(notebook);
  assert.match(source, /pathlib\.Path\("\/kaggle\/tmp\/ComfyUI"\)/);
  assert.match(source, /profile\.get\("workspaceRoot"\)/);
  assert.match(source, /profile\.get\("modelSources"\)/);
  assert.match(source, /profile\.get\("assetDataset"\)/);
  assert.match(source, /def expected_cache_files\(asset_role\):/);
  assert.match(source, /def locate_cache_dataset\(dataset_id, asset_role\):/);
  assert.match(source, /os\.walk\(input_root, followlinks=True\)/);
  assert.match(source, /no mounted directory containing/);
  assert.match(source, /refusing a slow network fallback/);
  assert.match(source, /cache_root \/ pathlib\.Path\(item\["target"\]\)\.name/);
  assert.match(source, /if profile\.get\("assetDatasets"\):/);
  assert.match(source, /publisher-verified Kaggle cache for/);
  assert.match(source, /skipping repeat SHA-256 read/);
  assert.match(source, /Verifying cached model cache SHA-256/);
  assert.match(source, /Cache SHA-256 progress for/);
  assert.match(source, /Verified cached model cache SHA-256/);
  assert.match(source, /Cached model SHA-256 mismatch/);
  assert.match(source, /Staged .* onto local disk in/);
  assert.match(source, /"UnetLoaderGGUF", "LoraLoaderModelOnly", "ModelSamplingSD3"/);
  assert.match(source, /Wan checkpoint SHA-256 does not match the pinned Civitai model version/);
  assert.match(source, /signed video LoRA manifest does not match the pinned Civitai assets/);
  assert.match(source, /Video LoRA SHA-256 mismatch/);
  assert.match(source, /Model file SHA-256 mismatch/);
  assert.match(source, /ComfyUI-GGUF/);
  assert.match(source, /--require-hashes/);

  const embedded = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(source);
  assert.ok(embedded);
  const payload = JSON.parse(Buffer.from(embedded[1], "base64").toString("utf8"));
  assert.equal(payload.profile.modelSources.length, 2);
  assert.deepEqual(payload.profile.modelSources.map((item) => item.checkpoint.versionId), [2500306, 2500309]);
  assert.deepEqual(payload.profile.modelHubCheckpoints.map((item) => item.targetDirectory), ["unet", "unet"]);
  assert.match(payload.profile.extraModelFiles[0].target, /SVI_v2_PRO_Wan2\.2-I2V-A14B_HIGH/);
  assert.equal(payload.profile.videoLoraSource.key, "wan22-video-loras");
  assert.deepEqual(payload.profile.videoLoraFiles.map((item) => item.key), [
    "penis_insert", "penis_insert", "vag_detail", "vag_detail",
  ]);
});

test("dual-GPU worker installs pinned DisTorch2 only for the opt-in profile and requires two GPUs", () => {
  const profile = resolveVideoProfile({}, "wan22-fm-lightning-q8-dual-gpu");
  assert.throws(
    () => buildVideoWorkerNotebook({
      sessionId: "session-wan22-dual-gpu",
      token: "d".repeat(64),
      bridgeBaseUrl: "https://models.example.test",
      profile,
    }),
    /requires at least two visible GPUs/,
  );

  const notebook = buildVideoWorkerNotebook({
    sessionId: "session-wan22-dual-gpu",
    token: "d".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile,
    expectedGpuCount: 2,
  });
  const source = videoWorkerSource(notebook);
  assert.match(source, /extension\["repository"\]/);
  assert.match(source, /extension\["revision"\]/);
  assert.match(source, /Installed pinned ComfyUI-MultiGPU DisTorch2 loader/);
  assert.match(source, /UnetLoaderGGUFDisTorch2MultiGPU/);
  assert.match(source, /CLIPLoaderMultiGPU/);
  assert.match(source, /VAELoaderMultiGPU/);
  assert.match(source, /profile\.get\("dualGpuDistribution"\) == "distorch2"/);
  assert.match(source, /profile\["multiGpuExtension"\]/);
  const embedded = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(source);
  assert.ok(embedded);
  const payload = JSON.parse(Buffer.from(embedded[1], "base64").toString("utf8"));
  assert.equal(
    payload.profile.multiGpuExtension.repository,
    "https://github.com/pollockjj/ComfyUI-MultiGPU.git",
  );
  assert.equal(
    payload.profile.multiGpuExtension.revision,
    "37a73feda5804045e4f39d8c640905d62a2169d1",
  );
  assert.match(source, /profile\.get\("dualGpu"\)/);
  assert.match(
    source,
    /comfy_memory_args = \(\s*\[\]\s*if profile\["id"\]\.startswith\("wan22-fm-lightning-q8"\)\s*else \["--lowvram"\]\s*\)/,
  );
  assert.equal((source.match(/\*comfy_memory_args/g) || []).length, 2);
  assert.doesNotMatch(source, /--normalvram/);

  const legacyProfile = resolveVideoProfile(
    { KAGGLE_VIDEO_DUAL_GPU_DISTRIBUTION: "legacy" },
    "wan22-fm-lightning-q8-dual-gpu",
  );
  const legacyNotebook = buildVideoWorkerNotebook({
    sessionId: "session-wan22-dual-gpu-rollback",
    token: "r".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile: legacyProfile,
    expectedGpuCount: 2,
  });
  const legacySource = videoWorkerSource(legacyNotebook);
  const loaderSource = fs.readFileSync(
    path.join(__dirname, "..", "gateway", "dual_device_gguf_loader.py"),
    "utf8",
  );
  assert.ok(legacySource.includes(Buffer.from(loaderSource).toString("base64")));
  assert.match(legacySource, /Installed the opt-in dual-device GGUF loader/);

  const defaultProfile = resolveVideoProfile({}, "wan22-fm-lightning-q8");
  const defaultNotebook = buildVideoWorkerNotebook({
    sessionId: "session-wan22-single-gpu",
    token: "s".repeat(64),
    bridgeBaseUrl: "https://models.example.test",
    profile: defaultProfile,
  });
  assert.doesNotMatch(
    videoWorkerSource(defaultNotebook),
    /Installed the opt-in dual-device GGUF loader/,
  );
  assert.doesNotMatch(
    videoWorkerSource(defaultNotebook),
    /Installed pinned ComfyUI-MultiGPU DisTorch2 loader/,
  );
});

test("starting the dual-GPU video profile is rejected when fewer than two GPUs are expected", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-dual-gpu-preflight-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db, { KAGGLE_EXPECTED_GPU_COUNT: "1" });
  try {
    await assert.rejects(
      provider.startWorker({ kind: "video", model: "wan22-fm-lightning-q8-dual-gpu" }),
      /requires KAGGLE_EXPECTED_GPU_COUNT=2 or greater/,
    );
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("dual-GPU worker reuses the standard Q8 profile's published asset cache", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-dual-gpu-cache-"));
  const hubRoot = path.join(root, "model-hub");
  const profile = resolveVideoProfile({}, "wan22-fm-lightning-q8-dual-gpu");
  const importedByFilename = new Map(profile.modelHubCheckpoints.map((checkpoint) => {
    const folderKey = `civitai-${checkpoint.modelId}-${checkpoint.versionId}`;
    const checkpointDir = path.join(hubRoot, folderKey, "checkpoints");
    fs.mkdirSync(checkpointDir, { recursive: true });
    fs.writeFileSync(path.join(checkpointDir, checkpoint.fileName), "test checkpoint");
    return [checkpoint.fileName, {
      status: "complete",
      modelId: checkpoint.modelId,
      versionId: checkpoint.versionId,
      expectedBytes: checkpoint.sizeBytes,
      folderKey,
    }];
  }));
  const loraRoot = path.join(hubRoot, "wan22-video-loras");
  for (const lora of profile.videoLoraFiles) {
    const filePath = path.join(loraRoot, ...lora.target.split("/"));
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    const fd = fs.openSync(filePath, "w");
    fs.ftruncateSync(fd, lora.sizeBytes);
    fs.closeSync(fd);
  }
  const modelHub = {
    hubRoot,
    infoForFilename(fileName) {
      return importedByFilename.get(fileName) || null;
    },
  };
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db, {
    KAGGLE_VIDEO_MAX_MODEL_GIB: "40",
    KAGGLE_EXPECTED_GPU_COUNT: "2",
    KAGGLE_VIDEO_ASSET_DATASETS: JSON.stringify({
      "wan22-fm-lightning-q8": {
        checkpoints: "test-user/wan22-q8-checkpoints",
        modelFiles: "test-user/wan22-common-assets",
      },
    }),
    KAGGLE_MODEL_TRANSFER_CONSENT: "true",
    KAGGLE_MODEL_STORE_SECRET: "s".repeat(32),
    MODEL_HUB_ROOT: hubRoot,
  }, modelHub);
  try {
    const session = await provider.startWorker({
      kind: "video",
      model: "wan22-fm-lightning-q8-dual-gpu",
    });
    const notebookDir = path.join(root, "kaggle", "jobs", session.id);
    const metadata = JSON.parse(
      fs.readFileSync(path.join(notebookDir, "kernel-metadata.json"), "utf8"),
    );
    assert.deepEqual(metadata.dataset_sources, [
      "test-user/wan22-q8-checkpoints",
      "test-user/wan22-common-assets",
    ]);
    const notebook = JSON.parse(fs.readFileSync(path.join(notebookDir, "code.ipynb"), "utf8"));
    const source = videoWorkerSource(notebook);
    const embedded = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(source);
    assert.ok(embedded);
    const payload = JSON.parse(Buffer.from(embedded[1], "base64").toString("utf8"));
    assert.equal(payload.profile.id, "wan22-fm-lightning-q8-dual-gpu");
    assert.deepEqual(payload.profile.assetDatasets, {
      checkpoints: "test-user/wan22-q8-checkpoints",
      modelFiles: "test-user/wan22-common-assets",
    });
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("startWorker with kind video records a video session and writes its notebook", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  try {
    const session = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    assert.equal(session.kind, "video");
    assert.equal(session.model, "wan-t2v-1.3b");
    assert.equal(session.status, "starting");
    assert.match(session.remoteId, /^test-user\/wan-worker-/);
    const notebook = JSON.parse(
      fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "code.ipynb"), "utf8"),
    );
    assert.equal(notebook.cells[0].id, "local-ai-video-worker-bootstrap");
    assert.match(videoWorkerSource(notebook), /payload = json\.loads/);
    const metadata = JSON.parse(
      fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "kernel-metadata.json"), "utf8"),
    );
    assert.equal(metadata.machine_shape, "NvidiaTeslaT4");
    assert.equal(metadata.title.toLowerCase().replaceAll(" ", "-"), metadata.id.split("/")[1]);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("startVideoWorker reuses the active session for the requested profile", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-reuse-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  const pushes = [];
  provider.runCli = async (args) => {
    pushes.push(args);
    return { stdout: "", stderr: "" };
  };
  try {
    const first = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    const reused = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    assert.equal(reused.id, first.id);
    assert.equal(pushes.length, 1);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("startVideoWorker creates a separate session when explicitly requested", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-instance-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  const pushes = [];
  provider.runCli = async (args) => {
    pushes.push(args);
    return { stdout: "", stderr: "" };
  };
  try {
    const first = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    const second = await provider.startWorker({
      kind: "video",
      model: "wan-t2v-1.3b",
      newInstance: true,
    });
    assert.notEqual(second.id, first.id);
    assert.equal(pushes.length, 2);
    assert.equal(queue.listSessions(db).filter((session) => session.kind === "video").length, 2);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("startVideoWorker refuses to launch a second active profile", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-conflict-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  const pushes = [];
  provider.runCli = async (args) => {
    pushes.push(args);
    return { stdout: "", stderr: "" };
  };
  try {
    await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    await assert.rejects(
      provider.startWorker({ kind: "video", model: "wan22-svi-q4km" }),
      (error) => error.status === 409 && /different video worker is already active/.test(error.message),
    );
    assert.equal(pushes.length, 1);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("startVideoWorker ignores stale stopping sessions and starts a fresh worker", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-stopping-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  const pushes = [];
  provider.runCli = async (args) => {
    pushes.push(args);
    return { stdout: "", stderr: "" };
  };
  try {
    queue.createSession(db, {
      id: "stale-video-session",
      token: "token-stale",
      kind: "video",
      model: "wan22-svi-cf-q4km",
      remoteId: "test-user/wan-worker-stale",
      idleTimeoutSeconds: 300,
      maxRuntimeSeconds: 3600,
    });
    queue.requestStop(db, "stale-video-session");
    const started = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    assert.equal(started.model, "wan-t2v-1.3b");
    assert.equal(pushes.length, 1);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("Civitai video profile requires the exact imported checkpoint and signs its Model Hub source", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-civitai-video-worker-"));
  const hubRoot = path.join(root, "model-hub");
  const folderKey = "civitai-1697081-1936498";
  const checkpointDir = path.join(hubRoot, folderKey, "checkpoints");
  fs.mkdirSync(checkpointDir, { recursive: true });
  fs.writeFileSync(path.join(checkpointDir, "wan_1.3B_exp_e14.safetensors"), "test checkpoint");
  const modelHub = {
    hubRoot,
    infoForFilename(fileName) {
      if (fileName !== "wan_1.3B_exp_e14.safetensors") return null;
      return {
        status: "complete",
        modelId: 1697081,
        versionId: 1936498,
        expectedBytes: 2838077328,
        folderKey,
      };
    },
  };
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db, {
    KAGGLE_MODEL_TRANSFER_CONSENT: "true",
    KAGGLE_MODEL_STORE_SECRET: "s".repeat(32),
    MODEL_HUB_ROOT: hubRoot,
  }, modelHub);
  try {
    const session = await provider.startWorker({
      kind: "video",
      model: "wan-nsfw-t2v-1.3b",
    });
    assert.equal(session.model, "wan-nsfw-t2v-1.3b");
    const notebook = JSON.parse(
      fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "code.ipynb"), "utf8"),
    );
    const source = videoWorkerSource(notebook);
    const embedded = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(source);
    assert.ok(embedded);
    const payload = JSON.parse(Buffer.from(embedded[1], "base64").toString("utf8"));
    assert.equal(payload.profile.modelSource.key, folderKey);
    assert.equal(payload.profile.modelHubCheckpoint.versionId, 1936498);
    assert.equal(payload.profile.modelHubCheckpoint.sha256, "86DAED5F43A68D7BE2CA1F0E85D2AC51F3BA7FB735C76FD9574323E18FE40C6C");
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("video asset cache map validates profile roles and dataset ownership", () => {
  assert.deepEqual(readVideoAssetDatasetMap(JSON.stringify({
    "wan22-svi-q4km": {
      checkpoints: "Test-User/fast-move",
      modelFiles: "test-user/shared-assets",
    },
  }), "test-user"), {
    "wan22-svi-q4km": {
      checkpoints: "test-user/fast-move",
      modelFiles: "test-user/shared-assets",
    },
  });
  assert.throws(
    () => readVideoAssetDatasetMap(JSON.stringify({
      "wan22-svi-q4km": { checkpoints: "other-user/fast-move" },
    }), "test-user"),
    /owned by test-user/,
  );
  assert.deepEqual(readVideoAssetDatasetMap(JSON.stringify({
    "wan22-svi-q4km": { checkpoints: "cache-owner/fast-move" },
  }), "test-user", ["cache-owner"]), {
    "wan22-svi-q4km": { checkpoints: "cache-owner/fast-move" },
  });
  assert.throws(
    () => readVideoAssetDatasetMap(JSON.stringify({
      "wan22-svi-q4km": { checkpoints: "unlisted-owner/fast-move" },
    }), "test-user", ["cache-owner"]),
    /trusted cache owner/,
  );
});

test("Wan 2.2 SVI worker requires both exact Model Hub imports and signs each source", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-wan22-svi-worker-"));
  const hubRoot = path.join(root, "model-hub");
  const profile = resolveVideoProfile({}, "wan22-svi-cf-q4km");
  const loraRoot = path.join(hubRoot, "wan22-video-loras");
  for (const lora of profile.videoLoraFiles) {
    const filePath = path.join(loraRoot, ...lora.target.split("/"));
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    const fd = fs.openSync(filePath, "w");
    fs.ftruncateSync(fd, lora.sizeBytes);
    fs.closeSync(fd);
  }
  const importedByFilename = new Map(profile.modelHubCheckpoints.map((checkpoint) => {
    const folderKey = `civitai-${checkpoint.modelId}-${checkpoint.versionId}`;
    const checkpointDir = path.join(hubRoot, folderKey, "checkpoints");
    fs.mkdirSync(checkpointDir, { recursive: true });
    fs.writeFileSync(path.join(checkpointDir, checkpoint.fileName), "test checkpoint");
    return [checkpoint.fileName, {
      status: "complete",
      modelId: checkpoint.modelId,
      versionId: checkpoint.versionId,
      expectedBytes: checkpoint.sizeBytes,
      folderKey,
    }];
  }));
  const modelHub = {
    hubRoot,
    infoForFilename(fileName) {
      return importedByFilename.get(fileName) || null;
    },
  };
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db, {
    KAGGLE_VIDEO_MAX_MODEL_GIB: "32",
    KAGGLE_VIDEO_ASSET_DATASETS: JSON.stringify({
      "wan22-svi-cf-q4km": {
        checkpoints: "test-user/wan22-svi-checkpoints",
        modelFiles: "test-user/wan22-svi-shared",
      },
    }),
    KAGGLE_MODEL_TRANSFER_CONSENT: "true",
    KAGGLE_MODEL_STORE_SECRET: "s".repeat(32),
    MODEL_HUB_ROOT: hubRoot,
  }, modelHub);
  try {
    const session = await provider.startWorker({ kind: "video", model: "wan22-svi-cf-q4km" });
    assert.equal(session.model, "wan22-svi-cf-q4km");
    const notebook = JSON.parse(
      fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "code.ipynb"), "utf8"),
    );
    const metadata = JSON.parse(
      fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "kernel-metadata.json"), "utf8"),
    );
    assert.deepEqual(metadata.dataset_sources, [
      "test-user/wan22-svi-checkpoints",
      "test-user/wan22-svi-shared",
    ]);
    const source = videoWorkerSource(notebook);
    const embedded = /base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/.exec(source);
    assert.ok(embedded);
    const payload = JSON.parse(Buffer.from(embedded[1], "base64").toString("utf8"));
    assert.deepEqual(payload.profile.assetDatasets, {
      checkpoints: "test-user/wan22-svi-checkpoints",
      modelFiles: "test-user/wan22-svi-shared",
    });
    assert.equal(payload.profile.modelSources.length, 2);
    assert.deepEqual(payload.profile.modelSources.map((item) => item.checkpoint.versionId), [2625965, 2625966]);
    assert.deepEqual(payload.profile.modelSources.map((item) => item.modelSource.key), [
      "civitai-2053259-2625965",
      "civitai-2053259-2625966",
    ]);
    assert.equal(payload.profile.videoLoraSource.key, "wan22-video-loras");
    assert.equal(payload.profile.videoLoraFiles.length, 4);
    assert.equal(payload.profile.videoLoraFiles[0].sha256, "2267F8991524408099BC3B71664E1E2EEE2CEC82554976B3F6D8A358D456AF35");

    const missingDb = new DatabaseSync(":memory:");
    const missingProvider = makeVideoProvider(root, missingDb, {
      KAGGLE_VIDEO_MAX_MODEL_GIB: "32",
      KAGGLE_VIDEO_ASSET_DATASETS: JSON.stringify({
        "wan22-svi-cf-q4km": {
          checkpoints: "test-user/wan22-svi-checkpoints",
          modelFiles: "test-user/wan22-svi-shared",
        },
      }),
      KAGGLE_MODEL_TRANSFER_CONSENT: "true",
      KAGGLE_MODEL_STORE_SECRET: "s".repeat(32),
      MODEL_HUB_ROOT: hubRoot,
    }, {
      hubRoot,
      infoForFilename: () => null,
    });

    test("video queue builds per-shot LoRAs into both SVI model branches", () => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-svi-job-"));
      const db = new DatabaseSync(":memory:");
      const provider = makeVideoProvider(root, db, { KAGGLE_VIDEO_MAX_MODEL_GIB: "32" });
      try {
        queue.createSession(db, {
          id: "svi-session",
          token: "s".repeat(64),
          kind: "video",
          model: "wan22-svi-cf-q4km",
          remoteId: "test-user/wan-worker",
          idleTimeoutSeconds: 900,
          maxRuntimeSeconds: 21600,
        });
        const job = provider.enqueueVideoWorkerJob("svi-session", {
          prompt: "a shot",
          keyframeBase64: Buffer.from("keyframe").toString("base64"),
          frames: 41,
          fps: 8,
          steps: 8,
          loras: ["penis_insert", "vag_detail"],
        });
        const row = db.prepare("SELECT request_json FROM compute_jobs WHERE id = ?").get(job.id);
        const request = JSON.parse(row.request_json);
        assert.deepEqual(request.loras, ["penis_insert", "vag_detail"]);
        assert.deepEqual(request.graph["5"].inputs.model, ["19", 0]);
        assert.deepEqual(request.graph["6"].inputs.model, ["20", 0]);
        assert.equal(request.graph["17"].inputs.strength_model, 0.9);
        assert.equal(request.graph["19"].inputs.strength_model, 0.75);
      } finally {
        provider.stop();
        db.close();
        fs.rmSync(root, { recursive: true, force: true });
      }
    });
    try {
      await assert.rejects(
        missingProvider.startWorker({ kind: "video", model: "wan22-svi-cf-q4km" }),
        /Import Civitai model 2053259, version 2625965/,
      );
    } finally {
      missingProvider.stop();
      missingDb.close();
    }
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("CogVideoX worker metadata has a Kaggle-compatible kernel title", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-cogvideox-worker-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  try {
    const session = await provider.startWorker({ kind: "video", model: "cogvideox-2b" });
    const metadata = JSON.parse(
      fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "kernel-metadata.json"), "utf8"),
    );
    assert.equal(session.model, "cogvideox-2b");
    assert.equal(metadata.id.split("/")[1], `cogvideox-worker-${session.id.replaceAll("-", "").slice(0, 12)}`);
    assert.equal(metadata.title.toLowerCase().replaceAll(" ", "-"), metadata.id.split("/")[1]);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("a failed Kaggle kernel push marks its worker session failed", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-push-fail-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  provider.runCli = async () => {
    throw new Error("Kaggle rejected the kernel metadata");
  };
  try {
    await assert.rejects(
      provider.startWorker({ kind: "video", model: "cogvideox-2b" }),
      /Kaggle rejected the kernel metadata/,
    );
    assert.equal(provider.listWorkers()[0].status, "failed");
    assert.match(provider.listWorkers()[0].error, /Kaggle rejected/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("syncWorkerStatus reconciles a failed Kaggle kernel and fails its queued jobs", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-status-fail-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  try {
    const session = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    queue.enqueueJob(db, {
      id: "queued-video-job",
      sessionId: session.id,
      kind: "video",
      model: session.model,
      requestJson: { prompt: "temporary test job" },
    });
    provider.runCli = async (args) => {
      assert.deepEqual(args, ["kernels", "status", session.remoteId]);
      return { stdout: "KernelWorkerStatus.ERROR", stderr: "" };
    };

    const current = await provider.syncWorkerStatus(session.id);
    assert.equal(current.status, "failed");
    assert.equal(current.remoteStatus, "failed");
    assert.match(current.error, /Open its Kaggle notebook output/);
    assert.equal(db.prepare("SELECT status FROM compute_jobs WHERE id = 'queued-video-job'").get().status, "failed");
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("syncWorkerStatus reports a running Kaggle kernel and status-query errors", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-status-running-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  try {
    const session = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    provider.runCli = async () => ({ stdout: "KernelWorkerStatus.RUNNING", stderr: "" });
    const current = await provider.syncWorkerStatus(session.id);
    assert.equal(current.status, "starting");
    assert.equal(current.remoteStatus, "running");

    provider.runCli = async () => {
      throw new Error("Kaggle CLI is temporarily unavailable");
    };
    const unavailable = await provider.syncWorkerStatus(session.id);
    assert.equal(unavailable.status, "starting");
    assert.match(unavailable.statusSyncError, /temporarily unavailable/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("video worker refuses a loopback bridge URL before submitting to Kaggle", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-worker-loopback-"));
  const db = new DatabaseSync(":memory:");
  let submitted = false;
  const provider = makeVideoProvider(root, db, {
    KAGGLE_MODEL_STORE_URL: "http://127.0.0.1:3988",
  });
  provider.runCli = async () => {
    submitted = true;
    return { stdout: "", stderr: "" };
  };
  try {
    await assert.rejects(
      provider.startWorker({ kind: "video" }),
      /KAGGLE_MODEL_STORE_URL must use HTTPS/,
    );
    assert.equal(submitted, false);
    const workerFiles = fs.existsSync(provider.jobsDir) ? fs.readdirSync(provider.jobsDir) : [];
    assert.deepEqual(workerFiles, []);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("enqueueVideoWorkerJob stores a ready-to-run graph and claims as a video job", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-job-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  try {
    const session = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    const job = provider.enqueueVideoWorkerJob(session.id, {
      prompt: "a wide shot of the harbour at dawn",
      frames: 9,
      fps: 8,
      steps: 8,
      seed: 99,
      priority: 5,
      story: { sceneId: "scene-1" },
    });
    assert.equal(job.kind, "video");
    assert.equal(job.provider, "kaggle-worker");

    const row = db.prepare("SELECT kind, priority, request_json FROM compute_jobs WHERE id = ?").get(job.id);
    assert.equal(row.kind, "video");
    assert.equal(row.priority, 5);
    const request = JSON.parse(row.request_json);
    assert.equal(request.seed, 99);
    assert.equal(request.frames, 9);
    assert.equal(request.graph["9"].class_type, "SaveWEBM");
    assert.equal(request.graph["6"].class_type, "EmptyHunyuanLatentVideo");
    assert.equal(request.graph["6"].inputs.length, 9);
    assert.deepEqual(request.story, { sceneId: "scene-1" });

    const claimed = queue.claimNextJob(db, session.id);
    assert.equal(claimed.kind, "video");
    assert.equal(claimed.id, job.id);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("an opt-in I2V video session requires and stores a keyframe", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-i2v-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db, {
    KAGGLE_VIDEO_I2V_ENABLED: "true",
    KAGGLE_VIDEO_MAX_MODEL_GIB: "40",
  });
  try {
    const session = await provider.startWorker({ kind: "video" });
    assert.equal(session.model, "wan-i2v-14b-480p-fp8");
    assert.throws(
      () => provider.enqueueVideoWorkerJob(session.id, { prompt: "a shot" }),
      /keyframeBase64 is required/,
    );
    const keyframe = Buffer.from("keyframe-bytes").toString("base64");
    const job = provider.enqueueVideoWorkerJob(session.id, {
      prompt: "a shot",
      keyframeBase64: keyframe,
      frames: 9,
      fps: 8,
    });
    const row = db.prepare("SELECT request_json FROM compute_jobs WHERE id = ?").get(job.id);
    const request = JSON.parse(row.request_json);
    assert.equal(request.keyframeBase64, keyframe);
    assert.equal(request.graph["11"].class_type, "LoadImage");
    assert.equal(request.graph["11"].inputs.image, `keyframe-${job.id}.png`);
    assert.equal(request.graph["8"].class_type, "WanImageToVideo");
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("a VACE video session queues a reference-image graph with the pinned profile", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-vace-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db, { KAGGLE_VIDEO_PROFILE: "wan-vace-1.3b" });
  try {
    const session = await provider.startWorker({ kind: "video" });
    assert.equal(session.model, "wan-vace-1.3b");
    const keyframe = Buffer.from("reference-image").toString("base64");
    const job = provider.enqueueVideoWorkerJob(session.id, {
      prompt: "a lantern bearer walks through a misty forest",
      negativePrompt: "blur, flicker",
      keyframeBase64: keyframe,
      frames: 17,
      fps: 8,
      steps: 8,
    });
    const row = db.prepare("SELECT request_json FROM compute_jobs WHERE id = ?").get(job.id);
    const request = JSON.parse(row.request_json);
    assert.equal(request.graph["6"].class_type, "WanVaceToVideo");
    assert.equal(request.graph["12"].class_type, "TrimVideoLatent");
    assert.deepEqual(request.graph["6"].inputs.reference_image, ["11", 0]);
    assert.equal(request.keyframeBase64, keyframe);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("a video worker rejects image jobs and an image session rejects video jobs", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-mismatch-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  try {
    const videoSession = await provider.startWorker({ kind: "video", model: "wan-t2v-1.3b" });
    assert.throws(
      () => provider.enqueueWorkerJob(videoSession.id, { prompt: "cat" }),
      /video worker/,
    );

    const imageSession = queue.createSession(db, {
      id: "image-session",
      token: "a".repeat(64),
      kind: "image",
      model: "qwen-image",
      remoteId: "test-user/qwen-worker",
      idleTimeoutSeconds: 900,
      maxRuntimeSeconds: 21600,
    });
    await assert.rejects(
      async () => provider.enqueueVideoWorkerJob(imageSession.id, { prompt: "cat" }),
      /not a video worker/,
    );
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("Krea-2 scene rebuild jobs carry up to three separately labeled references", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-multiref-scene-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeVideoProvider(root, db);
  const referenceData = Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    Buffer.from("synthetic"),
  ]).toString("base64");
  try {
    const session = queue.createSession(db, {
      id: "krea-scene-worker",
      token: "k".repeat(64),
      kind: "image",
      model: "Ultra Realistic by Stable Yogi Krea2 V1.0 Civ INT8-ConvRot C89.safetensors",
      remoteId: "test-user/krea-scene-worker",
      idleTimeoutSeconds: 900,
      maxRuntimeSeconds: 21600,
    });
    const job = provider.enqueueWorkerJob(session.id, {
      prompt: "rebuild the scene with two distinct fictional characters",
      referenceImages: [
        { data: referenceData, label: "character A" },
        { data: referenceData, label: "character B" },
        { data: referenceData, label: "continuity scene" },
      ],
    });
    const request = JSON.parse(
      db.prepare("SELECT request_json FROM compute_jobs WHERE id = ?").get(job.id).request_json,
    );
    assert.deepEqual(request.referenceImages.map(({ filename, label }) => [filename, label]), [
      ["worker_reference_image_1.png", "character A"],
      ["worker_reference_image_2.png", "character B"],
      ["worker_reference_image_3.png", "continuity scene"],
    ]);
    assert.deepEqual(
      [request.graph["4"].inputs.image1, request.graph["4"].inputs.image2, request.graph["4"].inputs.image3],
      [["4img1", 0], ["4img2", 0], ["4img3", 0]],
    );

    const notebook = buildQwenWorkerNotebook({
      sessionId: session.id,
      token: "k".repeat(64),
      model: session.model,
      bridgeBaseUrl: "https://models.example.test",
    });
    const source = notebook.cells[0].source.join("");
    assert.match(source, /job_request\.get\("referenceImages"\)/);
    assert.match(source, /for reference in references:/);
    assert.match(source, /reference\["filename"\]/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("claimNextJob orders by priority and never mixes worker kinds", () => {
  const db = new DatabaseSync(":memory:");
  try {
    queue.createSession(db, {
      id: "v", token: "v".repeat(64), kind: "video", model: "wan", remoteId: "u/v",
      idleTimeoutSeconds: 900, maxRuntimeSeconds: 21600,
    });
    queue.enqueueJob(db, { id: "low", sessionId: "v", kind: "video", model: "wan", requestJson: {}, priority: 0 });
    queue.enqueueJob(db, { id: "high", sessionId: "v", kind: "video", model: "wan", requestJson: {}, priority: 10 });
    queue.enqueueJob(db, { id: "wrong-kind", sessionId: "v", kind: "image", model: "qwen", requestJson: {} });
    assert.equal(queue.claimNextJob(db, "v").id, "high");
    assert.equal(queue.claimNextJob(db, "v").id, "low");
    assert.equal(queue.claimNextJob(db, "v"), null);
  } finally {
    db.close();
  }
});

test("failQueuedJobsForSession only closes that session's unfinished jobs", () => {
  const db = new DatabaseSync(":memory:");
  try {
    queue.createSession(db, {
      id: "a", token: "a".repeat(64), model: "m", remoteId: "u/a",
      idleTimeoutSeconds: 900, maxRuntimeSeconds: 21600,
    });
    queue.createSession(db, {
      id: "b", token: "b".repeat(64), model: "m", remoteId: "u/b",
      idleTimeoutSeconds: 900, maxRuntimeSeconds: 21600,
    });
    queue.enqueueJob(db, { id: "a1", sessionId: "a", kind: "image", model: "m", requestJson: {} });
    queue.enqueueJob(db, { id: "b1", sessionId: "b", kind: "image", model: "m", requestJson: {} });
    const failed = queue.failQueuedJobsForSession(db, "a", "worker gone");
    assert.deepEqual(failed, ["a1"]);
    assert.equal(db.prepare("SELECT status FROM compute_jobs WHERE id = 'a1'").get().status, "failed");
    assert.equal(db.prepare("SELECT status FROM compute_jobs WHERE id = 'b1'").get().status, "queued");
  } finally {
    db.close();
  }
});
