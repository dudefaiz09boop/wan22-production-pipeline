const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { execFileSync } = require("node:child_process");
const { DatabaseSync } = require("node:sqlite");
const express = require("express");
const http = require("node:http");
const {
  KaggleProvider,
  buildQwen21GgufWorkerNotebook,
  buildQwenWorkerNotebook,
} = require("../gateway/kaggle");
const { mountWorkerBridge } = require("../gateway/worker-bridge");
const queue = require("../gateway/worker-queue");

const KREA2_CHECKPOINT = "Ultra Realistic by Stable Yogi Krea2 V1.0 Civ INT8-ConvRot C89.safetensors";

function makeProvider(root, db, extraEnv = {}, modelHub = null) {
  const modelDir = path.join(root, "models", "qwen-image-21");
  fs.mkdirSync(modelDir, { recursive: true });
  fs.writeFileSync(path.join(modelDir, "qwen-image-2.1-step90000-FP8-comfy.safetensors"), "weights");
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
      KAGGLE_MODEL_STORE_MAP: JSON.stringify({ "qwen-image-21": modelDir }),
      KAGGLE_IMAGE_MODEL_KEY_MAP: JSON.stringify({
        "qwen-image-2.1-step90000-FP8-comfy.safetensors": "qwen-image-21",
      }),
      ...extraEnv,
    },
    modelHub,
  });
  provider.fetchImpl = async () => new Response(
    JSON.stringify({ service: "local-ai-model-store", status: "ok" }),
    { status: 200 },
  );
  provider.pump = () => {};
  return provider;
}

test("startWorker discovers staged Model Hub checkpoint roots without an env map", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-hub-root-"));
  const db = new DatabaseSync(":memory:");
  const hubRoot = path.join(root, "model-hub");
  const folderKey = "civitai-2741166-3425119";
  const filename = "Muse By Stable Yogi Krea2 V4.0 Extended CIV INT8-ConvRot C86.safetensors";
  const checkpointPath = path.join(hubRoot, folderKey, "checkpoints", filename);
  fs.mkdirSync(path.dirname(checkpointPath), { recursive: true });
  fs.writeFileSync(checkpointPath, "staged muse checkpoint");
  const modelHub = {
    hubRoot,
    folderKeyForFilename: (candidate) => candidate === filename ? folderKey : null,
    infoForFilename: (candidate) => candidate === filename ? { family: "krea2" } : null,
  };
  const provider = makeProvider(root, db, {
    KAGGLE_IMAGE_ASSET_DATASET: "test-user/krea2-image-model-cache",
  }, modelHub);
  provider.runCli = async () => ({ stdout: "", stderr: "" });
  try {
    const session = await provider.startWorker({ model: filename });
    assert.equal(session.status, "starting");
    const notebook = JSON.parse(fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "code.ipynb"), "utf8"));
    const source = notebook.cells[0].source.join("");
    const metadata = JSON.parse(fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "kernel-metadata.json"), "utf8"));
    const encodedPayload = source.match(/payload = json\.loads\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)/)?.[1];
    assert.ok(encodedPayload);
    const payload = JSON.parse(Buffer.from(encodedPayload, "base64").toString("utf8"));
    assert.equal(payload.modelSource.key, folderKey);
    assert.equal(payload.family, "krea2");
    assert.equal(payload.hubCheckpoint, true);
    assert.equal(payload.assetDataset, "test-user/krea2-image-model-cache");
    assert.equal(payload.assetFiles.length, 3);
    assert.equal(payload.assetFiles[0].destination, `diffusion_models/${filename}`);
    assert.deepEqual(metadata.dataset_sources, ["test-user/krea2-image-model-cache"]);
    assert.match(source, /Kaggle model cache manifest does not match the pinned Krea-2 assets/);
    assert.match(source, /hashlib\.sha256\(\)/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("startWorker pushes a persistent notebook kernel and records a worker session", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-start-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeProvider(root, db);
  const cliCalls = [];
  provider.runCli = async (args) => {
    cliCalls.push(args);
    return { stdout: "", stderr: "" };
  };
  try {
    const session = await provider.startWorker({ model: "qwen-image-2.1-step90000-FP8-comfy.safetensors" });
    assert.equal(session.status, "starting");
    assert.equal(session.model, "qwen-image-2.1-step90000-FP8-comfy.safetensors");
    assert.equal(cliCalls.length, 1);
    assert.equal(cliCalls[0][0], "kernels");
    assert.equal(cliCalls[0][1], "push");
    const notebookJson = fs.readFileSync(path.join(root, "kaggle", "jobs", session.id, "code.ipynb"), "utf8");
    const notebook = JSON.parse(notebookJson);
    assert.ok(Buffer.byteLength(notebookJson, "utf8") < 240 * 1024);
    assert.equal(notebook.cells[0].id, "local-ai-qwen-worker");
    const source = notebook.cells[0].source.join("");
    assert.match(source, /polling for jobs/);
    assert.match(source, /idle_timeout/);
    assert.match(source, /--require-hashes/);
    assert.match(source, /--no-deps/);
    assert.match(source, /len\(reference_bytes\) > 2 \* 1024 \* 1024/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("startWorker rejects a loopback model-store URL before pushing an unreachable kernel", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-loopback-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeProvider(root, db, { KAGGLE_MODEL_STORE_URL: "http://127.0.0.1:3988" });
  let pushCount = 0;
  provider.runCli = async () => {
    pushCount++;
    return { stdout: "", stderr: "" };
  };
  try {
    await assert.rejects(
      provider.startWorker({ model: "qwen-image-2.1-step90000-FP8-comfy.safetensors" }),
      (error) => error.status === 400 && /HTTPS and be publicly reachable/.test(error.message),
    );
    assert.equal(pushCount, 0);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("Krea-2 worker jobs preserve reference strength, LoRA, and advanced image settings", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-krea-"));
  const db = new DatabaseSync(":memory:");
  const kreaDir = path.join(root, "models", "stable-yogi-krea2");
  fs.mkdirSync(kreaDir, { recursive: true });
  fs.writeFileSync(path.join(kreaDir, KREA2_CHECKPOINT), "weights");
  const provider = makeProvider(root, db, {
    KAGGLE_MODEL_STORE_MAP: JSON.stringify({ "stable-yogi-krea2": kreaDir }),
    KAGGLE_IMAGE_MODEL_KEY_MAP: JSON.stringify({ [KREA2_CHECKPOINT]: "stable-yogi-krea2" }),
  });
  provider.runCli = async () => ({ stdout: "", stderr: "" });
  try {
    const session = await provider.startWorker({ model: KREA2_CHECKPOINT });
    const referencePng = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j0aYAAAAASUVORK5CYII=",
      "base64",
    );
    const job = provider.enqueueWorkerJob(session.id, {
      prompt: "portrait with studio lighting",
      negativePrompt: "blurred",
      width: 768,
      height: 1152,
      steps: 18,
      cfg: 1.8,
      seed: 42,
      lora: "krea2_darkbrush.safetensors",
      loraStrength: 0.8,
      referenceStrength: 0.35,
      referenceImage: referencePng.toString("base64"),
      inpaint: true,
    });
    const row = db.prepare("SELECT request_json FROM compute_jobs WHERE id = ?").get(job.id);
    const request = JSON.parse(row.request_json);
    assert.equal(request.negativePrompt, "blurred");
    assert.equal(request.width, 768);
    assert.equal(request.height, 1152);
    assert.equal(request.graph["1l"].inputs.strength_model, 0.8);
    assert.equal(request.graph["5"].inputs.denoise, 0.35);
    assert.equal(request.inpaint, true);
    assert.equal(request.graph["4c"].class_type, "VAEEncodeForInpaint");
    assert.equal(request.graph["4img"].inputs.image, "worker_reference_image.png");
    assert.equal(request.graph["5"].inputs.seed, 42);
    assert.equal(request.referenceImage, referencePng.toString("base64"));

    const jpegJob = provider.enqueueWorkerJob(session.id, {
      prompt: "portrait",
      referenceImage: Buffer.from([0xff, 0xd8, 0xff]).toString("base64"),
    });
    const jpegRow = db.prepare("SELECT request_json FROM compute_jobs WHERE id = ?").get(jpegJob.id);
    const jpegRequest = JSON.parse(jpegRow.request_json);
    assert.equal(jpegRequest.graph["4img"].inputs.image, "worker_reference_image.jpg");

    const largeReference = Buffer.alloc(300_000, 1);
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(largeReference);
    const largeJob = provider.enqueueWorkerJob(session.id, {
      prompt: "compose a scene from references",
      referenceImages: [{ data: largeReference.toString("base64"), label: "identity" }],
    });
    const largeRow = db.prepare("SELECT request_json FROM compute_jobs WHERE id = ?").get(largeJob.id);
    assert.equal(JSON.parse(largeRow.request_json).referenceImages[0].data, largeReference.toString("base64"));

    const oversizedReference = Buffer.alloc(2 * 1024 * 1024 + 1, 1);
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(oversizedReference);
    assert.throws(
      () => provider.enqueueWorkerJob(session.id, {
        prompt: "oversized reference",
        referenceImages: [{ data: oversizedReference.toString("base64") }],
      }),
      (error) => error.status === 413 && /exceeds the 2 MiB worker limit/.test(error.message),
    );
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("worker bridge requires a valid session token and scopes jobs to their owning session", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-bridge-"));
  const db = new DatabaseSync(":memory:");
  const outputDir = path.join(root, "downloads");
  fs.mkdirSync(outputDir, { recursive: true });
  const tokenA = "a".repeat(64);
  const tokenB = "b".repeat(64);
  const sessionA = queue.createSession(db, {
    id: "session-a", token: tokenA, model: "qwen-image-2.1-step90000-FP8-comfy.safetensors",
    remoteId: "user/kernel-a", idleTimeoutSeconds: 900, maxRuntimeSeconds: 21600,
  });
  const sessionB = queue.createSession(db, {
    id: "session-b", token: tokenB, model: "qwen-image-2.1-step90000-FP8-comfy.safetensors",
    remoteId: "user/kernel-b", idleTimeoutSeconds: 900, maxRuntimeSeconds: 21600,
  });
  queue.enqueueJob(db, { id: "job-a", sessionId: sessionA.id, kind: "image", model: sessionA.model, requestJson: { prompt: "a cat" } });
  const app = express();
  app.use(express.json({ limit: "64mb" }));
  mountWorkerBridge(app, { db, outputDir, logger: { error() {} } });
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  try {
    const unauthorized = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/claim`, { method: "POST" });
    assert.equal(unauthorized.status, 401);

    const wrongToken = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/claim`, {
      method: "POST", headers: { Authorization: `Bearer ${tokenB}` },
    });
    assert.equal(wrongToken.status, 401);

    const claimed = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/claim`, {
      method: "POST", headers: { Authorization: `Bearer ${tokenA}` },
    });
    assert.equal(claimed.status, 200);
    const claimBody = await claimed.json();
    assert.equal(claimBody.job.id, "job-a");
    assert.equal(claimBody.job.request.prompt, "a cat");

    const progress = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/jobs/job-a/progress`, {
      method: "POST",
      headers: { Authorization: `Bearer ${tokenA}`, "Content-Type": "application/json" },
      body: JSON.stringify({ message: "GPU 0 70% util, 12000/15360 MiB VRAM" }),
    });
    assert.equal(progress.status, 200);
    assert.equal((await progress.json()).statusMessage, "GPU 0 70% util, 12000/15360 MiB VRAM");
    const runningHeartbeat = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/heartbeat`, {
      method: "POST", headers: { Authorization: `Bearer ${tokenA}`, "Content-Type": "application/json" },
      body: JSON.stringify({ status: "running" }),
    });
    assert.equal(runningHeartbeat.status, 200);
    assert.equal(queue.getSession(db, sessionA.id).status, "busy");
    const jobStatus = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/jobs/job-a`, {
      headers: { Authorization: `Bearer ${tokenA}` },
    });
    assert.equal((await jobStatus.json()).status, "running");

    // session B must not be able to complete session A's job.
    const crossSessionComplete = await fetch(`${baseUrl}/v1/workers/${sessionB.id}/jobs/job-a/complete`, {
      method: "POST",
      headers: { Authorization: `Bearer ${tokenB}`, "Content-Type": "application/json" },
      body: JSON.stringify({ result: {}, images: [{ data: Buffer.from("png").toString("base64"), seed: 1 }] }),
    });
    assert.equal(crossSessionComplete.status, 403);

    const complete = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/jobs/job-a/complete`, {
      method: "POST",
      headers: { Authorization: `Bearer ${tokenA}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        result: { model: sessionA.model, width: 1024, height: 1024 },
        images: [{ data: Buffer.from("fake-png-bytes").toString("base64"), seed: 42 }],
      }),
    });
    assert.equal(complete.status, 200);
    assert.ok(fs.existsSync(path.join(outputDir, "job-a", "candidate-1.png")));

    const duplicateComplete = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/jobs/job-a/complete`, {
      method: "POST",
      headers: { Authorization: `Bearer ${tokenA}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        result: { model: sessionA.model, width: 1024, height: 1024 },
        images: [{ data: Buffer.from("fake-png-bytes").toString("base64"), seed: 42 }],
      }),
    });
    assert.equal(duplicateComplete.status, 200);
    assert.deepEqual(await duplicateComplete.json(), { ok: true, duplicate: true });

    const heartbeat = await fetch(`${baseUrl}/v1/workers/${sessionA.id}/heartbeat`, {
      method: "POST", headers: { Authorization: `Bearer ${tokenA}`, "Content-Type": "application/json" },
      body: JSON.stringify({ status: "idle" }),
    });
    assert.equal(heartbeat.status, 200);
    assert.deepEqual(await heartbeat.json(), { stop: false });
    assert.equal(queue.getSession(db, sessionA.id).status, "idle");
  } finally {
    await new Promise((resolve) => server.close(resolve));
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("worker bridge reports cancellation and rejects stale progress/results", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-cancel-"));
  const db = new DatabaseSync(":memory:");
  const outputDir = path.join(root, "downloads");
  fs.mkdirSync(outputDir, { recursive: true });
  const token = "z".repeat(64);
  const session = queue.createSession(db, {
    id: "cancel-session",
    token,
    kind: "video",
    model: "wan22-svi-q4km",
    remoteId: "user/cancel-worker",
    idleTimeoutSeconds: 900,
    maxRuntimeSeconds: 21600,
  });
  queue.enqueueJob(db, {
    id: "cancel-job",
    sessionId: session.id,
    kind: "video",
    model: session.model,
    requestJson: { prompt: "synthetic scene" },
  });
  const app = express();
  app.use(express.json({ limit: "64mb" }));
  mountWorkerBridge(app, { db, outputDir, logger: { error() {} } });
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  const headers = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
  try {
    const claim = await fetch(`${baseUrl}/v1/workers/${session.id}/claim`, {
      method: "POST",
      headers,
    });
    assert.equal((await claim.json()).job.id, "cancel-job");
    db.prepare("UPDATE compute_jobs SET status = 'cancelled' WHERE id = ?").run("cancel-job");

    const status = await fetch(`${baseUrl}/v1/workers/${session.id}/jobs/cancel-job`, { headers });
    assert.deepEqual(await status.json(), { status: "cancelled", statusMessage: null });

    const progress = await fetch(`${baseUrl}/v1/workers/${session.id}/jobs/cancel-job/progress`, {
      method: "POST",
      headers,
      body: JSON.stringify({ message: "late progress" }),
    });
    assert.equal(progress.status, 409);

    const complete = await fetch(`${baseUrl}/v1/workers/${session.id}/jobs/cancel-job/complete`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        result: { model: session.model },
        videos: [{ data: Buffer.from("stale output").toString("base64"), mime: "video/webm" }],
      }),
    });
    assert.equal(complete.status, 409);
    assert.equal(fs.existsSync(path.join(outputDir, "cancel-job")), false);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("claim reports stop:true once a session's stop has been requested", async () => {
  const db = new DatabaseSync(":memory:");
  const stopToken = "c".repeat(64);
  const session = queue.createSession(db, {
    id: "stop-session", token: stopToken, model: "m", remoteId: "user/k", idleTimeoutSeconds: 900, maxRuntimeSeconds: 21600,
  });
  queue.requestStop(db, session.id);
  const app = express();
  app.use(express.json());
  mountWorkerBridge(app, { db, outputDir: fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-stop-")), logger: { error() {} } });
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const response = await fetch(`http://127.0.0.1:${server.address().port}/v1/workers/${session.id}/claim`, {
      method: "POST", headers: { Authorization: `Bearer ${stopToken}` },
    });
    assert.deepEqual(await response.json(), { job: null, stop: true });
  } finally {
    await new Promise((resolve) => server.close(resolve));
    db.close();
  }
});

test("worker notebook embeds the claim/complete/fail polling loop against the bridge", () => {
  const notebook = buildQwenWorkerNotebook({
    sessionId: "session-xyz",
    token: "f".repeat(64),
    model: "Jib_Mix_Moss_V2_H_int8_00001_.safetensors",
    modelSource: {
      baseUrl: "https://models.example.test", key: "jib-mix-moss", expiresAt: 1234567890,
      signature: "e".repeat(64), maxBytes: 1024 ** 3, totalBytes: 512 * 1024 ** 2,
    },
    bridgeBaseUrl: "https://models.example.test",
    idleTimeoutSeconds: 600,
    maxRuntimeSeconds: 7200,
  });
  const source = notebook.cells[0].source.join("");
  assert.match(source, /2c75dc1b379a59e0880619d029607af5169a62ef/);
  assert.match(source, /\/v1\/workers\/\{session_id\}\/claim/);
  assert.match(source, /\/v1\/workers\/\{session_id\}\/jobs\/\{job_id\}\/complete/);
  assert.match(source, /\/v1\/workers\/\{session_id\}\/jobs\/\{job_id\}\/fail/);
  assert.match(source, /\/v1\/workers\/\{session_id\}\/heartbeat/);
  assert.match(source, /\/v1\/workers\/\{session_id\}\/jobs\/\{job_id\}\/progress/);
  assert.match(source, /now - last_job_progress >= 15/);
  assert.match(source, /\(history_record or \{\}\)\.get\("outputs"\)/);
  assert.match(source, /find_generated_images\(history, output_directory, comfyui_dir, submitted_at\)/);
  assert.match(source, /ComfyUI reported completion but no generated PNG was found/);
  assert.match(source, /report_startup\(f"fetching model manifest/);
  assert.match(source, /shutil\.disk_usage\("\/kaggle\/working"\)/);
  assert.match(source, /required_disk_bytes = manifest\["totalBytes"\] \+ companion_total_bytes \+ 1024 \*\* 3/);
  assert.match(source, /wait\(pending, timeout=10, return_when=FIRST_COMPLETED\)/);
  assert.match(source, /model download \{completed_chunks\}\/\{len\(chunk_tasks\)\} chunks/);
  assert.match(source, /companion download \{companion\['filename'\]\}/);
  assert.match(source, /clip_type = "moss_image" if preset\["family"\] == "moss" else "qwen_image"/);
  assert.match(source, /shutil\.rmtree\(comfyui_dir, ignore_errors=True\)/);
});

test("worker resolves the prompt's ComfyUI history image even when file timestamps lag", (t) => {
  let python;
  for (const candidate of ["python", "py"]) {
    try {
      execFileSync(candidate, ["-c", "import sys; sys.exit(0)"], { stdio: "ignore" });
      python = candidate;
      break;
    } catch {}
  }
  if (!python) return t.skip("Python is unavailable, so generated image selection was not exercised");

  const notebook = buildQwenWorkerNotebook({
    sessionId: "image-output-check",
    token: "f".repeat(64),
    model: "Jib_Mix_Moss_V2_H_int8_00001_.safetensors",
    modelSource: {
      baseUrl: "https://models.example.test", key: "jib-mix-moss", expiresAt: 1234567890,
      signature: "e".repeat(64), maxBytes: 1024 ** 3, totalBytes: 512 * 1024 ** 2,
    },
    bridgeBaseUrl: "https://models.example.test",
    idleTimeoutSeconds: 600,
    maxRuntimeSeconds: 7200,
  });
  const source = notebook.cells[0].source.join("");
  const helperStart = source.indexOf("def find_generated_images(");
  const helperEnd = source.indexOf("\ntry:\n    for _ in range(180):", helperStart);
  assert.ok(helperStart >= 0 && helperEnd > helperStart, "generated worker must contain the image resolver");
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-output-"));
  const outputDirectory = path.join(root, "output");
  fs.mkdirSync(outputDirectory);
  const imagePath = path.join(outputDirectory, "candidate.png");
  fs.writeFileSync(imagePath, "image");
  fs.utimesSync(imagePath, new Date(0), new Date(0));
  const pythonSource = [
    "import json",
    "import pathlib",
    source.slice(helperStart, helperEnd),
    "root = pathlib.Path(__import__('sys').argv[1])",
    "history = json.loads(__import__('sys').argv[2])",
    "images = find_generated_images(history, root / 'output', root, 9999999999)",
    "print(images[0] if images else '')",
  ].join("\n");
  try {
    const result = execFileSync(python, [
      "-c",
      pythonSource,
      root,
      JSON.stringify({
        outputs: {
          "7": { images: [{ filename: "candidate.png", subfolder: "", type: "output" }] },
        },
      }),
    ], { encoding: "utf8" }).trim();
    assert.equal(result, imagePath);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("Qwen-Image-2.1 GGUF worker routes detailed reference and LoRA requests", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-qwen21-worker-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeProvider(root, db);
  provider.runCli = async () => ({ stdout: "", stderr: "" });
  try {
    const session = await provider.startWorker({ model: "qwen-image-2.1-UC-Q4_K_M.gguf" });
    assert.equal(session.status, "starting");
    assert.equal(session.model, "qwen-image-2.1-UC-Q4_K_M.gguf");

    const notebook = JSON.parse(fs.readFileSync(
      path.join(root, "kaggle", "jobs", session.id, "code.ipynb"),
      "utf8",
    ));
    const source = notebook.cells[0].source.join("");
    const metadata = JSON.parse(fs.readFileSync(
      path.join(root, "kaggle", "jobs", session.id, "kernel-metadata.json"),
      "utf8",
    ));
    assert.deepEqual(metadata.dataset_sources, ["test-user/qwen-image-2-1-gguf-assets"]);
    const assetFilesMatch = source.match(/asset_files = (.+)\ninput_root/);
    assert.ok(assetFilesMatch);
    const assetFiles = JSON.parse(assetFilesMatch[1]);
    assert.deepEqual(assetFiles.map((item) => item.path), [
      "qwen-image-2.1-UC-Q4_K_M.gguf",
      "qwen3vl_8b_int8_convrot.safetensors",
      "qwen_image_2.1_vae_bf16.safetensors",
      "NSFW Qwen by TheseAlpacas V2.safetensors",
    ]);
    assert.deepEqual(assetFiles.map((item) => item.target), [
      "unet/qwen-image-2.1-UC-Q4_K_M.gguf",
      "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
      "vae/qwen_image_2.1_vae_bf16.safetensors",
      "loras/NSFW Qwen by TheseAlpacas V2.safetensors",
    ]);
    assert.match(source, /UnetLoaderGGUF/);
    assert.match(source, /TextEncodeQwenImage21/);
    assert.match(source, /QwenImage21Cache/);
    assert.match(source, /input_root = pathlib\.Path\("\/kaggle\/input"\)/);
    assert.match(source, /Qwen model asset failed SHA-256 verification/);
    assert.doesNotMatch(source, /QwenImage21Pipeline|dequantize_gguf_tensor/);
    assert.match(source, /qwen-image-2\.1-UC-Q4_K_M\.gguf/);
    assert.match(source, /comfyui_input = comfyui_dir \/ "input"/);
    assert.match(source, /comfyui_input\.mkdir\(parents=True, exist_ok=True\)/);
    assert.match(source, /IMG_ARCH_LIST = \{/);
    assert.match(source, /"qwen_image", "qwen_image21"\}/);
    assert.match(source, /encoder_inputs\["images\.image_\{\}"\.format\(index\)\]/);
    assert.match(source, /"negative_prompt": request\.get\("negativePrompt"\) or " "/);
    assert.match(source, /strength = float\(request\.get\("loraStrength", 1\.0\)\) \* 0\.9/);
    assert.match(source, /"device": "cpu"/);
    assert.match(source, /"dtype": "int8"/);

    let python;
    for (const candidate of ["python", "py"]) {
      try {
        execFileSync(candidate, ["-c", "import sys; sys.exit(0)"], { stdio: "ignore" });
        python = candidate;
        break;
      } catch {}
    }
    if (python) {
      execFileSync(python, ["-c", "import ast, sys; ast.parse(sys.stdin.read())"], {
        input: source,
        stdio: ["pipe", "ignore", "pipe"],
      });
    } else {
      t.skip("Python is unavailable, so generated Qwen worker source was not parsed");
    }

    const reference = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).toString("base64");
    assert.throws(() => provider.enqueueWorkerJob(session.id, {
      prompt: "single reference required",
      mode: "Edit Image (1 Ref)",
      width: 1344,
      height: 768,
      referenceImages: [],
    }), /requires exactly 1 reference image/);
    assert.throws(() => provider.enqueueWorkerJob(session.id, {
      prompt: "unsupported adapter",
      mode: "Text to Image",
      width: 1344,
      height: 768,
      loraAdapter: "Unknown",
    }), /supports only the base model and the pinned NSFW/);

    const job = provider.enqueueWorkerJob(session.id, {
      prompt: "Keep the detailed prompt unchanged.",
      mode: "Transform & Swap (2 Refs)",
      width: 1344,
      height: 768,
      steps: 28,
      seed: 42,
      negativePrompt: "base negative, no text",
      loraAdapter: "NSFW (general)",
      loraStrength: 1,
      referenceImages: [
        { data: reference, label: "identity plate" },
        { data: reference, label: "pose plate" },
      ],
    });
    assert.equal(job.status, "queued");
    const claimed = queue.claimNextJob(db, session.id);
    assert.equal(claimed.request.prompt, "Keep the detailed prompt unchanged.");
    assert.equal(claimed.request.mode, "Transform & Swap (2 Refs)");
    assert.equal(claimed.request.negativePrompt, "base negative, no text");
    assert.equal(claimed.request.loraAdapter, "NSFW (general)");
    assert.equal(claimed.request.loraStrength, 1);
    assert.deepEqual(
      claimed.request.referenceImages.map((item) => item.label),
      ["identity plate", "pose plate"],
    );
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("enqueueWorkerJob rejects unknown sessions and stopped sessions", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-enqueue-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeProvider(root, db);
  provider.runCli = async () => ({ stdout: "", stderr: "" });
  try {
    assert.throws(() => provider.enqueueWorkerJob("missing", { prompt: "a cat" }), /not found/);
    const session = await provider.startWorker({ model: "qwen-image-2.1-step90000-FP8-comfy.safetensors" });
    const job = provider.enqueueWorkerJob(session.id, { prompt: "a cat astronaut" });
    assert.equal(job.status, "queued");
    assert.equal(job.provider, "kaggle-worker");
    await provider.stopWorker(session.id);
    assert.equal(provider.getJob(job.id).status, "failed");
    assert.match(provider.getJob(job.id).error, /stopped before the task completed/);
    assert.throws(() => provider.enqueueWorkerJob(session.id, { prompt: "another" }), /stopped/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("stopWorker leaves a remote session active when Kaggle deletion fails", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-stop-failure-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeProvider(root, db);
  let remoteId;
  let failDelete = false;
  provider.runCli = async (args) => {
    if (failDelete && args[0] === "kernels" && args[1] === "delete") {
      throw new Error("Kaggle permission denied");
    }
    if (args[0] === "kernels" && args[1] === "list") {
      return { stdout: JSON.stringify([{ ref: remoteId }]), stderr: "" };
    }
    return { stdout: "", stderr: "" };
  };
  try {
    const session = await provider.startWorker({ model: "qwen-image-2.1-step90000-FP8-comfy.safetensors" });
    remoteId = session.remoteId;
    failDelete = true;
    await assert.rejects(
      provider.stopWorker(session.id),
      (error) => error.status === 502 && /may still be running/.test(error.message),
    );
    const current = provider.getWorkerStatus(session.id);
    assert.equal(current.status, "stopping");
    assert.equal(current.stopRequested, true);
    assert.match(current.error, /Kaggle permission denied/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("stopWorker clears a stale local session when its Kaggle kernel is absent", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-stop-stale-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeProvider(root, db);
  let failDelete = false;
  provider.runCli = async (args) => {
    if (failDelete && args[0] === "kernels" && args[1] === "delete") {
      throw new Error("Kaggle permission denied");
    }
    if (args[0] === "kernels" && args[1] === "list") {
      return { stdout: "[]", stderr: "" };
    }
    return { stdout: "", stderr: "" };
  };
  try {
    const session = await provider.startWorker({ model: "qwen-image-2.1-step90000-FP8-comfy.safetensors" });
    failDelete = true;
    const stopped = await provider.stopWorker(session.id);
    assert.equal(stopped.status, "stopped");
    assert.equal(stopped.stopRequested, true);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("syncWorkerStatus closes queued work after Kaggle acknowledges cancellation", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-sync-cancel-"));
  const db = new DatabaseSync(":memory:");
  const provider = makeProvider(root, db);
  provider.runCli = async (args) => ({
    stdout: args[0] === "kernels" && args[1] === "status"
      ? "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"
      : "",
    stderr: "",
  });
  try {
    const session = await provider.startWorker({ model: "qwen-image-2.1-step90000-FP8-comfy.safetensors" });
    const job = provider.enqueueWorkerJob(session.id, { prompt: "a cat astronaut" });
    const synced = await provider.syncWorkerStatus(session.id);
    assert.equal(synced.status, "failed");
    assert.equal(synced.remoteStatus, "failed");
    assert.equal(provider.getJob(job.id).status, "failed");
    assert.match(provider.getJob(job.id).error, /failed/);
  } finally {
    provider.stop();
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("claim → complete → claim returns null instead of 500 after job completion", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-worker-claim-cycle-"));
  const db = new DatabaseSync(path.join(root, "gateway.sqlite"));
  const outputDir = path.join(root, "downloads");
  fs.mkdirSync(outputDir, { recursive: true });
  const app = express();
  app.use(express.json({ limit: "64mb" }));
  mountWorkerBridge(app, { db, outputDir, logger: { error() {} } });
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const base = `http://127.0.0.1:${port}`;
  const token = "t".repeat(64);

  const session = queue.createSession(db, {
    id: "claim-cycle-session",
    token,
    model: "test-model",
    remoteId: "user/kernel-1",
    idleTimeoutSeconds: 900,
    maxRuntimeSeconds: 21600,
  });
  queue.enqueueJob(db, {
    id: "claim-cycle-job",
    sessionId: session.id,
    kind: "image",
    model: "test-model",
    requestJson: { prompt: "a cat" },
  });

  function post(path, body) {
    return new Promise((resolve, reject) => {
      const req = http.request(`${base}${path}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
          "Content-Length": body ? Buffer.byteLength(body) : 0,
        },
      }, (res) => {
        let data = "";
        res.on("data", chunk => data += chunk);
        res.on("end", () => resolve({ status: res.statusCode, body: data }));
      });
      req.on("error", reject);
      if (body) req.write(body);
      req.end();
    });
  }

  try {
    // Claim 1: should return the job
    let r = await post("/v1/workers/claim-cycle-session/claim");
    assert.equal(r.status, 200, `Claim 1 failed: ${r.body}`);
    let body = JSON.parse(r.body);
    assert.equal(body.job.id, "claim-cycle-job");

    // Complete the job
    r = await post("/v1/workers/claim-cycle-session/jobs/claim-cycle-job/complete", JSON.stringify({
      result: { ok: true },
      images: [{ data: "iVBORw0KGgo=", seed: 1 }],
    }));
    assert.equal(r.status, 200, `Complete failed: ${r.body}`);

    // Claim 2: should return null (no more jobs), NOT 500
    r = await post("/v1/workers/claim-cycle-session/claim");
    assert.equal(r.status, 200, `Claim 2 failed: ${r.body}`);
    body = JSON.parse(r.body);
    assert.equal(body.job, null);
    assert.equal(body.stop, false);

    // Claim 3: should also return null
    r = await post("/v1/workers/claim-cycle-session/claim");
    assert.equal(r.status, 200, `Claim 3 failed: ${r.body}`);
    body = JSON.parse(r.body);
    assert.equal(body.job, null);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("ensureSchema is idempotent when called repeatedly", () => {
  const db = new DatabaseSync(":memory:");
  try {
    // Calling ensureSchema multiple times should not throw
    for (let i = 0; i < 5; i++) {
      queue.ensureSchema(db);
    }
    // Verify the schema was created correctly
    const session = queue.createSession(db, {
      id: "schema-test-session",
      token: "s".repeat(64),
      model: "test-model",
      remoteId: "user/kernel-1",
      idleTimeoutSeconds: 900,
      maxRuntimeSeconds: 21600,
    });
    assert.equal(session.id, "schema-test-session");
    assert.equal(session.status, "starting");
  } finally {
    db.close();
  }
});
