const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {
  ModelStoreProvider,
  createModelStoreApp,
  scanModelDirectory,
  signatureFor,
  assertPublicHttpsOrigin,
} = require("../gateway/model-store");
const { buildImageNotebook } = require("../gateway/kaggle");

const secret = "local-model-store-test-secret-0123456789";

test("Kaggle model-store URL requires a public HTTPS origin", () => {
  assert.equal(assertPublicHttpsOrigin("https://models.example.test").origin, "https://models.example.test");
  for (const url of [
    "http://127.0.0.1:3988",
    "https://localhost",
    "https://192.168.1.20",
    "https://models.example.test/path",
  ]) {
    assert.throws(() => assertPublicHttpsOrigin(url), /publicly reachable|without a path/);
  }
});

test("model-store access is scoped to an SSD directory and emits a signed expiring grant", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-store-"));
  const modelDir = path.join(root, "sd15");
  fs.mkdirSync(path.join(modelDir, "unet"), { recursive: true });
  fs.writeFileSync(path.join(modelDir, "model_index.json"), "{}");
  fs.writeFileSync(path.join(modelDir, "unet", "weights.safetensors"), "model");
  const provider = new ModelStoreProvider({
    KAGGLE_MODEL_TRANSFER_CONSENT: "true",
    KAGGLE_MODEL_STORE_URL: "https://models.example.test",
    KAGGLE_MODEL_STORE_SECRET: secret,
    KAGGLE_MODEL_STORE_MAP: JSON.stringify({ sd15: modelDir }),
    KAGGLE_MODEL_CACHE_MAX_GIB: "1",
  });
  try {
    const grant = provider.accessFor("sd15");
    assert.equal(grant.key, "sd15");
    assert.equal(grant.baseUrl, "https://models.example.test");
    assert.equal(grant.totalBytes, 7);
    assert.equal(grant.signature.length, 64);
    assert.ok(grant.expiresAt > Math.floor(Date.now() / 1000));
    assert.throws(() => provider.accessFor("missing"), /not configured/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("model-store refuses directories larger than the configured Kaggle cache ceiling", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-limit-"));
  fs.writeFileSync(path.join(root, "weights.bin"), "oversize");
  try {
    assert.throws(() => scanModelDirectory(root, 2), /exceeds the Kaggle cache limit/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("read-only model store requires a valid grant and supports authenticated range reads", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-http-"));
  const modelDir = path.join(root, "sd15");
  fs.mkdirSync(modelDir);
  fs.writeFileSync(path.join(modelDir, "weights.bin"), "abcdef");
  const app = createModelStoreApp({
    env: {
      KAGGLE_MODEL_TRANSFER_CONSENT: "true",
      KAGGLE_MODEL_STORE_SECRET: secret,
      KAGGLE_MODEL_STORE_MAP: JSON.stringify({ sd15: modelDir }),
      KAGGLE_MODEL_CACHE_MAX_GIB: "1",
    },
    logger: { error() {} },
  });
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  const expiresAt = Math.floor(Date.now() / 1000) + 60;
  const headers = {
    Authorization: `Bearer ${signatureFor(secret, "sd15", expiresAt)}`,
    "X-Model-Expires": String(expiresAt),
  };
  try {
    const health = await fetch(`${baseUrl}/health`);
    assert.equal(health.status, 200);
    assert.deepEqual(await health.json(), { service: "local-ai-model-store", status: "ok" });
    const denied = await fetch(`${baseUrl}/v1/models/sd15/manifest`);
    assert.equal(denied.status, 401);
    const manifest = await fetch(`${baseUrl}/v1/models/sd15/manifest`, { headers });
    assert.equal(manifest.status, 200);
    assert.deepEqual(await manifest.json(), {
      files: [{ path: "weights.bin", size: 6 }],
      totalBytes: 6,
    });
    const range = await fetch(`${baseUrl}/v1/models/sd15/files/weights.bin`, {
      headers: { ...headers, Range: "bytes=1-3" },
    });
    assert.equal(range.status, 206);
    assert.equal(await range.text(), "bcd");
    const expired = await fetch(`${baseUrl}/v1/models/sd15/manifest`, {
      headers: {
        Authorization: `Bearer ${signatureFor(secret, "sd15", expiresAt - 120)}`,
        "X-Model-Expires": String(expiresAt - 120),
      },
    });
    assert.equal(expired.status, 401);
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("model store serves staged Model Hub checkpoint roots with a signed grant", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-hub-store-"));
  const key = "civitai-2741166-3425119";
  const filename = "Muse By Stable Yogi Krea2 V4.0 Extended CIV INT8-ConvRot C86.safetensors";
  const checkpoint = path.join(root, key, "checkpoints", filename);
  fs.mkdirSync(path.dirname(checkpoint), { recursive: true });
  fs.writeFileSync(checkpoint, "Muse model test bytes");
  const app = createModelStoreApp({
    env: {
      KAGGLE_MODEL_TRANSFER_CONSENT: "true",
      KAGGLE_MODEL_STORE_SECRET: secret,
      KAGGLE_MODEL_CACHE_MAX_GIB: "1",
      MODEL_HUB_ROOT: root,
    },
    logger: { error() {} },
  });
  const server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  const expiresAt = Math.floor(Date.now() / 1000) + 60;
  const headers = {
    Authorization: `Bearer ${signatureFor(secret, key, expiresAt)}`,
    "X-Model-Expires": String(expiresAt),
  };
  try {
    const manifest = await fetch(`${baseUrl}/v1/models/${key}/manifest`, { headers });
    assert.equal(manifest.status, 200);
    assert.deepEqual(await manifest.json(), {
      files: [{ path: `checkpoints/${filename}`, size: Buffer.byteLength("Muse model test bytes") }],
      totalBytes: Buffer.byteLength("Muse model test bytes"),
    });
  } finally {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("image notebook downloads the SSD model into temporary Kaggle storage and enforces the cache ceiling", () => {
  const notebook = buildImageNotebook({
    prompt: "test",
    negativePrompt: "",
    model: "stable-diffusion-v1-5/stable-diffusion-v1-5",
    steps: 20,
    seed: 1,
    modelSource: {
      baseUrl: "https://models.example.test",
      key: "sd15",
      expiresAt: 1234567890,
      signature: "a".repeat(64),
      maxBytes: 1024,
      totalBytes: 512,
    },
  });
  const source = notebook.cells[0].source.join("");
  assert.match(source, /Kaggle cache limit/);
  assert.match(source, /tempfile\.mkdtemp\(prefix="local-ai-model-", dir="\/tmp"\)/);
  assert.match(source, /local_files_only=bool\(model_source\)/);
  assert.match(source, /shutil\.rmtree\(model_directory/);
  assert.match(source, /laptop-ssd/);
});
