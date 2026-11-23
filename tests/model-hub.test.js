const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const { ModelHub } = require("../gateway/model-hub");

function fakeFetchFor({ model, fileBytes }) {
  return async (url) => {
    if (url.includes("/api/v1/models/")) {
      return { ok: true, status: 200, json: async () => model };
    }
    // Download URL: return a minimal Response-like object with a streaming body.
    const chunks = [fileBytes];
    let index = 0;
    return {
      ok: true,
      status: 200,
      body: {
        getReader() {
          return {
            async read() {
              if (index >= chunks.length) return { done: true, value: undefined };
              const value = chunks[index];
              index += 1;
              return { done: false, value };
            },
          };
        },
      },
    };
  };
}

function waitForStatus(hub, id, status, attempts = 200) {
  return new Promise((resolve, reject) => {
    const check = (remaining) => {
      const entry = hub.get(id);
      if (entry && entry.status === status) return resolve(entry);
      if (entry && entry.status === "failed" && status !== "failed") return reject(new Error(entry.error));
      if (remaining <= 0) return reject(new Error(`Timed out waiting for status ${status}; last status ${entry?.status}`));
      setTimeout(() => check(remaining - 1), 10);
    };
    check(attempts);
  });
}

test("ModelHub imports a Civitai checkpoint, downloads it, and stages it under its own folder", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-hub-"));
  const db = new DatabaseSync(":memory:");
  const fileBytes = Buffer.from("fake-checkpoint-bytes");
  const model = {
    name: "Test Checkpoint",
    type: "Checkpoint",
    modelVersions: [{
      id: 42,
      baseModel: "SD 1.5",
      files: [{ id: 99, name: "test-checkpoint.safetensors", sizeKB: fileBytes.length / 1024, primary: true, downloadUrl: "https://civitai.com/api/download/models/42?fileId=99" }],
    }],
  };
  const hub = new ModelHub({ db, hubRoot: root, fetchImpl: fakeFetchFor({ model, fileBytes }), env: {} });
  try {
    const entry = await hub.importFromUrl("https://civitai.com/models/1/test");
    assert.equal(entry.kind, "checkpoint");
    assert.equal(entry.filename, "test-checkpoint.safetensors");
    assert.ok(["pending", "downloading"].includes(entry.status));
    const complete = await waitForStatus(hub, entry.id, "complete");
    assert.equal(complete.downloadedBytes, fileBytes.length);
    const staged = path.join(root, complete.folderKey, "checkpoints", "test-checkpoint.safetensors");
    assert.equal(fs.readFileSync(staged, "utf8"), fileBytes.toString("utf8"));
    assert.equal(hub.folderKeyForFilename("test-checkpoint.safetensors"), complete.folderKey);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("ModelHub stages a LoRA under a loras/ subfolder and tracks it as allowlisted", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-hub-lora-"));
  const db = new DatabaseSync(":memory:");
  const fileBytes = Buffer.from("fake-lora-bytes");
  const model = {
    name: "Test LoRA",
    type: "LORA",
    modelVersions: [{
      id: 7,
      baseModel: "SDXL 1.0",
      files: [{ id: 8, name: "test-lora.safetensors", sizeKB: fileBytes.length / 1024, primary: true, downloadUrl: "https://civitai.com/api/download/models/7?fileId=8" }],
    }],
  };
  const hub = new ModelHub({ db, hubRoot: root, fetchImpl: fakeFetchFor({ model, fileBytes }), env: {} });
  try {
    const entry = await hub.importFromUrl("2");
    const complete = await waitForStatus(hub, entry.id, "complete");
    assert.equal(complete.kind, "lora");
    const staged = path.join(root, complete.folderKey, "loras", "test-lora.safetensors");
    assert.ok(fs.existsSync(staged));
    assert.equal(hub.isAllowlistedLora("test-lora.safetensors"), true);
    assert.equal(hub.isAllowlistedLora("not-imported.safetensors"), false);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("ModelHub reports only exact compatible generic-image LoRAs", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-hub-compatibility-"));
  const db = new DatabaseSync(":memory:");
  const hub = new ModelHub({ db, hubRoot: root, env: {} });
  const insert = db.prepare(`
    INSERT INTO model_hub_imports
      (id, source_url, model_id, version_id, model_name, kind, base_model, filename, folder_key, expected_bytes, status)
    VALUES (?, '', ?, ?, ?, ?, ?, ?, ?, 1, 'complete')
  `);
  insert.run("checkpoint", 1, 1, "SDXL checkpoint", "checkpoint", "SDXL 1.0", "checkpoint.safetensors", "checkpoint");
  insert.run("compatible", 2, 1, "SDXL LoRA", "lora", "SDXL", "compatible.safetensors", "compatible");
  insert.run("incompatible", 3, 1, "SD 1.5 LoRA", "lora", "SD 1.5", "incompatible.safetensors", "incompatible");
  try {
    assert.equal(hub.isCompatibleLora("compatible.safetensors", "checkpoint.safetensors"), true);
    assert.equal(hub.isCompatibleLora("incompatible.safetensors", "checkpoint.safetensors"), false);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("ModelHub neutralizes path traversal by staging only the basename", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-hub-traversal-"));
  const db = new DatabaseSync(":memory:");
  const fileBytes = Buffer.from("fake-checkpoint-bytes");
  const model = {
    name: "Traversal",
    type: "Checkpoint",
    modelVersions: [{
      id: 1,
      baseModel: "SD 1.5",
      files: [{ id: 99, name: "../../../windows/evil.safetensors", sizeKB: fileBytes.length / 1024, primary: true, downloadUrl: "https://civitai.com/api/download/models/1?fileId=99" }],
    }],
  };
  const hub = new ModelHub({ db, hubRoot: root, fetchImpl: fakeFetchFor({ model, fileBytes }), env: {} });
  try {
    const entry = await hub.importFromUrl("https://civitai.com/models/1/test");
    // Traversal was stripped: only the safe basename is staged.
    assert.equal(entry.filename, "evil.safetensors");
    const complete = await waitForStatus(hub, entry.id, "complete");
    const staged = path.join(root, complete.folderKey, "checkpoints", "evil.safetensors");
    assert.equal(fs.readFileSync(staged, "utf8"), fileBytes.toString("utf8"));
    // No directory was created outside the hub root.
    assert.equal(fs.readdirSync(root).length, 1);
    assert.equal(fs.readdirSync(root)[0], complete.folderKey);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("ModelHub rejects a Civitai file whose basename has an unsafe extension", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-hub-badext-"));
  const db = new DatabaseSync(":memory:");
  const model = {
    name: "BadExt",
    type: "Checkpoint",
    modelVersions: [{
      id: 1,
      baseModel: "SD 1.5",
      files: [{ id: 99, name: "evil.exe", sizeKB: 1024, primary: true, downloadUrl: "https://civitai.com/api/download/models/1?fileId=99" }],
    }],
  };
  const hub = new ModelHub({ db, hubRoot: root, fetchImpl: fakeFetchFor({ model, fileBytes: Buffer.from("x") }), env: {} });
  try {
    await assert.rejects(hub.importFromUrl("https://civitai.com/models/1/test"), /Unsupported Civitai file extension/);
    assert.equal(fs.readdirSync(root).length, 0);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("ModelHub.remove deletes the staged directory and the database row", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-hub-remove-"));
  const db = new DatabaseSync(":memory:");
  const fileBytes = Buffer.from("x");
  const model = {
    name: "Removable",
    type: "Checkpoint",
    modelVersions: [{ id: 5, baseModel: "SD 1.5", files: [{ id: 6, name: "r.safetensors", sizeKB: fileBytes.length / 1024, primary: true, downloadUrl: "https://civitai.com/api/download/models/5?fileId=6" }] }],
  };
  const hub = new ModelHub({ db, hubRoot: root, fetchImpl: fakeFetchFor({ model, fileBytes }), env: {} });
  try {
    const entry = await hub.importFromUrl("3");
    await waitForStatus(hub, entry.id, "complete");
    assert.equal(hub.remove(entry.id), true);
    assert.equal(hub.get(entry.id), null);
    assert.equal(hub.remove(entry.id), false);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});