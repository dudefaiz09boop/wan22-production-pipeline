"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {
  datasetPlan,
  verifyFile,
  writeDatasetMetadata,
  missingExpectedFiles,
  MAX_DATASET_BYTES,
} = require("../scripts/cache-krea2-image-model");

test("Krea-2 cache plan pins the checkpoint and both ComfyUI companion files", () => {
  const plan = datasetPlan({
    KAGGLE_USERNAME: "Test-User",
    KAGGLE_DATA_DIR: "D:/kaggle",
    MODEL_HUB_ROOT: "D:/models/model-hub",
  });
  assert.equal(plan.id, "test-user/krea2-image-model-cache");
  assert.equal(plan.files.length, 3);
  assert.deepEqual(plan.files.map((file) => file.path), [
    "Muse By Stable Yogi Krea2 V4.0 Extended CIV INT8-ConvRot C86.safetensors",
    "qwen3vl_4b_fp8_scaled.safetensors",
    "qwen_image_vae.safetensors",
  ]);
  assert.equal(plan.totalBytes, 18_988_226_462);
  assert.ok(plan.totalBytes < MAX_DATASET_BYTES);
});

test("Krea-2 cache publisher verifies exact hashes and writes its manifest", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "krea2-cache-"));
  const content = Buffer.from("pinned Krea cache asset");
  const digest = crypto.createHash("sha256").update(content).digest("hex");
  const file = path.join(directory, "asset.safetensors");
  fs.writeFileSync(file, content);
  const plan = {
    id: "test-user/krea2-image-model-cache",
    title: "Private Krea-2 Image Model Cache",
    subtitle: "Pinned Krea-2 assets",
    description: "Private cache",
    folder: directory,
    files: [{ path: "asset.safetensors", sizeBytes: content.length, sha256: digest, description: "test asset" }],
  };
  try {
    assert.doesNotThrow(() => verifyFile(file, plan.files[0]));
    assert.throws(() => verifyFile(file, { ...plan.files[0], sha256: "0".repeat(64) }), /SHA-256/);
    writeDatasetMetadata(plan);
    const manifest = JSON.parse(fs.readFileSync(path.join(directory, "asset-manifest.json"), "utf8"));
    assert.deepEqual(manifest.files, [{ path: "asset.safetensors", sizeBytes: content.length, sha256: digest }]);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("Krea-2 cache publisher detects missing and size-mismatched Kaggle assets", () => {
  const plan = {
    id: "test-user/krea2-image-model-cache",
    files: [
      { path: "checkpoints/model.safetensors", sizeBytes: 10 },
      { path: "vae/vae.safetensors", sizeBytes: 5 },
    ],
  };
  assert.deepEqual(missingExpectedFiles(plan, JSON.stringify([
    { name: "model.safetensors", size: 10 },
    { name: "vae.safetensors", size: 4 },
  ])), [plan.files[1]]);
});
