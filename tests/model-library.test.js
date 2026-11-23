const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const { ModelLibrary } = require("../gateway/model-library");

const QWEN_CHECKPOINT = "qwen-image-2.1-step90000-FP8-comfy.safetensors";
const MOSS_CHECKPOINT = "Jib_Mix_Moss_V2_H_int8_00001_.safetensors";
const KREA2_CHECKPOINT = "Ultra Realistic by Stable Yogi Krea2 V1.0 Civ INT8-ConvRot C89.safetensors";

function writeFile(filePath, bytes) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, Buffer.alloc(bytes));
}

test("ModelLibrary reports missing checkpoints as incomplete", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-library-"));
  const db = new DatabaseSync(":memory:");
  const library = new ModelLibrary({ db, modelsRoot: path.join(root, "models"), env: {} });
  const qwen = library.qwenFamilyEntries().find((entry) => entry.checkpoint === QWEN_CHECKPOINT);
  assert.equal(qwen.complete, false);
  assert.equal(qwen.files.diffusion_model.present, false);
});

test("ModelLibrary reports a checkpoint complete only when every file matches its known byte size", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-library-"));
  const base = path.join(root, "models", "qwen-image-21");
  writeFile(path.join(base, "diffusion_models", QWEN_CHECKPOINT), 10);
  writeFile(path.join(base, "text_encoders", "qwen3vl_8b_int8_convrot.safetensors"), 10);
  writeFile(path.join(base, "vae", "qwen_image_2.1_vae_bf16.safetensors"), 10);
  const db = new DatabaseSync(":memory:");
  const library = new ModelLibrary({ db, modelsRoot: path.join(root, "models"), env: {} });
  const qwen = library.qwenFamilyEntries().find((entry) => entry.checkpoint === QWEN_CHECKPOINT);
  assert.equal(qwen.files.diffusion_model.present, true);
  assert.equal(qwen.files.diffusion_model.verified, false);
  assert.equal(qwen.complete, false);
});

test("ModelLibrary persists the active Qwen checkpoint across instances sharing a database", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-library-"));
  const db = new DatabaseSync(":memory:");
  const library = new ModelLibrary({ db, modelsRoot: path.join(root, "models"), env: {} });
  assert.equal(library.activeQwenCheckpoint(), QWEN_CHECKPOINT);
  library.setActiveQwenCheckpoint(MOSS_CHECKPOINT);
  assert.equal(library.activeQwenCheckpoint(), MOSS_CHECKPOINT);
  const second = new ModelLibrary({ db, modelsRoot: path.join(root, "models"), env: {} });
  assert.equal(second.activeQwenCheckpoint(), MOSS_CHECKPOINT);
  const entries = second.qwenFamilyEntries();
  assert.equal(entries.find((entry) => entry.checkpoint === MOSS_CHECKPOINT).active, true);
  assert.equal(entries.find((entry) => entry.checkpoint === QWEN_CHECKPOINT).active, false);
});

test("ModelLibrary tracks the staged Krea-2 add-on under its own folder", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-library-"));
  const base = path.join(root, "models", "stable-yogi-krea2");
  writeFile(path.join(base, "diffusion_models", KREA2_CHECKPOINT), 10);
  writeFile(path.join(base, "text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"), 10);
  writeFile(path.join(base, "vae", "qwen_image_vae.safetensors"), 10);
  const db = new DatabaseSync(":memory:");
  const library = new ModelLibrary({ db, modelsRoot: path.join(root, "models"), env: {} });
  const krea = library.qwenFamilyEntries().find((entry) => entry.checkpoint === KREA2_CHECKPOINT);
  assert.equal(krea.folder, "stable-yogi-krea2");
  assert.equal(krea.files.diffusion_model.present, true);
  assert.equal(krea.files.diffusion_model.verified, false);
  assert.equal(krea.complete, false);
});

test("ModelLibrary excludes unsupported video architectures from generic image choices", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-library-"));
  const hubRoot = path.join(root, "model-hub");
  const imports = [
    {
      status: "complete",
      kind: "checkpoint",
      filename: "wan-high.gguf",
      folderKey: "civitai-1-2",
      expectedBytes: 1,
      modelName: "Wan GGUF",
      baseModel: "Wan Video",
      modelId: 1,
      versionId: 2,
    },
    {
      status: "complete",
      kind: "checkpoint",
      filename: "portrait.safetensors",
      folderKey: "civitai-3-4",
      expectedBytes: 1,
      modelName: "Portrait",
      baseModel: "SDXL",
      modelId: 3,
      versionId: 4,
    },
    {
      status: "complete",
      kind: "checkpoint",
      filename: "wan-video.safetensors",
      folderKey: "civitai-5-6",
      expectedBytes: 1,
      modelName: "Wan Video",
      baseModel: "Wan Video 2.2 I2V-A14B",
      modelId: 5,
      versionId: 6,
    },
  ];
  writeFile(path.join(hubRoot, "civitai-3-4", "checkpoints", "portrait.safetensors"), 1);
  const db = new DatabaseSync(":memory:");
  const library = new ModelLibrary({
    db,
    modelsRoot: path.join(root, "models"),
    env: {},
    modelHub: {
      hubRoot,
      list: () => imports,
      infoForFilename: () => ({ family: null }),
    },
  });
  assert.deepEqual(library.hubModelEntries().map((entry) => entry.checkpoint), ["portrait.safetensors"]);
  assert.equal(
    library.imageModelEntries().find((entry) => entry.id === "portrait.safetensors").baseModel,
    "SDXL",
  );
});

test("ModelLibrary rejects activating an unknown checkpoint", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-library-"));
  const db = new DatabaseSync(":memory:");
  const library = new ModelLibrary({ db, modelsRoot: path.join(root, "models"), env: {} });
  assert.throws(
    () => library.setActiveQwenCheckpoint("not-a-real-checkpoint.safetensors"),
    /Unknown checkpoint/,
  );
});

test("ModelLibrary summary reports disk usage and other staged model families", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-model-library-"));
  const base = path.join(root, "models", "qwen-image-21");
  writeFile(path.join(base, "diffusion_models", QWEN_CHECKPOINT), 2048);
  const db = new DatabaseSync(":memory:");
  const library = new ModelLibrary({ db, modelsRoot: path.join(root, "models"), env: {} });
  const summary = library.summary();
  assert.equal(summary.diskUsageBytes, 2048);
  assert.ok(Array.isArray(summary.qwenModels));
  assert.ok(Array.isArray(summary.otherModels));
  assert.deepEqual(summary.otherModels, []);
});
