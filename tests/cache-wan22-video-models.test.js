"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {
  checkpointPlan,
  modelFilePlan,
  selectedProfileIds,
  buildAssetDatasetMap,
  datasetPlans,
  verifyFile,
  runKaggleCli,
  waitForPublishedDatasets,
  isDatasetComplete,
  VIDEO_CACHE_PROFILE_IDS,
  MAX_DATASET_BYTES,
} = require("../scripts/cache-wan22-video-models");

test("Wan cache plan contains the two SVI profiles and required Q8 Lightning experts", () => {
  const plan = checkpointPlan();
  assert.deepEqual(VIDEO_CACHE_PROFILE_IDS, [
    "wan22-svi-q4km",
    "wan22-svi-cf-q4km",
    "wan22-fm-lightning-q8",
  ]);
  assert.equal(plan.length, 6);
  assert.deepEqual(plan.map(({ profileId, checkpoint }) => [profileId, checkpoint.versionId]), [
    ["wan22-svi-q4km", 2500306],
    ["wan22-svi-q4km", 2500309],
    ["wan22-svi-cf-q4km", 2625965],
    ["wan22-svi-cf-q4km", 2625966],
    ["wan22-fm-lightning-q8", 2540892],
    ["wan22-fm-lightning-q8", 2540896],
  ]);
});

test("Wan cache datasets include the mandatory Q8 pair and shared pinned assets", () => {
  const env = {
    KAGGLE_USERNAME: "Test-User",
    KAGGLE_DATA_DIR: "D:/kaggle",
  };
  const plans = datasetPlans(env);
  assert.equal(plans.length, 4);
  assert.deepEqual(plans.map((plan) => plan.id), [
    "test-user/wan22-svi-fast-move-cache",
    "test-user/wan22-svi-consistent-face-cache",
    "test-user/wan22-fast-move-lightning-q8-cache",
    "test-user/wan22-svi-common-assets-cache",
  ]);
  assert.deepEqual(plans.map((plan) => plan.files.length), [2, 2, 2, 4]);
  assert.ok(plans[2].files.every((file) => !file.path.includes("/")));
  assert.match(plans[2].subtitle, /wan22-fm-lightning-q8/);
  assert.doesNotMatch(plans[2].description, /Q4KM/);
  assert.deepEqual(datasetPlans({
    ...env,
    KAGGLE_VIDEO_CACHE_PROFILES: "wan22-svi-q4km",
    KAGGLE_VIDEO_CACHE_SHARED_ONLY: "true",
  }).map((plan) => plan.id), ["test-user/wan22-svi-common-assets-cache"]);
  assert.ok(plans.every((plan) =>
    plan.files.reduce((sum, file) => sum + file.sizeBytes, 0) <= MAX_DATASET_BYTES));
  assert.equal(modelFilePlan().length, 4);
  assert.deepEqual(selectedProfileIds({ KAGGLE_VIDEO_CACHE_PROFILES: "wan22-svi-q4km" }), [
    "wan22-svi-q4km",
  ]);
  assert.deepEqual(buildAssetDatasetMap("Test-User", ["wan22-svi-q4km"]), {
    "wan22-svi-q4km": {
      checkpoints: "test-user/wan22-svi-fast-move-cache",
      modelFiles: "test-user/wan22-svi-common-assets-cache",
    },
  });
  assert.deepEqual(buildAssetDatasetMap("Test-User", ["wan22-fm-lightning-q8"]), {
    "wan22-fm-lightning-q8": {
      checkpoints: "test-user/wan22-fast-move-lightning-q8-cache",
      modelFiles: "test-user/wan22-svi-common-assets-cache",
    },
  });
  assert.throws(
    () => selectedProfileIds({ KAGGLE_VIDEO_CACHE_PROFILES: "unknown-profile" }),
    /KAGGLE_VIDEO_CACHE_PROFILES/,
  );
});

test("cache publisher verifies file size and SHA-256", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "wan-cache-verify-"));
  const file = path.join(directory, "checkpoint.gguf");
  const contents = Buffer.from("pinned checkpoint test");
  fs.writeFileSync(file, contents);
  const expected = {
    sizeBytes: contents.length,
    sha256: crypto.createHash("sha256").update(contents).digest("hex"),
  };
  try {
    assert.doesNotThrow(() => verifyFile(file, expected));
    assert.throws(() => verifyFile(file, { ...expected, sizeBytes: contents.length + 1 }), /expected/);
    assert.throws(() => verifyFile(file, { ...expected, sha256: "0".repeat(64) }), /SHA-256/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("cache publisher falls back to the installed Kaggle Python module", () => {
  const invocations = [];
  const fakeExec = (executable, args) => {
    invocations.push({ executable, args });
    if (executable === "kaggle") {
      const error = new Error("not found");
      error.code = "ENOENT";
      throw error;
    }
    return "ok";
  };
  const result = runKaggleCli(["datasets", "list"], {}, { KAGGLE_CLI: "kaggle" }, fakeExec);
  assert.equal(result, "ok");
  assert.deepEqual(invocations, [
    { executable: "kaggle", args: ["datasets", "list"] },
    { executable: "python", args: ["-m", "kaggle", "datasets", "list"] },
  ]);
});

test("cache publisher waits for Kaggle dataset files and verifies their sizes", () => {
  let calls = 0;
  const waits = [];
  const plan = {
    id: "test-user/model-cache",
    files: [{ path: "weights/model.safetensors", sizeBytes: 4 }],
  };
  const fakeExec = (_executable, args) => {
    assert.deepEqual(args, ["datasets", "files", plan.id, "--format", "json"]);
    calls += 1;
    return JSON.stringify(calls === 1 ? [] : [{ name: "model.safetensors", size: 4 }]);
  };

  waitForPublishedDatasets(
    [plan],
    { KAGGLE_CLI: "kaggle" },
    fakeExec,
    { maxAttempts: 2, intervalMs: 25, sleepImpl: (duration) => waits.push(duration) },
  );
  assert.equal(calls, 2);
  assert.deepEqual(waits, [25]);
});

test("cache publisher reuses an existing dataset only when every expected file size matches", () => {
  const plan = {
    id: "test-user/model-cache",
    files: [{ path: "model.safetensors", sizeBytes: 4 }],
  };
  const completeExec = (_executable, args) => {
    assert.deepEqual(args, ["datasets", "files", plan.id, "--format", "json"]);
    return JSON.stringify([{ name: "model.safetensors", size: 4 }]);
  };
  const incompleteExec = (_executable, args) => {
    assert.deepEqual(args, ["datasets", "files", plan.id, "--format", "json"]);
    return JSON.stringify([{ name: "model.safetensors", size: 3 }]);
  };

  assert.equal(isDatasetComplete(plan, { KAGGLE_CLI: "kaggle" }, completeExec), true);
  assert.equal(isDatasetComplete(plan, { KAGGLE_CLI: "kaggle" }, incompleteExec), false);
});
