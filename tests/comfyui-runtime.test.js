"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");
const test = require("node:test");
const {
  COMFYUI_COMMIT,
  COMFYUI_RELEASE,
  COMFYUI_REPOSITORY,
  buildComfyUiCheckoutPython,
  buildComfyUiDependencyInstallPython,
} = require("../gateway/comfyui-runtime");

test("ComfyUI notebook checkout verifies the immutable release commit", () => {
  const source = buildComfyUiCheckoutPython();
  assert.equal(COMFYUI_RELEASE, "v0.44.0");
  assert.equal(COMFYUI_COMMIT, "2c75dc1b379a59e0880619d029607af5169a62ef");
  assert.equal(COMFYUI_REPOSITORY, "https://github.com/Comfy-Org/ComfyUI.git");
  assert.match(source, /"--branch", "v0\.44\.0"/);
  assert.match(source, /rev-parse/);
  assert.match(source, /comfyui_commit != "2c75dc1b379a59e0880619d029607af5169a62ef"/);
});

test("ComfyUI installs use the Python 3.13 hash lock without replacing Kaggle CUDA packages", () => {
  const source = buildComfyUiDependencyInstallPython();
  assert.match(source, /--require-hashes/);
  assert.match(source, /--no-deps/);
  assert.match(source, /comfyui-requirements-py313\.txt/);
  assert.match(source, /^import zlib/m);
  assert.match(source, /zlib\.decompress\(base64\.b64decode/);
  assert.match(source, /STARTUP: installing/);
  assert.match(source, /STARTUP: pinned ComfyUI dependencies installed/);
  assert.doesNotMatch(source, /"--quiet"/);
  assert.match(source, /timeout=600/);
  assert.match(source, /dependency installation exceeded 600 seconds/);

  const lock = fs.readFileSync(path.join(__dirname, "..", "gateway", "comfyui-requirements-py313.txt"), "utf8");
  const embeddedLock = source.match(/write_bytes\(zlib\.decompress\(base64\.b64decode\("([A-Za-z0-9+/=]+)"\)\)\)/);
  assert.ok(embeddedLock, "the generated notebook must embed the full lock");
  const compressedLock = Buffer.from(embeddedLock[1], "base64");
  assert.ok(compressedLock.length < Buffer.byteLength(lock));
  assert.equal(zlib.inflateSync(compressedLock).toString("utf8"), lock);
  const packages = lock.split(/\r?\n(?=[A-Za-z0-9_.-]+==)/)
    .filter((block) => /^[A-Za-z0-9_.-]+==/m.test(block));
  assert.ok(packages.length > 50, "the lock should include the transitive ComfyUI dependency set");
  for (const entry of packages) assert.match(entry, /--hash=sha256:[a-f0-9]{64}/);
  assert.match(lock, /^numpy==/m);
  assert.doesNotMatch(lock, /^(?:torch|torchvision|cuda-bindings|cuda-toolkit|nvidia-[A-Za-z0-9-]+|triton)==/m);
});
