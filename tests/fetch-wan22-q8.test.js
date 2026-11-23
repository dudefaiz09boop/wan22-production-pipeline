"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { CHUNK_BYTES, chunkRanges, downloadFile, downloadRange } = require("../scripts/fetch-wan22-q8");

test("Q8 download ranges cover the exact file size without overlap", () => {
  assert.deepEqual(chunkRanges(CHUNK_BYTES + 7), [
    { start: 0, end: CHUNK_BYTES - 1, length: CHUNK_BYTES },
    { start: CHUNK_BYTES, end: CHUNK_BYTES + 6, length: 7 },
  ]);
});

test("Q8 downloader verifies range responses and resumes completed chunks", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "wan22-q8-range-test-"));
  const outputPath = path.join(directory, "checkpoint.part");
  const partsDir = path.join(directory, "parts");
  fs.mkdirSync(partsDir);
  const fileHandle = await fsp.open(outputPath, "w+");
  await fileHandle.truncate(8);
  const file = { versionId: 2540892, fileId: 2429064, sizeBytes: 8 };
  const range = { start: 2, end: 5, length: 4 };
  let fetchCalls = 0;
  const fetchImpl = async (url, options) => {
    fetchCalls += 1;
    assert.equal(url, "https://civitai.com/api/download/models/2540892?fileId=2429064");
    assert.equal(options.headers.Range, "bytes=2-5");
    return new Response(Buffer.from("data"), {
      status: 206,
      headers: { "content-range": "bytes 2-5/8" },
    });
  };

  try {
    assert.equal(await downloadRange({
      fileHandle, file, range, rangeIndex: 0, partsDir, headers: {}, fetchImpl,
    }), true);
    assert.equal(await downloadRange({
      fileHandle, file, range, rangeIndex: 0, partsDir, headers: {}, fetchImpl,
    }), false);
    assert.equal(fetchCalls, 1);
    assert.equal(fs.readFileSync(outputPath).subarray(2, 6).toString(), "data");
  } finally {
    await fileHandle.close();
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("Q8 downloader clears incomplete state after a final hash mismatch", async () => {
  const hubRoot = fs.mkdtempSync(path.join(os.tmpdir(), "wan22-q8-hash-test-"));
  const file = {
    versionId: 2540892,
    fileId: 2429064,
    fileName: "checkpoint.gguf",
    sizeBytes: 4,
    sha256: "0".repeat(64),
  };
  const fetchImpl = async () => new Response(Buffer.from("data"), {
    status: 206,
    headers: { "content-range": "bytes 0-3/4" },
  });
  const outputPath = path.join(
    hubRoot,
    "civitai-2053259-2540892",
    "checkpoints",
    `${file.fileName}.part`,
  );
  try {
    await assert.rejects(
      downloadFile(file, { hubRoot, headers: {}, fetchImpl }),
      /failed its pinned size or SHA-256 check/,
    );
    assert.equal(fs.existsSync(outputPath), false);
    assert.equal(fs.existsSync(`${outputPath}.parts`), false);
  } finally {
    fs.rmSync(hubRoot, { recursive: true, force: true });
  }
});
