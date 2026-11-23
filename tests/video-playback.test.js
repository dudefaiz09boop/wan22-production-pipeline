"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { VideoPlayback } = require("../gateway/video-playback");

test("creates a cached browser-compatible MP4 without deleting the original", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-playback-"));
  const source = path.join(root, "render.webm");
  fs.writeFileSync(source, "source-video");
  let conversions = 0;
  const playback = new VideoPlayback({
    execFileImpl: async (executable, args) => {
      conversions += 1;
      assert.equal(executable, "ffmpeg-test");
      assert.ok(args.includes("libx264"));
      assert.ok(args.includes("+faststart"));
      await new Promise((resolve) => setTimeout(resolve, 5));
      fs.writeFileSync(args.at(-1), "mp4-video");
      return { stdout: "", stderr: "" };
    },
    ffmpeg: "ffmpeg-test",
  });
  try {
    const [first, second] = await Promise.all([playback.mp4Path(source), playback.mp4Path(source)]);
    assert.equal(first, path.join(root, "render.playback.mp4"));
    assert.equal(second, first);
    assert.equal(conversions, 1);
    assert.equal(fs.readFileSync(first, "utf8"), "mp4-video");
    assert.equal(fs.readFileSync(source, "utf8"), "source-video");
    assert.equal(await playback.mp4Path(source), first);
    assert.equal(conversions, 1);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("passes through MP4 sources without invoking FFmpeg", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-video-mp4-"));
  const source = path.join(root, "render.MP4");
  fs.writeFileSync(source, "source-mp4");
  const playback = new VideoPlayback({
    execFileImpl: async () => {
      assert.fail("FFmpeg must not be invoked for an MP4 source");
    },
  });
  try {
    assert.equal(await playback.mp4Path(source), source);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
