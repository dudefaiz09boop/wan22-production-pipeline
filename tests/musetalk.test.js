"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { MuseTalkRunner } = require("../gateway/musetalk");

test("MuseTalk runner submits the rendered shot and narration to pinned 1.5 inference", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "musetalk-runner-test-"));
  const repo = path.join(root, "musetalk");
  const modelDir = path.join(repo, "models", "musetalkV15");
  const inputVideo = path.join(root, "shot.mp4");
  const inputAudio = path.join(root, "dialogue.wav");
  fs.mkdirSync(path.join(repo, "scripts"), { recursive: true });
  fs.mkdirSync(modelDir, { recursive: true });
  fs.writeFileSync(path.join(repo, "scripts", "inference.py"), "");
  fs.writeFileSync(path.join(modelDir, "unet.pth"), "weights");
  fs.writeFileSync(path.join(modelDir, "musetalk.json"), "{}");
  fs.writeFileSync(inputVideo, "video");
  fs.writeFileSync(inputAudio, "audio");
  let outputPath;

  try {
    const runner = new MuseTalkRunner({
      env: { MUSETALK_ROOT: repo, FFMPEG_PATH: "C:\\tools\\ffmpeg\\ffmpeg.exe" },
      execFileImpl: async (python, args, options) => {
        assert.equal(python, "python");
        assert.equal(options.cwd, repo);
        assert.ok(args.includes("--use_float16"));
        assert.equal(args[args.indexOf("--version") + 1], "v15");
        assert.equal(args[args.indexOf("--ffmpeg_path") + 1], "C:\\tools\\ffmpeg");
        const configPath = args[args.indexOf("--inference_config") + 1];
        const config = fs.readFileSync(configPath, "utf8");
        assert.ok(config.includes(JSON.stringify(inputVideo)));
        assert.ok(config.includes(JSON.stringify(inputAudio)));
        outputPath = path.join(args[args.indexOf("--result_dir") + 1], "v15", "musetalk-output.mp4");
        fs.mkdirSync(path.dirname(outputPath), { recursive: true });
        fs.writeFileSync(outputPath, "lip-synced video");
      },
    });

    const result = await runner.run({ videoPath: inputVideo, audioPath: inputAudio, shotId: "shot-1" });
    assert.equal(result.filePath, outputPath);
    assert.equal(fs.readFileSync(result.filePath, "utf8"), "lip-synced video");
    result.cleanup();
    assert.equal(fs.existsSync(outputPath), false);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("MuseTalk runner fails explicitly when its prepared model files are missing", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "musetalk-missing-test-"));
  const repo = path.join(root, "musetalk");
  fs.mkdirSync(path.join(repo, "scripts"), { recursive: true });
  fs.writeFileSync(path.join(repo, "scripts", "inference.py"), "");
  const videoPath = path.join(root, "shot.mp4");
  const audioPath = path.join(root, "dialogue.wav");
  fs.writeFileSync(videoPath, "video");
  fs.writeFileSync(audioPath, "audio");

  try {
    const runner = new MuseTalkRunner({ env: { MUSETALK_ROOT: repo } });
    await assert.rejects(
      runner.run({ videoPath, audioPath }),
      /MuseTalk 1.5 weights is missing/,
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
