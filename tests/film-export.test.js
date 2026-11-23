"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFileSync, spawnSync } = require("node:child_process");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const { ProjectVault } = require("../gateway/project-vault");
const { FilmExporter, buildSrt } = require("../gateway/film-export");

const ffmpeg = process.env.FFMPEG_PATH || "ffmpeg";
const ffprobe = process.env.FFPROBE_PATH || "ffprobe";
const ffmpegAvailable = spawnSync(ffmpeg, ["-version"], { stdio: "ignore" }).status === 0;

test("SRT cues follow actual segment durations and skip silent shots", () => {
  const srt = buildSrt([
    { duration: 2.5, narration: "First line." },
    { duration: 1.25, narration: "" },
    { duration: 3, narration: "Second line." },
  ]);
  assert.match(srt, /00:00:00,000 --> 00:00:02,500/);
  assert.match(srt, /00:00:03,750 --> 00:00:06,750/);
  assert.equal((srt.match(/^\d+$/gm) || []).length, 2);
});

test("export assembles mixed video and still-image shots into synchronized 540p MP4 and SRT", {
  skip: !ffmpegAvailable,
  timeout: 120_000,
}, async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-film-test-"));
  const sourceDir = path.join(root, "source");
  fs.mkdirSync(sourceDir);
  const db = new DatabaseSync(":memory:");
  const vault = new ProjectVault({ db, projectsRoot: path.join(root, "projects") });
  const project = vault.createProject({ name: "A Short Film" });
  const episode = vault.createEpisode(project.id, { number: 1, title: "First Light" });
  const scene = vault.createScene(project.id, episode.id, { number: 1, title: "The coast" });
  const firstVideo = path.join(sourceDir, "first.mp4");
  const secondImage = path.join(sourceDir, "second.png");
  const firstAudio = path.join(sourceDir, "first.mp3");
  const secondAudio = path.join(sourceDir, "second.mp3");
  try {
    execFileSync(ffmpeg, [
      "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=navy:s=64x36:r=8",
      "-t", "1.2", "-c:v", "libx264", "-pix_fmt", "yuv420p", firstVideo,
    ]);
    execFileSync(ffmpeg, [
      "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=orange:s=64x36",
      "-frames:v", "1", secondImage,
    ]);
    for (const [file, hz, duration] of [[firstAudio, "440", "1.5"], [secondAudio, "660", "0.8"]]) {
      execFileSync(ffmpeg, [
        "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", `sine=frequency=${hz}:duration=${duration}`,
        "-c:a", "libmp3lame", file,
      ]);
    }

    const firstVideoAsset = vault.addAssetFromPath({
      projectId: project.id, category: "videos", filePath: firstVideo, mime: "video/mp4",
    });
    const stillAsset = vault.addAssetFromPath({
      projectId: project.id, category: "images", filePath: secondImage, mime: "image/png",
    });
    const firstAudioAsset = vault.addAssetFromPath({
      projectId: project.id, category: "audio", filePath: firstAudio, mime: "audio/mpeg",
    });
    const secondAudioAsset = vault.addAssetFromPath({
      projectId: project.id, category: "audio", filePath: secondAudio, mime: "audio/mpeg",
    });
    const firstShot = vault.createShot(project.id, scene.id, {
      number: 1,
      title: "Moving light",
      durationSeconds: 2,
      segmentAssetId: firstVideoAsset.id,
      metadata: {
        narration: "The sea answered.",
        narrationAssetId: firstAudioAsset.id,
        renderBatchId: "batch-a",
        lipSyncEnabled: true,
      },
    });
    vault.createShot(project.id, scene.id, {
      number: 2,
      title: "The lantern",
      durationSeconds: 2,
      keyframeAssetId: stillAsset.id,
      metadata: { narration: "Mira followed it.", narrationAssetId: secondAudioAsset.id },
    });
    assert.ok(firstShot.id);

    const lipSyncCalls = [];
    const exporter = new FilmExporter({
      vault,
      env: { FFMPEG_PATH: ffmpeg, FFPROBE_PATH: ffprobe },
      lipSync: {
        async run(request) {
          lipSyncCalls.push(request);
          return { filePath: firstVideo };
        },
      },
    });
    const native = await exporter.exportEpisode({ episodeId: episode.id, resolution: "480p" });
    assert.equal(native.width, 832);
    assert.equal(native.height, 480);
    assert.equal(vault.getAsset(native.video.id).metadata.scaling, "none");
    assert.deepEqual(lipSyncCalls[0], {
      videoPath: firstVideoAsset.filePath,
      audioPath: firstAudioAsset.filePath,
      shotId: firstShot.id,
    });

    const result = await exporter.exportEpisode({ episodeId: episode.id, resolution: "540p" });
    assert.equal(result.width, 960);
    assert.equal(result.height, 540);
    assert.equal(result.shotCount, 2);
    assert.ok(result.durationSeconds >= 3.5);
    assert.ok(fs.existsSync(result.video.filePath));
    assert.ok(fs.existsSync(result.subtitles.filePath));
    const probe = JSON.parse(execFileSync(ffprobe, [
      "-v", "error", "-show_streams", "-show_format", "-of", "json", result.video.filePath,
    ], { encoding: "utf8" }));
    const video = probe.streams.find((stream) => stream.codec_type === "video");
    const audio = probe.streams.find((stream) => stream.codec_type === "audio");
    const subtitles = probe.streams.find((stream) => stream.codec_type === "subtitle");
    assert.equal(video.width, 960);
    assert.equal(video.height, 540);
    assert.ok(audio, "the export must contain audio");
    assert.equal(subtitles.codec_name, "mov_text");
    const completedBeats = await exporter.exportEpisode({
      episodeId: episode.id,
      resolution: "540p",
      renderedOnly: true,
    });
    assert.equal(completedBeats.shotCount, 1);
    assert.equal(completedBeats.renderedOnly, true);
    vault.createShot(project.id, scene.id, {
      number: 3,
      title: "A later batch",
      segmentAssetId: firstVideoAsset.id,
      metadata: { renderBatchId: "batch-b" },
    });
    const firstBatch = await exporter.exportEpisode({
      episodeId: episode.id,
      resolution: "540p",
      renderedOnly: true,
      batchId: "batch-a",
    });
    const laterBatch = await exporter.exportEpisode({
      episodeId: episode.id,
      resolution: "540p",
      renderedOnly: true,
      batchId: "batch-b",
    });
    assert.equal(firstBatch.shotCount, 1, "batch exports must exclude completed clips from other batches");
    assert.equal(laterBatch.shotCount, 1, "each batch export must contain only its own completed clips");
    vault.createShot(project.id, scene.id, { number: 4, title: "Unrendered shot" });
    await assert.rejects(
      exporter.exportEpisode({ episodeId: episode.id, resolution: "540p" }),
      /needs a rendered video or an attached still/,
    );
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
