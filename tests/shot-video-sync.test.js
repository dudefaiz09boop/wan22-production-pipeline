"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { attachCompletedShotVideo, syncShotVideo } = require("../gateway/shot-video-sync");

function makeFixture(job) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-shot-video-sync-"));
  const videoPath = path.join(root, "render.webm");
  fs.writeFileSync(videoPath, "test video");
  const shot = {
    id: "shot-1",
    projectId: "project-1",
    number: 1,
    title: "First Meeting",
    segmentAssetId: null,
    status: "draft",
    metadata: { globalNumber: 3, renderTaskId: job.id, renderTaskStatus: "running" },
  };
  const assets = [];
  let currentJob = job;
  const vault = {
    addAssetFromPath(options) {
      const asset = { id: `asset-${assets.length + 1}`, ...options };
      assets.push(asset);
      return asset;
    },
    updateShot(_id, patch) {
      Object.assign(shot, patch);
      return { ...shot, metadata: { ...shot.metadata } };
    },
    getScene: () => ({ episodeId: "episode-1" }),
  };
  const kaggle = { getJob: (id) => id === currentJob.id ? currentJob : null };
  return { root, shot, assets, vault, kaggle, setJob: (value) => { currentJob = value; } };
}

test("completed video jobs attach to shots once and survive restoration reads", () => {
  const fixture = makeFixture({
    id: "task-1",
    kind: "video",
    status: "completed",
    result: { videoPath: path.join(os.tmpdir(), "unused"), mime: "video/webm" },
  });
  fixture.setJob({
    id: "task-1",
    kind: "video",
    status: "completed",
    result: { videoPath: path.join(fixture.root, "render.webm"), mime: "video/webm" },
  });
  try {
    const attached = syncShotVideo(fixture.vault, fixture.kaggle, fixture.shot);
    assert.equal(attached.segmentAssetId, "asset-1");
    assert.equal(attached.status, "video_ready");
    assert.equal(attached.metadata.renderTaskStatus, "completed");
    assert.equal(attached.metadata.renderSyncedTaskId, "task-1");
    assert.equal(fixture.assets.length, 1);
    assert.equal(fixture.assets[0].name, "Beat 3 - First Meeting.webm");
    assert.equal(fixture.assets[0].metadata.beatNumber, 3);
    assert.equal(syncShotVideo(fixture.vault, fixture.kaggle, attached).segmentAssetId, "asset-1");
    assert.equal(fixture.assets.length, 1, "restoring a production must not duplicate attached videos");
  } finally {
    fs.rmSync(fixture.root, { recursive: true, force: true });
  }
});

test("failed render status is persisted so it remains visible after reopening", () => {
  const fixture = makeFixture({ id: "task-2", kind: "video", status: "failed", error: "Kaggle kernel failed" });
  try {
    const updated = syncShotVideo(fixture.vault, fixture.kaggle, fixture.shot);
    assert.equal(updated.metadata.renderTaskStatus, "failed");
    assert.equal(updated.metadata.renderTaskError, "Kaggle kernel failed");
    assert.equal(fixture.assets.length, 0);
  } finally {
    fs.rmSync(fixture.root, { recursive: true, force: true });
  }
});

test("manual and automatic video attachment use the same idempotent sync path", () => {
  const fixture = makeFixture({
    id: "task-3",
    kind: "video",
    status: "completed",
    result: { videoPath: path.join(os.tmpdir(), "unused"), mime: "video/mp4" },
  });
  fixture.setJob({
    id: "task-3",
    kind: "video",
    status: "completed",
    result: { videoPath: path.join(fixture.root, "render.webm"), mime: "video/mp4" },
  });
  try {
    const attached = attachCompletedShotVideo(fixture.vault, fixture.kaggle, fixture.shot, "task-3");
    assert.equal(attached.segmentAssetId, "asset-1");
    assert.equal(fixture.assets[0].mime, "video/mp4");
    assert.equal(fixture.assets.length, 1);
  } finally {
    fs.rmSync(fixture.root, { recursive: true, force: true });
  }
});
