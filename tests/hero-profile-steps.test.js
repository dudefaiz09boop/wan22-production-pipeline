"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  VIDEO_PROFILES,
  resolveVideoProfile,
  buildVideoGraph,
} = require("../gateway/video-workflow");

test("wan22-fm-hero keeps the pinned Q8 assets and hero-quality sampler settings", () => {
  const hero = resolveVideoProfile({}, "wan22-fm-hero");
  assert.equal(hero.id, "wan22-fm-hero");
  assert.equal(hero.imageConditioning, "wan22-lightning");
  const lightning = resolveVideoProfile({}, "wan22-fm-lightning-q8");
  assert.deepEqual(
    hero.modelHubCheckpoints.map((item) => item.versionId),
    lightning.modelHubCheckpoints.map((item) => item.versionId),
  );
  assert.deepEqual(
    hero.modelHubCheckpoints.map((item) => item.sha256),
    lightning.modelHubCheckpoints.map((item) => item.sha256),
  );
  assert.equal(hero.steps, 12);
  assert.equal(hero.cfg, 4.0);
  assert.equal(hero.sampler, "euler");
  assert.equal(hero.scheduler, "simple");
  assert.deepEqual(hero.shotSettings, {
    width: 832,
    height: 480,
    frames: 81,
    fps: 16,
    steps: 12,
    cfg: 4,
  });
});

test("wan22-fm-hero graphs split twelve steps evenly across both KSamplerAdvanced passes", () => {
  const hero = resolveVideoProfile({}, "wan22-fm-hero");
  const result = buildVideoGraph({
    profile: hero,
    prompt: "At 0 seconds: static, subject, action, lighting",
    width: 832,
    height: 480,
    frames: 41,
    fps: 8,
    seed: 42,
    keyframeFilename: "keyframe.jpg",
    loras: [],
  });
  assert.equal(result.steps, 12);
  assert.equal(result.cfg, 4);
  assert.equal(result.graph["11"].inputs.end_at_step, 6);
  assert.equal(result.graph["11"].inputs.start_at_step, 0);
  assert.equal(result.graph["12"].inputs.start_at_step, 6);
  assert.equal(result.graph["12"].inputs.end_at_step, 12);
  assert.equal(result.graph["11"].inputs.cfg, 4);
  assert.equal(result.graph["12"].inputs.cfg, 4);
  assert.equal(result.graph["11"].inputs.sampler_name, "euler");
  assert.equal(result.graph["11"].inputs.scheduler, "simple");
});

test("the hero profile leaves the lightning and dual-GPU contracts untouched", () => {
  const lightning = resolveVideoProfile({}, "wan22-fm-lightning-q8");
  assert.equal(lightning.steps, 8);
  assert.equal(lightning.cfg, 1);
  const dualGpu = resolveVideoProfile({}, "wan22-fm-lightning-q8-dual-gpu");
  assert.equal(dualGpu.steps, 8);
  assert.equal(dualGpu.cfg, 1);
  assert.equal(dualGpu.dualGpu, true);
  assert.ok(!VIDEO_PROFILES["wan22-fm-hero"].dualGpu);
});
