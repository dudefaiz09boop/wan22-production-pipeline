"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  VIDEO_PROFILES,
  DEFAULT_REVISION,
  MAX_VIDEO_FRAMES,
  resolveVideoProfile,
  assertProfileFits,
  buildVideoGraph,
} = require("../gateway/video-workflow");

test("resolveVideoProfile defaults to the required Wan 2.2 Fast Move Q8 I2V profile", () => {
  const profile = resolveVideoProfile({});
  assert.equal(profile.id, "wan22-fm-lightning-q8");
  assert.equal(profile.conditioning, "image");
  assert.equal(profile.imageConditioning, "wan22-lightning");
  assert.equal(profile.clipVision, null);
  assert.equal(profile.steps, 8);
  assert.equal(profile.cfg, 1);
  assert.equal(profile.width, 832);
  assert.equal(profile.height, 480);
  assert.equal(profile.sampler, "euler");
  assert.equal(profile.scheduler, "simple");
  assert.equal(profile.revision, "ee6f4a40737a995bf5818954cfce6d59443b0f04");
  assert.match(profile.revision, /^[a-f0-9]{40}$/);
});

test("image-to-video is opt-in and honours a pinned session profile", () => {
  const enabled = resolveVideoProfile({ KAGGLE_VIDEO_I2V_ENABLED: "true" });
  assert.equal(enabled.id, "wan-i2v-14b-480p-fp8");
  assert.equal(enabled.conditioning, "image");
  assert.ok(enabled.clipVision);

  const pinned = resolveVideoProfile({ KAGGLE_VIDEO_I2V_ENABLED: "false" }, "wan-i2v-14b-480p-fp8");
  assert.equal(pinned.id, "wan-i2v-14b-480p-fp8");
  assert.equal(pinned.revision, DEFAULT_REVISION);
});

test("Wan VACE reference-image profile fits the default budget and keeps the model revision pinned", () => {
  const profile = resolveVideoProfile({}, "wan-vace-1.3b");
  assert.equal(profile.conditioning, "image");
  assert.equal(profile.imageConditioning, "vace-reference");
  assert.equal(profile.diffusion, "wan2.1_vace_1.3B_fp16.safetensors");
  assert.equal(profile.revision, DEFAULT_REVISION);
  assert.equal(profile.lora, null);
  assert.ok(profile.modelBytes / 1024 ** 3 < 11);
  assert.equal(assertProfileFits(profile, 20).id, "wan-vace-1.3b");

  const experimental = resolveVideoProfile(
    { KAGGLE_VIDEO_HIRESFIX_LORA_ENABLED: "true" },
    "wan-vace-1.3b",
  );
  assert.equal(experimental.label, "Wan 2.1 VACE 1.3B + experimental high-resolution LoRA");
  assert.equal(experimental.lora.fileName, "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors");
  assert.equal(experimental.lora.sha256, "BC5F39B3A6E55FCBF4F3C84806CB37B324996ADB0D2A6EE9E9B9789E23948515");
  assert.equal(experimental.lora.strength, 3);
  assert.ok(experimental.modelBytes / 1024 ** 3 < 20);
  assert.equal(assertProfileFits(experimental, 20).id, "wan-vace-1.3b");
});

test("the imported Wan 1.3B v14 checkpoint is an experimental text-only T2V profile", () => {
  const profile = resolveVideoProfile({}, "wan-nsfw-t2v-1.3b");
  assert.equal(profile.conditioning, "text");
  assert.equal(profile.imageConditioning, undefined);
  assert.equal(profile.diffusion, "wan_1.3B_exp_e14.safetensors");
  assert.equal(profile.modelHubCheckpoint.modelId, 1697081);
  assert.equal(profile.modelHubCheckpoint.versionId, 1936498);
  assert.equal(profile.modelHubCheckpoint.sizeBytes, 2_838_077_328);
  assert.equal(profile.modelHubCheckpoint.sha256, "86DAED5F43A68D7BE2CA1F0E85D2AC51F3BA7FB735C76FD9574323E18FE40C6C");
  assert.equal(assertProfileFits(profile, 20).id, "wan-nsfw-t2v-1.3b");

  const result = buildVideoGraph({
    profile,
    prompt: "A courier walks slowly toward the camera in a rainy alley",
    frames: 17,
    fps: 8,
    steps: 20,
    seed: 42,
  });
  assert.equal(result.width, 832);
  assert.equal(result.height, 480);
  assert.equal(result.graph["6"].class_type, "EmptyHunyuanLatentVideo");
  assert.ok(!Object.values(result.graph).some((node) => node.class_type === "LoadImage"));
});

test("Wan 2.2 Fast Move Q4KM + SVI is a pinned experimental I2V profile", () => {
  const profile = resolveVideoProfile({
    KAGGLE_VIDEO_REPO: "wrong/repo",
    KAGGLE_VIDEO_MODEL_REVISION: "f".repeat(40),
    KAGGLE_VIDEO_DIFFUSION: "wrong.safetensors",
    KAGGLE_VIDEO_TEXT_ENCODER: "wrong-encoder.safetensors",
    KAGGLE_VIDEO_VAE: "wrong-vae.safetensors",
    KAGGLE_VIDEO_CLIP_VISION: "wrong-clip.safetensors",
  }, "wan22-svi-q4km");
  assert.equal(profile.repo, "Comfy-Org/Wan-2.2");
  assert.equal(profile.revision, "ee6f4a40737a995bf5818954cfce6d59443b0f04");
  assert.equal(profile.textEncoder, "umt5_xxl_fp8_e4m3fn_scaled.safetensors");
  assert.equal(profile.vae, "wan_2.1_vae.safetensors");
  assert.equal(profile.clipVision, null);
  assert.equal(profile.conditioning, "image");
  assert.equal(profile.imageConditioning, "wan22-svi");
  assert.equal(profile.width, 480);
  assert.equal(profile.height, 320);
  assert.equal(profile.shotSettings.steps, 6);
  assert.equal(profile.shotSettings.cfg, 1);
  assert.equal(profile.defaultModelBudgetGiB, 32);
  assert.equal(profile.videoLoraFiles.length, 4);
  assert.deepEqual(
    profile.videoLoraFiles.map((item) => [item.key, item.stage, item.versionId]),
    [
      ["penis_insert", "high", 2377035],
      ["penis_insert", "low", 2377244],
      ["vag_detail", "high", 3034683],
      ["vag_detail", "low", 3034713],
    ],
  );
  assert.equal(profile.videoLoraFiles[2].baseModel, "Wan Video 2.2 T2V-A14B");
  assert.deepEqual(profile.videoLoraFiles[2].triggerWords, ["Vagina"]);
  assert.equal(profile.modelHubCheckpoints.length, 2);
  assert.ok(profile.modelHubCheckpoints.every((item) => item.sizeBytes < 12 * 1024 ** 3));
  assert.deepEqual(profile.modelHubCheckpoints.map((item) => item.versionId), [2500306, 2500309]);
  assert.deepEqual(profile.extraModelFiles.map((item) => item.target), [
    "loras/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",
    "loras/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors",
  ]);
  assert.deepEqual(profile.extraModelFiles.map((item) => item.revision), [
    "8260d429d19fd7a72304cad059160b95d843913f",
    "8260d429d19fd7a72304cad059160b95d843913f",
  ]);
  assert.deepEqual(profile.extraModelFiles.map((item) => item.sha256), [
    "6ddfead305cb315612bfe47291aada7431ca6bba688e88df3b2b43788ee11679",
    "eae247da55aab0ee67f8ad99fa46aac09b69156e2ae7080b4737d611780aa7af",
  ]);
  assert.equal(profile.ggufExtension.revision, "6ea2651e7df66d7585f6ffee804b20e92fb38b8a");
  assert.ok(profile.modelBytes / 1024 ** 3 < 32);
  assert.equal(assertProfileFits(profile, 32).id, "wan22-svi-q4km");
  assert.throws(() => assertProfileFits(profile, 20), /GiB budget/);
});

test("Wan 2.2 Consistent Face Q4KM + SVI selects the no-Lightning quality variant", () => {
  const profile = resolveVideoProfile({}, "wan22-svi-cf-q4km");
  assert.equal(profile.id, "wan22-svi-cf-q4km");
  assert.equal(profile.conditioning, "image");
  assert.equal(profile.imageConditioning, "wan22-svi");
  assert.equal(profile.steps, 8);
  assert.equal(profile.shotSettings.steps, 8);
  assert.equal(profile.cfg, 1);
  assert.equal(profile.modelHubCheckpoints[0].versionId, 2625965);
  assert.equal(profile.modelHubCheckpoints[0].fileName, "Wan2_2-I2V-A14B-HIGH_consistent_face_Q4KM.gguf");
  assert.equal(profile.modelHubCheckpoints[0].sha256, "32DB0B59BCBBB3FADA671DDDD60C307EB02023748508395B56DD7C019689624E");
  assert.equal(profile.modelHubCheckpoints[1].versionId, 2625966);
  assert.equal(profile.modelHubCheckpoints[1].fileName, "Wan2_2-I2V-A14B-LOW_consistent_face_Q4KM.gguf");
  assert.equal(profile.modelHubCheckpoints[1].sha256, "2397366794AB5F25E66367A49CE60FFAA01618BFDBCAF5906DFEF940CDFD24F4");
  assert.deepEqual(profile.extraModelFiles.map((item) => item.target), [
    "loras/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors",
    "loras/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors",
  ]);
  assert.equal(assertProfileFits(profile, 32).id, "wan22-svi-cf-q4km");
});

test("Wan 2.2 Fast Move Q8 uses pinned Lightning weights and the shot's anatomy LoRAs", () => {
  const profile = resolveVideoProfile({}, "wan22-fm-lightning-q8");
  assert.equal(profile.imageConditioning, "wan22-lightning");
  assert.deepEqual(profile.modelHubCheckpoints.map((item) => item.versionId), [2540892, 2540896]);
  assert.deepEqual(profile.modelHubCheckpoints.map((item) => item.sizeBytes), [
    15_406_619_136,
    15_406_619_136,
  ]);
  assert.equal(profile.modelBytes, 37_802_960_487);
  assert.deepEqual(profile.modelHubCheckpoints.map((item) => item.sha256), [
    "DEB6DCF39FEFEF7850D1DD49A52B70A64C65B8D8E7B5F03BC1B6A541CF58AAE9",
    "B5722E8E01A21F89A1C0B5E9DE9A3B907CE92324D1951D5CE70307956790DAAE",
  ]);
  assert.equal(profile.videoLoraFiles.length, 4);
  assert.equal(profile.dualGpu, undefined);

  const result = buildVideoGraph({
    profile,
    prompt: "At 0 seconds: static, subject, action, lighting",
    width: 832,
    height: 480,
    frames: 41,
    fps: 8,
    steps: 5,
    cfg: 1,
    seed: 42,
    keyframeFilename: "keyframe.jpg",
    loras: ["vag_detail"],
  });
  assert.equal(result.graph["1"].class_type, "UnetLoaderGGUF");
  assert.equal(result.graph["2"].class_type, "UnetLoaderGGUF");
  assert.equal(result.graph["10"].inputs.width, 832);
  assert.equal(result.graph["10"].inputs.height, 480);
  assert.equal(result.graph["11"].inputs.end_at_step, 2);
  assert.equal(result.graph["12"].inputs.start_at_step, 2);
  assert.equal(result.graph["12"].inputs.end_at_step, 5);
  assert.equal(result.graph["11"].inputs.cfg, 1);
  assert.equal(result.graph["12"].inputs.cfg, 1);
  assert.equal(result.graph["13"].class_type, "VAEDecodeTiled");
  assert.deepEqual(result.graph["13"].inputs, {
    samples: ["12", 0],
    vae: ["6", 0],
    tile_size: 512,
    overlap: 64,
    temporal_size: 16,
    temporal_overlap: 4,
  });
  assert.ok(result.requiredNodes.includes("VAEDecodeTiled"));
  const loraNodes = Object.values(result.graph).filter((node) => node.class_type === "LoraLoaderModelOnly");
  assert.equal(loraNodes.length, 2);
  assert.deepEqual(loraNodes.map((node) => node.inputs.lora_name), [
    "vag_detail_wan22_high.safetensors",
    "vag_detail_wan22_low.safetensors",
  ]);
  assert.deepEqual(loraNodes.map((node) => node.inputs.strength_model), [0.75, 0.75]);
  assert.ok(result.requiredNodes.includes("LoraLoaderModelOnly"));
});

test("dual-GPU Wan 2.2 is opt-in and distributes each GGUF expert across both T4s", () => {
  const profile = resolveVideoProfile({}, "wan22-fm-lightning-q8-dual-gpu");
  assert.equal(profile.id, "wan22-fm-lightning-q8-dual-gpu");
  assert.equal(profile.dualGpu, true);
  assert.equal(profile.dualGpuDistribution, "distorch2");
  assert.equal(profile.multiGpuExtension.revision, "37a73feda5804045e4f39d8c640905d62a2169d1");

  const result = buildVideoGraph({
    profile,
    prompt: "At 0 seconds: static, subject, action, lighting",
    width: 832,
    height: 480,
    frames: 41,
    fps: 8,
    steps: 5,
    cfg: 1,
    seed: 42,
    keyframeFilename: "keyframe.jpg",
    loras: [],
  });

  assert.equal(result.graph["1"].class_type, "UnetLoaderGGUFDisTorch2MultiGPU");
  assert.deepEqual(result.graph["1"].inputs, {
    unet_name: profile.modelHubCheckpoints[0].fileName,
    compute_device: "cuda:0",
    virtual_vram_gb: 0,
    donor_device: "cpu",
    expert_mode_allocations: "cuda:0,8gb;cuda:1,7gb;cpu,*",
    eject_models: true,
  });
  assert.equal(result.graph["2"].class_type, "UnetLoaderGGUFDisTorch2MultiGPU");
  assert.deepEqual(result.graph["2"].inputs, {
    unet_name: profile.modelHubCheckpoints[1].fileName,
    compute_device: "cuda:0",
    virtual_vram_gb: 0,
    donor_device: "cpu",
    expert_mode_allocations: "cuda:0,8gb;cuda:1,7gb;cpu,*",
    eject_models: true,
  });
  assert.equal(result.graph["5"].class_type, "CLIPLoaderMultiGPU");
  assert.deepEqual(result.graph["5"].inputs, {
    clip_name: profile.textEncoder,
    type: "wan",
    device: "cuda:1",
  });
  assert.equal(result.graph["6"].class_type, "VAELoaderMultiGPU");
  assert.deepEqual(result.graph["6"].inputs, {
    vae_name: profile.vae,
    device: "cuda:1",
  });
  assert.ok(result.requiredNodes.includes("UnetLoaderGGUFDisTorch2MultiGPU"));
  assert.ok(result.requiredNodes.includes("CLIPLoaderMultiGPU"));
  assert.ok(result.requiredNodes.includes("VAELoaderMultiGPU"));
});

test("dual-GPU legacy mode remains available as a rollback", () => {
  const profile = resolveVideoProfile(
    { KAGGLE_VIDEO_DUAL_GPU_DISTRIBUTION: "legacy" },
    "wan22-fm-lightning-q8-dual-gpu",
  );
  assert.equal(profile.dualGpuDistribution, "legacy");
  assert.equal(profile.multiGpuExtension, undefined);

  const result = buildVideoGraph({
    profile,
    prompt: "At 0 seconds: static, subject, action, lighting",
    width: 832,
    height: 480,
    frames: 41,
    fps: 8,
    steps: 5,
    cfg: 1,
    seed: 42,
    keyframeFilename: "keyframe.jpg",
    loras: [],
  });

  assert.equal(result.graph["1"].class_type, "LocalAIDualDeviceGGUFLoader");
  assert.deepEqual(result.graph["1"].inputs, {
    unet_name: profile.modelHubCheckpoints[0].fileName,
    device: "cuda:0",
  });
  assert.equal(result.graph["2"].class_type, "LocalAIDualDeviceGGUFLoader");
  assert.deepEqual(result.graph["2"].inputs, {
    unet_name: profile.modelHubCheckpoints[1].fileName,
    device: "cuda:1",
  });
  assert.ok(result.requiredNodes.includes("LocalAIDualDeviceGGUFLoader"));
  assert.throws(
    () => resolveVideoProfile(
      { KAGGLE_VIDEO_DUAL_GPU_DISTRIBUTION: "unknown" },
      "wan22-fm-lightning-q8-dual-gpu",
    ),
    /must be 'distorch2' or 'legacy'/,
  );
});

test("assertProfileFits refuses the 24 GB I2V checkpoint on the default T4 budget", () => {
  assert.throws(
    () => assertProfileFits(VIDEO_PROFILES["wan-i2v-14b-480p-fp8"], 12),
    /GiB budget/,
  );
  assert.equal(assertProfileFits(VIDEO_PROFILES["wan-t2v-1.3b"], 12).id, "wan-t2v-1.3b");
  assert.equal(assertProfileFits(VIDEO_PROFILES["wan-i2v-14b-480p-fp8"], 40).id, "wan-i2v-14b-480p-fp8");
});

test("CogVideoX 2B fits the default Kaggle model-file budget and uses its native dimensions", () => {
  const profile = resolveVideoProfile({}, "cogvideox-2b");
  assert.equal(profile.engine, "diffusers");
  assert.equal(profile.conditioning, "text");
  assert.equal(profile.width, 720);
  assert.equal(profile.height, 480);
  assert.equal(assertProfileFits(profile).id, "cogvideox-2b");
  assert.throws(() => assertProfileFits(profile, 12), /GiB budget/);
  const request = buildVideoGraph({
    profile,
    prompt: "A cinematic mountain pass",
    frames: 41,
    fps: 8,
    steps: 20,
    seed: 1,
  });
  assert.equal(request.graph, null);
  assert.deepEqual(request.requiredNodes, []);
  assert.throws(
    () => buildVideoGraph({
      profile,
      prompt: "A cinematic mountain pass",
      frames: 40,
      fps: 8,
      steps: 20,
      seed: 1,
    }),
    /CogVideoX output frames must be/,
  );
});

test("buildVideoGraph builds the validated T2V graph without image conditioning", () => {
  const result = buildVideoGraph({
    profile: VIDEO_PROFILES["wan-t2v-1.3b"],
    prompt: "a lighthouse in a storm",
    width: 480,
    height: 320,
    frames: 9,
    fps: 8,
    steps: 8,
    seed: 42,
  });
  assert.equal(result.seed, 42);
  assert.equal(result.frames, 9);
  assert.equal(result.keyframeFilename, null);
  assert.equal(result.graph["6"].class_type, "EmptyHunyuanLatentVideo");
  assert.equal(result.graph["9"].class_type, "SaveWEBM");
  assert.equal(result.graph["9"].inputs.fps, 8);
  assert.equal(result.graph["9"].inputs.crf, 18);
  assert.ok(!Object.values(result.graph).some((node) => node.class_type === "LoadImage"));
  assert.deepEqual(result.requiredNodes.includes("EmptyHunyuanLatentVideo"), true);
});

test("buildVideoGraph requires and wires a keyframe for the I2V profile", () => {
  const profile = VIDEO_PROFILES["wan-i2v-14b-480p-fp8"];
  assert.throws(
    () => buildVideoGraph({ profile, prompt: "a shot", frames: 9, fps: 8, steps: 8 }),
    /needs a keyframe image filename/,
  );
  const result = buildVideoGraph({
    profile,
    prompt: "a shot",
    keyframeFilename: "keyframe-123.png",
    frames: 9,
    fps: 8,
    steps: 8,
    seed: 7,
  });
  assert.equal(result.graph["11"].class_type, "LoadImage");
  assert.equal(result.graph["11"].inputs.image, "keyframe-123.png");
  assert.equal(result.graph["10"].class_type, "CLIPVisionLoader");
  assert.equal(result.graph["8"].class_type, "WanImageToVideo");
  assert.deepEqual(result.graph["8"].inputs.start_image, ["11", 0]);
  assert.deepEqual(result.graph["8"].inputs.clip_vision_output, ["10", 0]);
  assert.equal(result.keyframeFilename, "keyframe-123.png");
});

test("buildVideoGraph wires VACE reference conditioning and trims its reference latent", () => {
  const profile = resolveVideoProfile(
    { KAGGLE_VIDEO_HIRESFIX_LORA_ENABLED: "true" },
    "wan-vace-1.3b",
  );
  const result = buildVideoGraph({
    profile,
    prompt: "a lantern bearer walks through a misty forest",
    negativePrompt: "blur, flicker",
    keyframeFilename: "keyframe-123.png",
    frames: 17,
    fps: 8,
    steps: 8,
    seed: 7,
  });
  assert.equal(result.graph["11"].class_type, "LoadImage");
  assert.equal(result.graph["11"].inputs.image, "keyframe-123.png");
  assert.equal(result.graph["6"].class_type, "WanVaceToVideo");
  assert.deepEqual(result.graph["6"].inputs.reference_image, ["11", 0]);
  assert.equal(result.graph["6"].inputs.length, 17);
  assert.equal(result.graph["6"].inputs.strength, 1);
  assert.equal(result.graph["7"].class_type, "KSampler");
  assert.equal(result.graph["14"].class_type, "LoraLoaderModelOnly");
  assert.deepEqual(result.graph["14"].inputs.model, ["1", 0]);
  assert.equal(result.graph["14"].inputs.lora_name, profile.lora.fileName);
  assert.equal(result.graph["14"].inputs.strength_model, 3);
  assert.deepEqual(result.graph["7"].inputs.model, ["14", 0]);
  assert.equal(result.graph["12"].class_type, "TrimVideoLatent");
  assert.deepEqual(result.graph["12"].inputs.samples, ["7", 0]);
  assert.deepEqual(result.graph["12"].inputs.trim_amount, ["6", 3]);
  assert.deepEqual(result.graph["9"].inputs.samples, ["12", 0]);
  assert.equal(result.graph["13"].class_type, "SaveWEBM");
  assert.ok(result.requiredNodes.includes("WanVaceToVideo"));
  assert.ok(result.requiredNodes.includes("TrimVideoLatent"));
  assert.ok(result.requiredNodes.includes("LoraLoaderModelOnly"));
  assert.ok(!result.requiredNodes.includes("CLIPVisionLoader"));
});

test("video graphs preserve negative prompts beyond the former 1000-character cap", () => {
  const profile = resolveVideoProfile({}, "wan-vace-1.3b");
  const negativePrompt = "artifact ".repeat(149);
  const result = buildVideoGraph({
    profile,
    prompt: "a lantern bearer walks through a misty forest",
    negativePrompt,
    keyframeFilename: "keyframe-123.png",
    frames: 17,
    fps: 8,
    steps: 8,
    seed: 7,
  });

  assert.ok(Object.values(result.graph).some((node) => node.inputs?.text === negativePrompt));
  assert.throws(() => buildVideoGraph({
    profile,
    prompt: "a lantern bearer walks through a misty forest",
    negativePrompt: "x".repeat(4097),
    keyframeFilename: "keyframe-123.png",
    frames: 17,
    fps: 8,
    steps: 8,
    seed: 7,
  }), /4096 characters/);
});

test("buildVideoGraph wires Wan 2.2 Q4KM high/low GGUF models and SVI stages", () => {
  const profile = resolveVideoProfile({}, "wan22-svi-q4km");
  assert.throws(
    () => buildVideoGraph({ profile, prompt: "a shot", frames: 9 }),
    /needs a keyframe image filename/,
  );
  const result = buildVideoGraph({
    profile,
    prompt: "a reference-conditioned shot",
    negativePrompt: "blur",
    keyframeFilename: "keyframe-123.png",
    frames: 41,
    fps: 8,
    seed: 7,
  });
  assert.equal(result.width, 480);
  assert.equal(result.height, 320);
  assert.equal(result.steps, 6);
  assert.equal(result.cfg, 1);
  assert.equal(result.graph["1"].class_type, "UnetLoaderGGUF");
  assert.equal(result.graph["1"].inputs.unet_name, profile.modelHubCheckpoints[0].fileName);
  assert.equal(result.graph["2"].inputs.unet_name, profile.modelHubCheckpoints[1].fileName);
  assert.equal(result.graph["3"].inputs.lora_name, profile.extraModelFiles[0].target.slice("loras/".length));
  assert.equal(result.graph["4"].inputs.lora_name, profile.extraModelFiles[1].target.slice("loras/".length));
  assert.equal(result.graph["12"].class_type, "WanImageToVideo");
  assert.equal(result.graph["12"].inputs.length, 41);
  assert.equal(result.graph["13"].class_type, "KSamplerAdvanced");
  assert.equal(result.graph["13"].inputs.end_at_step, 3);
  assert.equal(result.graph["14"].inputs.start_at_step, 3);
  assert.equal(result.graph["14"].inputs.end_at_step, 6);
  assert.equal(result.graph["16"].class_type, "SaveWEBM");
  assert.equal(result.keyframeFilename, "keyframe-123.png");
});

test("Consistent Face SVI graph splits eight quality steps across the high/low stages", () => {
  const profile = resolveVideoProfile({}, "wan22-svi-cf-q4km");
  const result = buildVideoGraph({
    profile,
    prompt: "At 0 seconds, a woman smiles softly; at 5 seconds, she turns toward the camera.",
    keyframeFilename: "juvi.png",
    frames: 41,
    fps: 8,
    seed: 42,
  });
  assert.equal(result.steps, 8);
  assert.equal(result.graph["1"].inputs.unet_name, profile.modelHubCheckpoints[0].fileName);
  assert.equal(result.graph["2"].inputs.unet_name, profile.modelHubCheckpoints[1].fileName);
  assert.equal(result.graph["13"].inputs.end_at_step, 4);
  assert.equal(result.graph["14"].inputs.start_at_step, 4);
  assert.equal(result.graph["14"].inputs.end_at_step, 8);
  assert.equal(result.graph["13"].inputs.cfg, 1);
  assert.equal(result.graph["14"].inputs.cfg, 1);
});

test("Consistent Face SVI loads only each shot's selected anatomy LoRAs on both stages", () => {
  const profile = resolveVideoProfile({}, "wan22-svi-cf-q4km");
  const result = buildVideoGraph({
    profile,
    prompt: "a shot",
    keyframeFilename: "juvi.png",
    frames: 41,
    loras: ["vag_detail", "penis_insert"],
  });
  assert.deepEqual(result.loras, ["penis_insert", "vag_detail"]);
  assert.deepEqual(result.graph["5"].inputs.model, ["19", 0]);
  assert.deepEqual(result.graph["6"].inputs.model, ["20", 0]);
  assert.equal(result.graph["17"].inputs.lora_name, "penis_insert_wan22_high.safetensors");
  assert.equal(result.graph["17"].inputs.strength_model, 0.9);
  assert.equal(result.graph["18"].inputs.lora_name, "penis_insert_wan22_low.safetensors");
  assert.equal(result.graph["19"].inputs.lora_name, "vag_detail_wan22_high.safetensors");
  assert.equal(result.graph["19"].inputs.strength_model, 0.75);
  assert.deepEqual(result.graph["19"].inputs.model, ["17", 0]);
  assert.equal(result.graph["20"].inputs.lora_name, "vag_detail_wan22_low.safetensors");
  assert.equal(result.graph["20"].inputs.strength_model, 0.75);
  assert.deepEqual(result.graph["20"].inputs.model, ["18", 0]);
  assert.throws(
    () => buildVideoGraph({
      profile,
      prompt: "a shot",
      keyframeFilename: "juvi.png",
      frames: 41,
      loras: ["unknown"],
    }),
    /loras may only contain/,
  );
  assert.throws(
    () => buildVideoGraph({
      profile: VIDEO_PROFILES["wan-t2v-1.3b"],
      prompt: "a shot",
      frames: 9,
      loras: ["penis_insert"],
    }),
    /supported only by Fast Move Lightning and SVI profiles/,
  );
});

test("Wan VACE leaves the experimental high-resolution LoRA off by default", () => {
  const profile = resolveVideoProfile(
    {},
    "wan-vace-1.3b",
  );
  assert.equal(profile.label, "Wan 2.1 VACE 1.3B (reference image)");
  assert.ok(profile.modelBytes / 1024 ** 3 < 20);
  const result = buildVideoGraph({
    profile,
    prompt: "a lantern bearer walks through a misty forest",
    keyframeFilename: "keyframe-123.png",
    frames: 17,
    fps: 8,
    steps: 8,
    seed: 7,
  });
  assert.equal(result.graph["7"].inputs.model[0], "1");
  assert.equal(result.graph["14"], undefined);
  assert.ok(!result.requiredNodes.includes("LoraLoaderModelOnly"));
});

test("buildVideoGraph rejects out-of-contract frames, fps, steps and prompts", () => {
  const profile = VIDEO_PROFILES["wan-t2v-1.3b"];
  assert.throws(() => buildVideoGraph({ profile, prompt: "", frames: 9 }), /prompt/);
  assert.throws(() => buildVideoGraph({ profile, prompt: "x", frames: 4 }), /frames/);
  assert.throws(() => buildVideoGraph({ profile, prompt: "x", frames: MAX_VIDEO_FRAMES + 1 }), /frames/);
  assert.equal(buildVideoGraph({ profile, prompt: "x", frames: MAX_VIDEO_FRAMES }).frames, 81);
  assert.throws(() => buildVideoGraph({ profile, prompt: "x", frames: 9, fps: 30 }), /fps/);
  assert.throws(() => buildVideoGraph({ profile, prompt: "x", frames: 9, steps: 100 }), /steps/);
  assert.throws(
    () => buildVideoGraph({
      profile: VIDEO_PROFILES["wan-vace-1.3b"],
      prompt: "x",
      frames: 10,
      keyframeFilename: "keyframe.png",
    }),
    /Wan VACE output frames must be/,
  );
  assert.throws(
    () => buildVideoGraph({ profile: VIDEO_PROFILES["wan-i2v-14b-480p-fp8"], prompt: "x", frames: 9, keyframeFilename: "../evil.png" }),
    /keyframe/,
  );
});
