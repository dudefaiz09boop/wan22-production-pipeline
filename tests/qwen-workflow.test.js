const assert = require("node:assert/strict");
const test = require("node:test");
const { buildQwenGraph, resolveFamilyPreset } = require("../gateway/qwen-workflow");

const QWEN_CHECKPOINT = "qwen-image-2.1-step90000-FP8-comfy.safetensors";
const MOSS_CHECKPOINT = "Jib_Mix_Moss_V2_H_int8_00001_.safetensors";
const KREA2_CHECKPOINT = "Ultra Realistic by Stable Yogi Krea2 V1.0 Civ INT8-ConvRot C89.safetensors";

test("buildQwenGraph wires a Qwen-Image-2.1 checkpoint with real ComfyUI nodes", () => {
  const { graph, preset, seed } = buildQwenGraph({
    prompt: "a tiny dragon",
    negativePrompt: "blurry",
    width: 1024,
    height: 1024,
    steps: 30,
    seed: 42,
    cfg: 3.5,
    checkpoint: QWEN_CHECKPOINT,
  });
  assert.equal(preset.family, "qwen21");
  assert.equal(seed, 42);
  assert.equal(graph["1"].class_type, "UNETLoader");
  assert.equal(graph["1"].inputs.unet_name, QWEN_CHECKPOINT);
  assert.equal(graph["2"].class_type, "CLIPLoader");
  assert.equal(graph["2"].inputs.type, "qwen_image");
  assert.equal(graph["3"].class_type, "VAELoader");
  assert.equal(graph["4"].class_type, "TextEncodeQwenImage21");
  assert.equal(graph["4"].inputs.prompt, "a tiny dragon");
  assert.equal(graph["4"].inputs.negative_prompt, "blurry");
  assert.equal(graph["4c"].class_type, "EmptyLatentImage");
  assert.deepEqual(graph["5"].inputs.positive, ["4", 0]);
  assert.deepEqual(graph["5"].inputs.negative, ["4", 1]);
  assert.deepEqual(graph["5"].inputs.latent_image, ["4c", 0]);
  assert.equal(graph["5"].class_type, "KSampler");
  assert.equal(graph["6"].class_type, "VAEDecode");
  assert.equal(graph["7"].class_type, "SaveImage");
  assert.ok(!Object.values(graph).some((node) => node.class_type === "LoadCheckpoint"));
});

test("buildQwenGraph applies the checkpoint's recommended defaults when settings are omitted", () => {
  const { graph } = buildQwenGraph({ prompt: "test", checkpoint: QWEN_CHECKPOINT });
  assert.equal(graph["5"].inputs.steps, 35);
  assert.equal(graph["5"].inputs.cfg, 5);
  assert.equal(graph["5"].inputs.sampler_name, "er_sde");
  assert.equal(graph["5"].inputs.scheduler, "simple");
  assert.equal(graph["4"].inputs.negative_prompt, " ");
});

test("buildQwenGraph wires a MOSS-Image checkpoint with its own CLIP type and latent node", () => {
  const { graph, preset } = buildQwenGraph({
    prompt: "a studio portrait",
    negativePrompt: "",
    width: 1024,
    height: 1536,
    steps: 12,
    seed: 7,
    cfg: 1,
    checkpoint: MOSS_CHECKPOINT,
  });
  assert.equal(preset.family, "moss");
  assert.equal(graph["2"].inputs.type, "moss_image");
  assert.equal(graph["4"].class_type, "CLIPTextEncode");
  assert.equal(graph["4b"].class_type, "CLIPTextEncode");
  assert.equal(graph["4c"].class_type, "EmptyFlux2LatentImage");
  assert.equal(graph["4c"].inputs.width, 1024);
  assert.equal(graph["4c"].inputs.height, 1536);
  assert.deepEqual(graph["5"].inputs.positive, ["4", 0]);
  assert.deepEqual(graph["5"].inputs.negative, ["4b", 0]);
  assert.deepEqual(graph["5"].inputs.latent_image, ["4c", 0]);
});

test("buildQwenGraph applies the MOSS author's defaults with a core scheduler", () => {
  const { graph } = buildQwenGraph({ prompt: "test", checkpoint: MOSS_CHECKPOINT });
  assert.equal(graph["5"].inputs.steps, 12);
  assert.equal(graph["5"].inputs.cfg, 1);
  assert.equal(graph["5"].inputs.sampler_name, "er_sde");
  assert.equal(graph["5"].inputs.scheduler, "simple");
});

test("buildQwenGraph wires a Krea-2 checkpoint with the official Krea CLIP type and latent node", () => {
  const { graph, preset } = buildQwenGraph({
    prompt: "a studio portrait",
    negativePrompt: "blurry, low quality",
    width: 1024,
    height: 1536,
    steps: 12,
    seed: 9,
    cfg: 1.5,
    checkpoint: KREA2_CHECKPOINT,
  });
  assert.equal(preset.family, "krea2");
  assert.equal(graph["2"].inputs.type, "krea2");
  assert.equal(graph["4"].class_type, "CLIPTextEncode");
  assert.equal(graph["4b"].class_type, "CLIPTextEncode");
  assert.equal(graph["4c"].class_type, "EmptyLatentImage");
  assert.equal(graph["4c"].inputs.width, 1024);
  assert.equal(graph["4c"].inputs.height, 1536);
  assert.deepEqual(graph["5"].inputs.positive, ["4", 0]);
  assert.deepEqual(graph["5"].inputs.negative, ["4b", 0]);
  assert.deepEqual(graph["5"].inputs.latent_image, ["4c", 0]);
});

test("buildQwenGraph wires allowlisted Krea-2 LoRAs model-only", () => {
  const { graph } = buildQwenGraph({
    prompt: "a portrait",
    checkpoint: KREA2_CHECKPOINT,
    lora: "krea2_darkbrush.safetensors",
    loraStrength: 0.8,
  });
  assert.equal(graph["1l"].class_type, "LoraLoaderModelOnly");
  assert.equal(graph["1l"].inputs.lora_name, "krea2_darkbrush.safetensors");
  assert.equal(graph["1l"].inputs.strength_model, 0.8);
  assert.deepEqual(graph["5"].inputs.model, ["1l", 0]);
  assert.match(graph["4"].inputs.text, /monochrome ink wash style/);
});

test("buildQwenGraph wires Krea-2 reference-image conditioning", () => {
  const { graph } = buildQwenGraph({
    prompt: "a portrait",
    checkpoint: KREA2_CHECKPOINT,
    referenceImageName: "krea2_reference_image.png",
  });
  assert.equal(graph["4img"].class_type, "LoadImage");
  assert.equal(graph["4"].class_type, "TextEncodeQwenImageEditPlus");
  assert.equal(graph["4ref"].class_type, "FluxKontextMultiReferenceLatentMethod");
  assert.deepEqual(graph["4ref"].inputs, {
    conditioning: ["4", 0],
    reference_latents_method: "index_timestep_zero",
  });
  assert.equal(Object.hasOwn(graph["4ref"].inputs, "latent_method"), false);
  assert.equal(graph["4b"].class_type, "ConditioningZeroOut");
  assert.deepEqual(graph["5"].inputs.positive, ["4ref", 0]);
  assert.equal(graph["5"].inputs.denoise, 0.65);
  assert.throws(() => buildQwenGraph({
    prompt: "a portrait",
    checkpoint: KREA2_CHECKPOINT,
    referenceImageName: "krea2_reference_image.png",
    referenceStrength: 1,
  }), /reference_strength must be from 0.2 to 0.9/);
  const adjusted = buildQwenGraph({
    prompt: "a portrait",
    checkpoint: KREA2_CHECKPOINT,
    referenceImageName: "krea2_reference_image.png",
    referenceStrength: 0.35,
  });
  assert.equal(adjusted.graph["5"].inputs.denoise, 0.35);
});

test("buildQwenGraph keeps multiple character references separate in Krea-2 conditioning", () => {
  const { graph } = buildQwenGraph({
    prompt: "new scene with the same two characters",
    checkpoint: KREA2_CHECKPOINT,
    referenceImageNames: ["character-a.png", "character-b.jpg", "scene.png"],
  });
  assert.equal(graph["4img1"].inputs.image, "character-a.png");
  assert.equal(graph["4img2"].inputs.image, "character-b.jpg");
  assert.equal(graph["4img3"].inputs.image, "scene.png");
  assert.deepEqual(graph["4"].inputs.image1, ["4img1", 0]);
  assert.deepEqual(graph["4"].inputs.image2, ["4img2", 0]);
  assert.deepEqual(graph["4"].inputs.image3, ["4img3", 0]);
  assert.throws(() => buildQwenGraph({
    prompt: "too many references",
    checkpoint: KREA2_CHECKPOINT,
    referenceImageNames: ["a.png", "b.png", "c.png", "d.png"],
  }), /up to three reference images/);
});

test("buildQwenGraph uses the previous frame as a low-denoise Krea-2 handoff", () => {
  const { graph } = buildQwenGraph({
    prompt: "continue the shot",
    checkpoint: KREA2_CHECKPOINT,
    referenceImageName: "handoff.png",
    inpaint: true,
  });
  assert.equal(graph["4c"].class_type, "VAEEncodeForInpaint");
  assert.deepEqual(graph["4c"].inputs.pixels, ["4img", 0]);
  assert.deepEqual(graph["4c"].inputs.vae, ["3", 0]);
  assert.deepEqual(graph["4c"].inputs.mask, ["4mask", 0]);
  assert.equal(graph["4mask"].class_type, "SolidMask");
  assert.equal(graph["4mask"].inputs.value, 1);
  assert.equal(graph["5"].inputs.denoise, 0.25);
  assert.throws(() => buildQwenGraph({
    prompt: "continue the shot",
    checkpoint: KREA2_CHECKPOINT,
    inpaint: true,
  }), /requires a Krea-2 reference image/);
});

test("buildQwenGraph applies the Krea-2 defaults from the official turbo template", () => {
  const { graph } = buildQwenGraph({ prompt: "test", checkpoint: KREA2_CHECKPOINT });
  assert.equal(graph["5"].inputs.steps, 12);
  assert.equal(graph["5"].inputs.cfg, 1.5);
  assert.equal(graph["5"].inputs.sampler_name, "euler");
  assert.equal(graph["5"].inputs.scheduler, "simple");
});

test("buildQwenGraph generates a random seed when none is supplied", () => {
  const { seed } = buildQwenGraph({
    prompt: "test",
    checkpoint: QWEN_CHECKPOINT,
  });
  assert.ok(Number.isInteger(seed) && seed >= 0 && seed <= 0x7fffffff);
});

test("resolveFamilyPreset maps the old scaffold names to the real files", () => {
  const qwen = resolveFamilyPreset("qwen-image-21-uncensored-omni-quantized.safetensors", {});
  assert.equal(qwen.family, "qwen21");
  assert.equal(qwen.unetFilename, QWEN_CHECKPOINT);
  const moss = resolveFamilyPreset("jib-mix-moss-realistic.safetensors", {});
  assert.equal(moss.family, "moss");
  assert.equal(moss.unetFilename, MOSS_CHECKPOINT);
  const krea = resolveFamilyPreset("stable-yogi-krea2.safetensors", {});
  assert.equal(krea.family, "krea2");
  assert.equal(krea.unetFilename, KREA2_CHECKPOINT);
});

test("resolveFamilyPreset allows overriding clip/vae filenames via env", () => {
  const preset = resolveFamilyPreset(QWEN_CHECKPOINT, {
    QWEN_CLIP_FILENAME: "custom-clip.safetensors",
    QWEN_VAE_FILENAME: "custom-vae.safetensors",
  });
  assert.equal(preset.clipFilename, "custom-clip.safetensors");
  assert.equal(preset.vaeFilename, "custom-vae.safetensors");
});

test("resolveFamilyPreset allows overriding Krea-2 companion filenames via env", () => {
  const preset = resolveFamilyPreset(KREA2_CHECKPOINT, {
    KREA2_CLIP_FILENAME: "custom-krea-clip.safetensors",
    KREA2_VAE_FILENAME: "custom-krea-vae.safetensors",
  });
  assert.equal(preset.clipFilename, "custom-krea-clip.safetensors");
  assert.equal(preset.vaeFilename, "custom-krea-vae.safetensors");
});
