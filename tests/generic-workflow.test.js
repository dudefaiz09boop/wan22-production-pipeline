const assert = require("node:assert/strict");
const test = require("node:test");
const { buildGenericGraph } = require("../gateway/generic-workflow");

test("buildGenericGraph wires a plain single-file checkpoint text-to-image graph", () => {
  const { graph, seed } = buildGenericGraph({
    prompt: "a cabin in the woods",
    negativePrompt: "blurry",
    checkpointFilename: "example-checkpoint.safetensors",
    steps: 22,
    cfg: 6.5,
    seed: 7,
  });
  assert.equal(seed, 7);
  assert.equal(graph["1"].class_type, "CheckpointLoaderSimple");
  assert.equal(graph["1"].inputs.ckpt_name, "example-checkpoint.safetensors");
  assert.equal(graph["2"].class_type, "CLIPTextEncode");
  assert.equal(graph["2"].inputs.text, "a cabin in the woods");
  assert.equal(graph["3"].inputs.text, "blurry");
  assert.equal(graph["4"].class_type, "EmptyLatentImage");
  assert.equal(graph["5"].class_type, "KSampler");
  assert.equal(graph["5"].inputs.steps, 22);
  assert.equal(graph["5"].inputs.cfg, 6.5);
  assert.equal(graph["5"].inputs.denoise, 1);
  assert.deepEqual(graph["5"].inputs.model, ["1", 0]);
  assert.equal(graph["7"].class_type, "SaveImage");
});

test("buildGenericGraph chains LoRAs in order with independent strengths", () => {
  const { graph } = buildGenericGraph({
    prompt: "test",
    checkpointFilename: "ckpt.safetensors",
    loraFilenames: ["a.safetensors", "b.safetensors"],
    loraStrengths: [0.5, 1.2],
  });
  assert.equal(graph["1l0"].class_type, "LoraLoader");
  assert.equal(graph["1l0"].inputs.lora_name, "a.safetensors");
  assert.equal(graph["1l0"].inputs.strength_model, 0.5);
  assert.deepEqual(graph["1l0"].inputs.model, ["1", 0]);
  assert.equal(graph["1l1"].class_type, "LoraLoader");
  assert.equal(graph["1l1"].inputs.lora_name, "b.safetensors");
  assert.deepEqual(graph["1l1"].inputs.model, ["1l0", 0]);
  assert.deepEqual(graph["5"].inputs.model, ["1l1", 0]);
});

test("buildGenericGraph preserves zero LoRA strength instead of silently re-enabling it", () => {
  const { graph } = buildGenericGraph({
    prompt: "test",
    checkpointFilename: "ckpt.safetensors",
    loraFilenames: ["off.safetensors"],
    loraStrengths: [0],
  });
  assert.equal(graph["1l0"].inputs.strength_model, 0);
  assert.equal(graph["1l0"].inputs.strength_clip, 0);
});

test("buildGenericGraph switches to an image-to-image graph when a reference image is supplied", () => {
  const { graph } = buildGenericGraph({
    prompt: "test",
    checkpointFilename: "ckpt.safetensors",
    referenceImageName: "reference.png",
    denoise: 0.5,
  });
  assert.equal(graph["4img"].class_type, "LoadImage");
  assert.equal(graph["4"].class_type, "VAEEncode");
  assert.deepEqual(graph["4"].inputs.pixels, ["4img", 0]);
  assert.deepEqual(graph["5"].inputs.latent_image, ["4", 0]);
  assert.equal(graph["5"].inputs.denoise, 0.5);
});

test("buildGenericGraph rejects more than four chained LoRAs and bad reference filenames", () => {
  assert.throws(
    () => buildGenericGraph({
      prompt: "test",
      checkpointFilename: "ckpt.safetensors",
      loraFilenames: ["a", "b", "c", "d", "e"],
    }),
    /maximum of 4/,
  );
  assert.throws(
    () => buildGenericGraph({
      prompt: "test",
      checkpointFilename: "ckpt.safetensors",
      referenceImageName: "../escape.png",
    }),
    /simple PNG, JPEG, or WebP filename/,
  );
});

test("buildGenericGraph requires a prompt and a checkpoint filename", () => {
  assert.throws(() => buildGenericGraph({ checkpointFilename: "ckpt.safetensors" }), /prompt is required/);
  assert.throws(() => buildGenericGraph({ prompt: "test" }), /checkpointFilename is required/);
});