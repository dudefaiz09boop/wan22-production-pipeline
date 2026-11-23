const assert = require("node:assert/strict");
const test = require("node:test");
const { QwenImage } = require("../gateway/qwen-image");

const KREA2_CHECKPOINT = "Ultra Realistic by Stable Yogi Krea2 V1.0 Civ INT8-ConvRot C89.safetensors";

function fakeFetch(responses) {
  let call = 0;
  return async (url) => {
    const response = responses[call];
    call += 1;
    return response(url);
  };
}

test("QwenImage.submit posts a real ComfyUI graph (no LoadCheckpoint) for Qwen-Image-2.1", async () => {
  let sentBody;
  const fetchImpl = fakeFetch([
    async (url) => {
      assert.match(url, /\/prompt$/);
      return {
        ok: true,
        json: async () => ({ prompt_id: "abc123" }),
      };
    },
  ]);
  const wrappedFetch = async (url, options) => {
    sentBody = JSON.parse(options.body);
    return fetchImpl(url, options);
  };
  const qwen = new QwenImage({
    env: { QWEN_COMFY_URL: "http://127.0.0.1:8188" },
    fetchImpl: wrappedFetch,
  });
  const job = await qwen.submit({
    prompt: "a tiny dragon",
    negativePrompt: "",
    width: 1344,
    height: 1344,
    steps: 30,
    seed: 42,
    cfg: 3.5,
    checkpoint: "qwen-image-21-uncensored-omni-quantized.safetensors",
  });
  assert.equal(job.status, "queued");
  assert.equal(job.prompt_id, "abc123");
  const graph = sentBody.prompt;
  const classTypes = Object.values(graph).map((node) => node.class_type);
  assert.ok(classTypes.includes("UNETLoader"));
  assert.ok(classTypes.includes("CLIPLoader"));
  assert.ok(classTypes.includes("VAELoader"));
  assert.ok(classTypes.includes("TextEncodeQwenImage21"));
  assert.ok(!classTypes.includes("LoadCheckpoint"));
});

test("QwenImage.submit wires MOSS-Image checkpoints to moss_image CLIP type", async () => {
  let sentBody;
  const wrappedFetch = async (url, options) => {
    sentBody = JSON.parse(options.body);
    return { ok: true, json: async () => ({ prompt_id: "xyz789" }) };
  };
  const qwen = new QwenImage({
    env: { QWEN_COMFY_URL: "http://127.0.0.1:8188" },
    fetchImpl: wrappedFetch,
  });
  await qwen.submit({
    prompt: "a studio portrait",
    checkpoint: "jib-mix-moss-realistic.safetensors",
    width: 1024,
    height: 1536,
    steps: 12,
    seed: 7,
    cfg: 1,
  });
  const graph = sentBody.prompt;
  const clipLoader = Object.values(graph).find((node) => node.class_type === "CLIPLoader");
  assert.equal(clipLoader.inputs.type, "moss_image");
  const latentNode = Object.values(graph).find((node) => node.class_type === "EmptyFlux2LatentImage");
  assert.ok(latentNode);
});

test("QwenImage.submit wires Krea-2 checkpoints to the official krea2 CLIP type", async () => {
  let sentBody;
  const wrappedFetch = async (url, options) => {
    sentBody = JSON.parse(options.body);
    return { ok: true, json: async () => ({ prompt_id: "krea456" }) };
  };
  const qwen = new QwenImage({
    env: { QWEN_COMFY_URL: "http://127.0.0.1:8188" },
    fetchImpl: wrappedFetch,
  });
  await qwen.submit({
    prompt: "a realistic portrait",
    checkpoint: KREA2_CHECKPOINT,
    width: 1024,
    height: 1536,
    steps: 12,
    seed: 9,
    cfg: 1.5,
  });
  const graph = sentBody.prompt;
  const clipLoader = Object.values(graph).find((node) => node.class_type === "CLIPLoader");
  assert.equal(clipLoader.inputs.type, "krea2");
  const latentNode = Object.values(graph).find((node) => node.class_type === "EmptyLatentImage");
  assert.ok(latentNode);
});

test("QwenImage.submit uses the checkpoint's recommended defaults when settings are omitted", async () => {
  let sentBody;
  const wrappedFetch = async (url, options) => {
    sentBody = JSON.parse(options.body);
    return { ok: true, json: async () => ({ prompt_id: "defaults" }) };
  };
  const qwen = new QwenImage({
    env: { QWEN_COMFY_URL: "http://127.0.0.1:8188" },
    fetchImpl: wrappedFetch,
  });
  await qwen.submit({
    prompt: "a tiny dragon",
    checkpoint: "qwen-image-2.1-step90000-FP8-comfy.safetensors",
  });
  const sampler = Object.values(sentBody.prompt).find((node) => node.class_type === "KSampler");
  assert.equal(sampler.inputs.steps, 35);
  assert.equal(sampler.inputs.cfg, 5);
  assert.equal(sampler.inputs.sampler_name, "er_sde");
  assert.equal(sampler.inputs.scheduler, "simple");
});

test("QwenImage.submit reports an unreachable ComfyUI as a 503", async () => {
  const qwen = new QwenImage({
    env: { QWEN_COMFY_URL: "http://127.0.0.1:8188" },
    fetchImpl: async () => { throw new TypeError("fetch failed"); },
  });
  await assert.rejects(
    qwen.submit({ prompt: "test", checkpoint: "qwen-image-2.1-step90000-FP8-comfy.safetensors" }),
    (error) => error.status === 503 && /not reachable/.test(error.message),
  );
});

test("QwenImage.submit surfaces ComfyUI HTTP errors", async () => {
  const wrappedFetch = async () => ({ ok: false, status: 404, text: async () => "node not found" });
  const qwen = new QwenImage({
    env: { QWEN_COMFY_URL: "http://127.0.0.1:8188" },
    fetchImpl: wrappedFetch,
  });
  await assert.rejects(
    qwen.submit({ prompt: "test", checkpoint: "qwen-image-21-uncensored-omni-quantized.safetensors" }),
    /ComfyUI rejected Qwen Image job \(HTTP 404\)/,
  );
});
