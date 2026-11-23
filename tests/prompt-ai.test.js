"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { PromptAssistant, extractJson, MODEL_ID } = require("../gateway/prompt-ai");

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function fakeHuggingFace({ available = true, content = '{"prompt":"better prompt","negativePrompt":"blurry"}' } = {}) {
  return {
    available,
    host: "router.huggingface.co",
    calls: [],
    listModels() {
      return [{ id: MODEL_ID, label: "Llama-3-8B-Lexi · Featherless AI" }];
    },
    resolveModel(model) {
      if (!model || model === MODEL_ID) return MODEL_ID;
      const error = new Error("Only Llama-3-8B-Lexi is enabled.");
      error.status = 400;
      throw error;
    },
    async chat(request) {
      this.calls.push(request);
      if (content instanceof Error) throw content;
      if (typeof content === "number") return new Response("denied", { status: content });
      return jsonResponse({ choices: [{ message: { content } }] });
    },
  };
}

test("extractJson accepts fenced and noisy JSON", () => {
  assert.deepEqual(extractJson('```json\n{"prompt":"a","negativePrompt":"b"}\n```'), {
    prompt: "a",
    negativePrompt: "b",
  });
  assert.deepEqual(extractJson('Sure: {"prompt":"a","negativePrompt":"b"} thanks'), {
    prompt: "a",
    negativePrompt: "b",
  });
  assert.equal(extractJson("no json here"), null);
});

test("PromptAssistant uses only the configured Hugging Face model", async () => {
  const huggingFace = fakeHuggingFace();
  const assistant = new PromptAssistant({ huggingFace });
  const status = await assistant.describe();

  assert.equal(status.provider, "huggingface");
  assert.equal(status.available, true);
  assert.equal(status.local, false);
  assert.equal(status.model, MODEL_ID);
  assert.equal(status.providers.huggingface.available, true);
  assert.equal("ollama" in status.providers, false);
  assert.equal("remote" in status.providers, false);

  const result = await assistant.assist({
    kind: "image",
    prompt: "a castle",
    provider: "huggingface",
  });
  assert.equal(result.provider, "huggingface");
  assert.equal(result.model, MODEL_ID);
  assert.equal(result.prompt, "better prompt");
  assert.equal(huggingFace.calls.length, 1);
  assert.equal(huggingFace.calls[0].model, MODEL_ID);
  assert.equal(huggingFace.calls[0].maxTokens, 500);
});

test("PromptAssistant reports the Numes model that answered after failover", async () => {
  const requestedModel = "dealignai/glm-5.3-flash-uncensored-fp8:free";
  const servedModel = "nvidia/nemotron-3-ultra-550b-a55b:free";
  const numes = {
    available: true,
    host: "api.numes.ai",
    listModels: () => [{ id: requestedModel, label: "GLM Flash" }],
    resolveModel: (model) => model || requestedModel,
    chat: async () => {
      const response = jsonResponse({
        choices: [{ message: { content: '{"prompt":"better prompt","negativePrompt":"blurry"}' } }],
      });
      response.numesModel = servedModel;
      return response;
    },
  };
  const result = await new PromptAssistant({ numes }).assist({
    prompt: "a castle",
    provider: "numes",
    model: requestedModel,
  });
  assert.equal(result.model, servedModel);
});

test("PromptAssistant preserves the negative prompt when the model omits one", async () => {
  const assistant = new PromptAssistant({
    huggingFace: fakeHuggingFace({ content: '{"prompt":"better prompt"}' }),
  });
  const result = await assistant.assist({
    kind: "video",
    prompt: "a wave",
    negativePrompt: "flicker, cuts",
  });
  assert.equal(result.negativePrompt, "flicker, cuts");
});

test("PromptAssistant rejects non-Hugging-Face providers and models", async () => {
  const assistant = new PromptAssistant({ huggingFace: fakeHuggingFace() });
  await assert.rejects(assistant.assist({ prompt: "a fox", provider: "ollama" }), (error) => {
    assert.equal(error.status, 400);
    return true;
  });
  await assert.rejects(assistant.assist({ prompt: "a fox", model: "another/model" }), (error) => {
    assert.equal(error.status, 400);
    return true;
  });
});

test("PromptAssistant validates input and reports a missing HF token", async () => {
  const assistant = new PromptAssistant({ huggingFace: fakeHuggingFace({ available: false }) });
  assert.equal((await assistant.describe()).available, false);
  await assert.rejects(assistant.assist({ prompt: "a fox" }), (error) => {
    assert.equal(error.status, 503);
    assert.match(error.message, /HF_TOKEN/);
    return true;
  });
  await assert.rejects(assistant.assist({ prompt: "  " }), (error) => error.status === 503);
  await assert.rejects(
    new PromptAssistant({ huggingFace: fakeHuggingFace() }).assist({ prompt: "a fox", kind: "audio" }),
    (error) => error.status === 400 && /kind must be image or video/.test(error.message),
  );
});

test("PromptAssistant maps provider auth failures and timeouts", async () => {
  const authAssistant = new PromptAssistant({ huggingFace: fakeHuggingFace({ content: 401 }) });
  await assert.rejects(authAssistant.assist({ prompt: "a fox" }), (error) => {
    assert.equal(error.status, 401);
    assert.match(error.message, /HF_TOKEN/);
    return true;
  });

  const timeoutError = new Error("aborted");
  timeoutError.name = "TimeoutError";
  const timeoutAssistant = new PromptAssistant({
    env: { PROMPT_AI_TIMEOUT_SECONDS: "15" },
    huggingFace: fakeHuggingFace({ content: timeoutError }),
  });
  await assert.rejects(timeoutAssistant.assist({ prompt: "a fox" }), (error) => {
    assert.equal(error.status, 504);
    assert.match(error.message, /timed out after 15s/);
    return true;
  });
});

test("PromptAssistant rejects unparseable model output", async () => {
  const assistant = new PromptAssistant({ huggingFace: fakeHuggingFace({ content: "not json at all" }) });
  await assert.rejects(assistant.assist({ prompt: "a fox" }), (error) => {
    assert.equal(error.status, 502);
    assert.match(error.message, /unexpected format/);
    return true;
  });
});
