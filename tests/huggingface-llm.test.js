"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  HuggingFaceLlm,
  MODEL_ID,
  INFERENCE_URL,
} = require("../gateway/huggingface-llm");

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

test("HuggingFaceLlm exposes only the configured Llama-3-8B-Lexi model", () => {
  const llm = new HuggingFaceLlm({ env: { HF_TOKEN: "test-token" } });
  assert.equal(llm.available, true);
  assert.equal(llm.host, "router.huggingface.co");
  assert.deepEqual(llm.listModels().map((model) => model.id), [MODEL_ID]);
  assert.equal(llm.resolveModel(undefined), MODEL_ID);
  assert.equal(llm.resolveModel(MODEL_ID), MODEL_ID);
  assert.throws(() => llm.resolveModel("another/model"), (error) => error.status === 400);
});

test("HuggingFaceLlm sends an OpenAI-compatible request with the local HF token", async () => {
  const llm = new HuggingFaceLlm({
    env: { HF_TOKEN: "test-token" },
    fetchImpl: async (url, options) => {
      assert.equal(url, INFERENCE_URL);
      assert.equal(options.method, "POST");
      assert.equal(options.headers.authorization, "Bearer test-token");
      assert.equal(options.headers["content-type"], "application/json");
      assert.equal(options.headers.accept, "application/json");
      const body = JSON.parse(options.body);
      assert.equal(body.model, `${MODEL_ID}:featherless-ai`);
      assert.equal(body.stream, true);
      assert.equal(body.max_tokens, 64);
      assert.equal(body.temperature, 0.3);
      assert.deepEqual(body.messages, [{ role: "user", content: "hello" }]);
      return jsonResponse({ choices: [{ message: { content: "ok" } }] });
    },
  });
  const response = await llm.chat({
    model: MODEL_ID,
    messages: [{ role: "user", content: "hello" }],
    maxTokens: 64,
    temperature: 0.3,
    stream: true,
  });
  assert.equal(response.ok, true);
});

test("HuggingFaceLlm reports missing auth and rejects empty messages", async () => {
  const llm = new HuggingFaceLlm({ env: {}, fetchImpl: () => assert.fail("must not fetch") });
  assert.equal(llm.available, false);
  await assert.rejects(llm.chat({ messages: [{ role: "user", content: "hello" }] }), (error) => {
    assert.equal(error.status, 503);
    assert.match(error.message, /HF_TOKEN/);
    return true;
  });

  const configured = new HuggingFaceLlm({ env: { HF_TOKEN: "test-token" } });
  await assert.rejects(configured.chat({ messages: [] }), (error) => {
    assert.equal(error.status, 400);
    return true;
  });
});
