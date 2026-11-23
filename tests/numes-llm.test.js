"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  NumesLlm,
  CHAT_URL,
  FREE_MODELS,
  FREE_MODEL_IDS,
  DEFAULT_MODEL_ID,
  LAST_RESORT_MODEL_ID,
} = require("../gateway/numes-llm");

const UNCENSORED_IDS = FREE_MODELS.filter((model) => model.uncensored).map((model) => model.id);

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

test("NumesLlm enables every free model and defaults to the uncensored one", () => {
  const llm = new NumesLlm({ env: { NUMES_API_KEY: "test-token" } });
  assert.equal(llm.available, true);
  assert.equal(llm.host, "api.numes.ai");
  assert.equal(llm.endpoint, "https://api.numes.ai/api/v1");
  assert.equal(FREE_MODEL_IDS.length, 9);
  assert.deepEqual(llm.listModels().map((model) => model.id), FREE_MODEL_IDS);
  assert.equal(llm.resolveModel(undefined), DEFAULT_MODEL_ID);
  assert.equal(llm.resolveModel("z-ai/glm-5.3:free"), "z-ai/glm-5.3:free");
});

test("NumesLlm refuses paid models so a free request can never bill", () => {
  const llm = new NumesLlm({ env: { NUMES_API_KEY: "test-token" } });
  for (const paid of ["z-ai/glm-5.3", "openai/gpt-4o", "dealignai/glm-5.3-uncensored-fp8"]) {
    assert.throws(() => llm.resolveModel(paid), (error) => {
      assert.equal(error.status, 400);
      assert.match(error.message, /Only the free Numes models/);
      return true;
    });
  }
});

test("NumesLlm sends an OpenAI-compatible request with privacy and JSON mode", async () => {
  const llm = new NumesLlm({
    env: { NUMES_API_KEY: "test-token" },
    fetchImpl: async (url, options) => {
      assert.equal(url, CHAT_URL);
      assert.equal(options.method, "POST");
      assert.equal(options.headers.authorization, "Bearer test-token");
      const body = JSON.parse(options.body);
      assert.equal(body.model, DEFAULT_MODEL_ID);
      assert.equal(body.stream, false);
      assert.equal(body.max_tokens, 64);
      assert.equal(body.temperature, 0.3);
      assert.deepEqual(body.response_format, { type: "json_object" });
      assert.deepEqual(body.provider, { data_collection: "deny", zdr: true });
      assert.deepEqual(body.messages, [{ role: "user", content: "hello" }]);
      return jsonResponse({ choices: [{ message: { content: "ok" } }] });
    },
  });
  const response = await llm.chat({
    messages: [{ role: "user", content: "hello" }],
    maxTokens: 64,
    temperature: 0.3,
    responseFormat: { type: "json_object" },
  });
  assert.equal(response.ok, true);
});

test("NumesLlm drops the privacy object only when data collection is allowed", async () => {
  const llm = new NumesLlm({
    env: { NUMES_API_KEY: "test-token", NUMES_ALLOW_DATA_COLLECTION: "true" },
    fetchImpl: async (_url, options) => {
      assert.equal(JSON.parse(options.body).provider, undefined);
      return jsonResponse({ choices: [] });
    },
  });
  await llm.chat({ messages: [{ role: "user", content: "hello" }] });
});

test("NumesLlm reports missing auth and rejects empty messages", async () => {
  const llm = new NumesLlm({ env: {}, fetchImpl: () => assert.fail("must not fetch") });
  assert.equal(llm.available, false);
  await assert.rejects(llm.chat({ messages: [{ role: "user", content: "hello" }] }), (error) => {
    assert.equal(error.status, 503);
    assert.match(error.message, /NUMES_API_KEY/);
    return true;
  });

  const configured = new NumesLlm({ env: { NUMES_API_KEY: "test-token" } });
  await assert.rejects(configured.chat({ messages: [] }), (error) => {
    assert.equal(error.status, 400);
    return true;
  });
});

test("NumesLlm failover chain starts with the request and ends with Nemotron", () => {
  const llm = new NumesLlm({ env: { NUMES_API_KEY: "test-token" } });
  const chain = llm.failoverChain("z-ai/glm-5.3:free");
  assert.equal(chain[0], "z-ai/glm-5.3:free");
  assert.equal(chain[chain.length - 1], LAST_RESORT_MODEL_ID);
  assert.equal(new Set(chain).size, chain.length);
  assert.equal(chain.length, FREE_MODEL_IDS.length);
  // After the requested model, the uncensored models come before the censored ones.
  const rest = chain.slice(1);
  const firstCensored = rest.findIndex((id) => !UNCENSORED_IDS.includes(id));
  const lastUncensored = rest.reduce(
    (last, id, index) => (UNCENSORED_IDS.includes(id) ? index : last),
    -1,
  );
  assert.ok(lastUncensored < firstCensored);
});

test("NumesLlm retries the next model when one is unavailable", async () => {
  const attempts = [];
  const llm = new NumesLlm({
    env: { NUMES_API_KEY: "test-token" },
    fetchImpl: async (_url, options) => {
      const { model } = JSON.parse(options.body);
      attempts.push(model);
      if (attempts.length === 1) return jsonResponse({ error: "down" }, 503);
      return jsonResponse({ choices: [{ message: { content: "ok" } }] });
    },
  });
  const response = await llm.chat({ messages: [{ role: "user", content: "hello" }] });
  assert.equal(response.ok, true);
  assert.equal(attempts.length, 2);
  assert.equal(response.numesModel, attempts[1]);
  assert.equal(response.numesAttempts, 2);
});

test("NumesLlm retries after a transient network error", async () => {
  let calls = 0;
  const llm = new NumesLlm({
    env: { NUMES_API_KEY: "test-token" },
    fetchImpl: async () => {
      calls += 1;
      if (calls === 1) {
        const error = new Error("fetch failed");
        error.cause = { code: "ECONNRESET" };
        throw error;
      }
      return jsonResponse({ choices: [{ message: { content: "ok" } }] });
    },
  });
  const response = await llm.chat({ messages: [{ role: "user", content: "hello" }] });
  assert.equal(response.ok, true);
  assert.equal(calls, 2);
});

test("NumesLlm retries fetch failures without a platform-specific error code", async () => {
  let calls = 0;
  const llm = new NumesLlm({
    env: { NUMES_API_KEY: "test-token" },
    fetchImpl: async () => {
      calls += 1;
      if (calls === 1) throw new TypeError("fetch failed");
      return jsonResponse({ choices: [{ message: { content: "ok" } }] });
    },
  });
  const response = await llm.chat({ messages: [{ role: "user", content: "hello" }] });
  assert.equal(response.ok, true);
  assert.equal(calls, 2);
});

test("NumesLlm does not fail over on a terminal status", async () => {
  let calls = 0;
  const llm = new NumesLlm({
    env: { NUMES_API_KEY: "test-token" },
    fetchImpl: async () => {
      calls += 1;
      return jsonResponse({ error: "bad request" }, 400);
    },
  });
  const response = await llm.chat({ messages: [{ role: "user", content: "hello" }] });
  assert.equal(response.status, 400);
  assert.equal(calls, 1);
});

test("NumesLlm falls back to Nemotron when every other model fails", async () => {
  const attempts = [];
  const llm = new NumesLlm({
    env: { NUMES_API_KEY: "test-token" },
    fetchImpl: async (_url, options) => {
      const { model } = JSON.parse(options.body);
      attempts.push(model);
      if (model === LAST_RESORT_MODEL_ID) {
        return jsonResponse({ choices: [{ message: { content: "ok" } }] });
      }
      return jsonResponse({ error: "down" }, 503);
    },
  });
  const response = await llm.chat({ messages: [{ role: "user", content: "hello" }] });
  assert.equal(response.ok, true);
  assert.equal(response.numesModel, LAST_RESORT_MODEL_ID);
  assert.equal(attempts[attempts.length - 1], LAST_RESORT_MODEL_ID);
  assert.equal(attempts.length, FREE_MODEL_IDS.length);
});

test("NumesLlm reports every failure when no model answers", async () => {
  const llm = new NumesLlm({
    env: { NUMES_API_KEY: "test-token" },
    fetchImpl: async () => jsonResponse({ error: "down" }, 503),
  });
  await assert.rejects(llm.chat({ messages: [{ role: "user", content: "hello" }] }), (error) => {
    assert.equal(error.status, 502);
    assert.match(error.message, /Every free Numes model failed/);
    assert.match(error.message, /HTTP 503/);
    return true;
  });
});

test("NumesLlm honours NUMES_MODEL_FAILOVER=false and NUMES_MAX_ATTEMPTS", async () => {
  let calls = 0;
  const single = new NumesLlm({
    env: { NUMES_API_KEY: "test-token", NUMES_MODEL_FAILOVER: "false" },
    fetchImpl: async () => {
      calls += 1;
      return jsonResponse({ error: "down" }, 503);
    },
  });
  assert.deepEqual(single.failoverChain(undefined), [DEFAULT_MODEL_ID]);
  await assert.rejects(single.chat({ messages: [{ role: "user", content: "hello" }] }));
  assert.equal(calls, 1);

  const limited = new NumesLlm({
    env: { NUMES_API_KEY: "test-token", NUMES_MAX_ATTEMPTS: "2" },
    fetchImpl: async () => jsonResponse({ error: "down" }, 503),
  });
  const chain = limited.failoverChain(undefined);
  assert.equal(chain.length, 2);
  assert.equal(chain[0], DEFAULT_MODEL_ID);
  assert.equal(chain[1], LAST_RESORT_MODEL_ID);

  const oneAttempt = new NumesLlm({
    env: { NUMES_API_KEY: "test-token", NUMES_MAX_ATTEMPTS: "1" },
  });
  assert.deepEqual(oneAttempt.failoverChain("z-ai/glm-5.3:free"), ["z-ai/glm-5.3:free"]);
});
