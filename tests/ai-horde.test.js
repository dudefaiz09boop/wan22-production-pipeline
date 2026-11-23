"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  AIHordeProvider,
  BASE_URL,
  ANONYMOUS_KEY,
  DEFAULT_MODEL,
  DEFAULT_SAMPLER,
} = require("../gateway/ai-horde");

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

test("AIHordeProvider falls back to the anonymous key and snaps dimensions", () => {
  const horde = new AIHordeProvider({
    env: { HORDE_WIDTH: "1000", HORDE_HEIGHT: "9999" },
    fetchImpl: () => assert.fail("must not fetch"),
  });
  assert.equal(horde.configured, true);
  assert.equal(horde.host, "aihorde.net");
  assert.equal(horde.endpoint, BASE_URL);
  assert.equal(horde.apiKey, ANONYMOUS_KEY);
  assert.equal(horde.anonymous, true);
  // 1000 -> 1024 (nearest multiple of 64), 9999 -> clamped to 3072.
  assert.equal(horde.defaultWidth, 1024);
  assert.equal(horde.defaultHeight, 3072);
  assert.equal(horde.defaultModel, DEFAULT_MODEL);
  assert.equal(horde.defaultSampler, DEFAULT_SAMPLER);
  assert.equal(horde.defaultNsfw, true);
});

test("AIHordeProvider keeps a real key and honours HORDE_NSFW=false", () => {
  const horde = new AIHordeProvider({
    env: { HORDE_API_KEY: "real-key", HORDE_NSFW: "false", HORDE_SAMPLER: "not-a-sampler" },
  });
  assert.equal(horde.apiKey, "real-key");
  assert.equal(horde.anonymous, false);
  assert.equal(horde.defaultNsfw, false);
  // Unknown samplers fall back instead of being forwarded to the API.
  assert.equal(horde.resolveSampler("not-a-sampler"), DEFAULT_SAMPLER);
  assert.equal(horde.resolveSampler("k_dpmpp_2m"), "k_dpmpp_2m");
});

test("AIHordeProvider submits an uncensored, unshared job with inline images", async () => {
  const horde = new AIHordeProvider({
    env: {},
    fetchImpl: async (url, options) => {
      assert.equal(url, `${BASE_URL}/generate/async`);
      assert.equal(options.method, "POST");
      assert.equal(options.headers["client-agent"].startsWith("local-ai-studio"), true);
      const body = JSON.parse(options.body);
      assert.equal(body.prompt, "a lighthouse at dusk");
      assert.equal(body.nsfw, true);
      assert.equal(body.censor_nsfw, false);
      assert.equal(body.shared, false);
      assert.equal(body.trusted_workers, false);
      assert.equal(body.replacement_filter, false);
      // r2 must stay false so poll() receives base64 in generation.img.
      assert.equal(body.r2, false);
      assert.deepEqual(body.models, [DEFAULT_MODEL]);
      assert.equal(body.params.width, 1024);
      assert.equal(body.params.height, 1024);
      assert.equal(body.params.sampler_name, DEFAULT_SAMPLER);
      assert.equal(body.params.n, 1);
      return jsonResponse({ id: "horde-job-1", kudos: 12 });
    },
  });
  const job = await horde.submit({ prompt: "a lighthouse at dusk" });
  assert.equal(job.horde_id, "horde-job-1");
  assert.equal(job.status, "queued");
  assert.equal(job.kudos, 12);
});

test("AIHordeProvider rejects an empty prompt without calling the API", async () => {
  const horde = new AIHordeProvider({ env: {}, fetchImpl: () => assert.fail("must not fetch") });
  await assert.rejects(horde.submit({ prompt: "   " }), (error) => {
    assert.equal(error.status, 400);
    return true;
  });
});

test("AIHordeProvider poll walks queued -> running -> completed and persists the image", async () => {
  const calls = [];
  let phase = 0;
  const horde = new AIHordeProvider({
    env: {},
    fetchImpl: async (url) => {
      calls.push(url);
      if (url.endsWith("/generate/async")) return jsonResponse({ id: "horde-job-2" });
      if (url.includes("/generate/check/")) {
        phase += 1;
        if (phase === 1) return jsonResponse({ done: false, queue_position: 3, wait_time: 40 });
        return jsonResponse({ done: true });
      }
      if (url.includes("/generate/status/")) {
        return jsonResponse({
          kudos: 5,
          generations: [{ img: Buffer.from("webp-bytes").toString("base64"), seed: "42", model: "stable_diffusion", worker_name: "w1", censored: false }],
        });
      }
      throw new Error(`unexpected url ${url}`);
    },
  });

  const job = await horde.submit({ prompt: "a lighthouse at dusk" });

  const running = await horde.poll(job.id);
  assert.equal(running.status, "running");
  assert.equal(running.queue_position, 3);

  const completed = await horde.poll(job.id);
  assert.equal(completed.status, "completed");
  assert.equal(completed.outputs.images.length, 1);
  assert.equal(completed.outputs.images[0].seed, "42");
  assert.equal(completed.outputs.images[0].censored, false);
  assert.equal(completed.outputs.images[0].url, `/v1/horde/tasks/${job.id}/image/0`);

  // A finished job is returned as-is instead of re-polling the API.
  const before = calls.length;
  const again = await horde.poll(job.id);
  assert.equal(again.status, "completed");
  assert.equal(calls.length, before);
});

test("AIHordeProvider poll fails the job when the horde reports a fault", async () => {
  const horde = new AIHordeProvider({
    env: {},
    fetchImpl: async (url) => {
      if (url.endsWith("/generate/async")) return jsonResponse({ id: "horde-job-3" });
      return jsonResponse({ faulted: true });
    },
  });
  const job = await horde.submit({ prompt: "a lighthouse at dusk" });
  const result = await horde.poll(job.id);
  assert.equal(result.status, "failed");
  assert.match(result.error, /faulted/);
  assert.equal((await horde.getJob(job.id)).status, "failed");
});
