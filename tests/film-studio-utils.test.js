const assert = require("node:assert/strict");
const test = require("node:test");
const {
  detectRomanizedLanguage,
  detectVoiceProfile,
  defaultKeyframeMode,
  dialogueCharacters,
  ensureWorkerReady,
  filenameShotNumber,
  hasKeyframeForShot,
  hasNarrationText,
  isTimecodedPrompt,
  hasYamlStartFrame,
  mapStillFilesToShots,
  parseDialogueTurns,
  requiresImageKeyframe,
  supportsPerShotLoras,
} = require("../gateway/public/film-studio-utils");

function shot(number, keyframeAssetId = null) {
  return { number, metadata: { globalNumber: number }, keyframeAssetId };
}

test("bulk still mapping uses beat tags and fills only unattached beats alphabetically", () => {
  const shots = [shot(1), shot(2, "existing"), shot(3)];
  const result = mapStillFilesToShots([
    { name: "unmatched-b.png" },
    { name: "c02.png" },
    { name: "unmatched-a.png" },
  ], shots);

  assert.deepEqual(result.map(({ shot: target, file }) => [target.number, file.name]), [
    [1, "unmatched-a.png"],
    [2, "c02.png"],
    [3, "unmatched-b.png"],
  ]);
  assert.equal(filenameShotNumber("story-beat-004-final.webp"), 4);
  assert.equal(filenameShotNumber("shot_12.jpg"), 12);
  assert.equal(filenameShotNumber("cover.png"), null);
});

test("bulk still mapping rejects unknown and duplicate beat tags before upload", () => {
  const shots = [shot(1), shot(2)];
  assert.throws(
    () => mapStillFilesToShots([{ name: "c03.png" }], shots),
    /names beat 3/,
  );
  assert.throws(
    () => mapStillFilesToShots([{ name: "c01.png" }, { name: "beat-01.png" }], shots),
    /More than one image names beat 1/,
  );
});

test("worker readiness uses compute-manager reuse and polls until the session is ready", async () => {
  const calls = [];
  const statuses = [
    { id: "worker-1", kind: "video", model: "wan-test", status: "starting" },
    { id: "worker-1", kind: "video", model: "wan-test", status: "idle" },
  ];
  const updates = [];
  const worker = await ensureWorkerReady({
    request: async (url, options) => {
      calls.push({ url, options });
      return statuses.shift();
    },
    kind: "video",
    model: "wan-test",
    pollIntervalMs: 0,
    timeoutMs: 1000,
    onStatus: (status) => updates.push(status.status),
  });

  assert.equal(worker.status, "idle");
  assert.deepEqual(updates, ["starting", "idle"]);
  assert.equal(calls[0].url, "/v1/compute/workers");
  assert.deepEqual(JSON.parse(calls[0].options.body), { kind: "video", model: "wan-test" });
  assert.equal(calls[1].url, "/v1/workers/worker-1");
});

test("worker readiness surfaces failed sessions and leaves a timed-out worker running", async () => {
  await assert.rejects(
    ensureWorkerReady({
      request: async () => ({ id: "worker-failed", status: "failed", error: "kernel failed" }),
      kind: "image",
      model: "image-test",
      timeoutMs: 1000,
    }),
    /kernel failed/,
  );
  await assert.rejects(
    ensureWorkerReady({
      request: async () => ({ id: "worker-starting", status: "starting" }),
      kind: "image",
      model: "image-test",
      timeoutMs: 0,
    }),
    /worker was left running/,
  );
});

test("worker readiness decodes HTTP responses and surfaces API errors", async () => {
  await assert.rejects(
    ensureWorkerReady({
      request: async () => new Response(JSON.stringify({ error: "model-store tunnel is unavailable" }), {
        status: 503,
        headers: { "content-type": "application/json" },
      }),
      kind: "video",
      model: "wan-test",
    }),
    /model-store tunnel is unavailable/,
  );
});

test("voice detection selects supported Hindi, Urdu, and English voices", () => {
  const voices = [
    { id: "en-US-AriaNeural", language: "en-US" },
    { id: "en-US-GuyNeural", language: "en-US" },
    { id: "en-IN-NeerjaNeural", language: "en-IN" },
    { id: "en-IN-PrabhatNeural", language: "en-IN" },
    { id: "hi-IN-MadhurNeural", language: "hi-IN" },
    { id: "hi-IN-SwaraNeural", language: "hi-IN" },
    { id: "ur-IN-GulNeural", language: "ur-IN" },
    { id: "ur-IN-SalmanNeural", language: "ur-IN" },
    { id: "ur-PK-UzmaNeural", language: "ur-PK" },
  ];

  assert.deepEqual(detectVoiceProfile("नमस्ते दुनिया", voices), {
    voiceId: "hi-IN-SwaraNeural",
    language: "hi",
  });
  assert.deepEqual(detectVoiceProfile("السلام علیکم", voices), {
    voiceId: "ur-IN-GulNeural",
    language: "ur",
  });
  assert.deepEqual(detectVoiceProfile("A quiet evening.", voices, "en-US-GuyNeural"), {
    voiceId: "en-US-GuyNeural",
    language: "en",
  });
  assert.deepEqual(detectVoiceProfile("A quiet evening.", voices, "ur-IN-SalmanNeural"), {
    voiceId: "en-IN-PrabhatNeural",
    language: "en",
  });
  assert.deepEqual(detectVoiceProfile("नमस्ते दुनिया", voices, "ur-IN-SalmanNeural"), {
    voiceId: "hi-IN-MadhurNeural",
    language: "hi",
  });
  assert.deepEqual(detectVoiceProfile("السلام علیکم", voices, "en-IN-NeerjaNeural"), {
    voiceId: "ur-IN-GulNeural",
    language: "ur",
  });
});

test("romanized Hindi and Urdu select matching voices without treating English as Indic", () => {
  const voices = [
    { id: "en-US-AriaNeural", language: "en-US" },
    { id: "en-IN-NeerjaNeural", language: "en-IN" },
    { id: "en-IN-PrabhatNeural", language: "en-IN" },
    { id: "hi-IN-MadhurNeural", language: "hi-IN" },
    { id: "hi-IN-SwaraNeural", language: "hi-IN" },
    { id: "ur-IN-GulNeural", language: "ur-IN" },
    { id: "ur-IN-SalmanNeural", language: "ur-IN" },
    { id: "ur-PK-UzmaNeural", language: "ur-PK" },
  ];

  assert.equal(detectRomanizedLanguage("Ammi ka phone aaya hai, utha leti hu. Haan Ammi."), "hi");
  assert.equal(detectVoiceProfile("Ammi ka phone aaya hai, utha leti hu. Haan Ammi.", voices).voiceId, "hi-IN-SwaraNeural");
  assert.equal(detectVoiceProfile("Haan Ammi.", voices, "hi-IN-SwaraNeural").voiceId, "hi-IN-SwaraNeural");
  assert.equal(detectVoiceProfile("Chai bana de.", voices, "ur-IN-GulNeural").voiceId, "ur-IN-GulNeural");
  assert.equal(detectVoiceProfile("An English line.", voices, "ur-IN-SalmanNeural").voiceId, "en-IN-PrabhatNeural");
  assert.equal(detectRomanizedLanguage("Aap kahan hain?", "ur-PK-UzmaNeural"), "ur");
  assert.equal(detectVoiceProfile("Aap kahan hain?", voices).voiceId, "ur-IN-GulNeural");
  assert.equal(detectVoiceProfile("Aap kahan hain?", voices, "ur-PK-UzmaNeural").voiceId, "ur-PK-UzmaNeural");
  assert.equal(detectRomanizedLanguage("A quiet evening by the sea."), null);
});

test("render and narration eligibility include YAML frames and ignore blank text", () => {
  const imported = {
    metadata: { sourceBeat: { id: "c01", startFrame: "startframes/c01.png" } },
  };
  assert.equal(hasYamlStartFrame(imported), true);
  assert.equal(hasKeyframeForShot(imported), true);
  assert.equal(hasYamlStartFrame({ metadata: { sourceBeat: { id: "c01", startFrame: "startframes/c02.png" } } }), false);
  assert.equal(hasKeyframeForShot(shot(1, "uploaded-still")), true);
  assert.equal(hasNarrationText("  namaste  "), true);
  assert.equal(hasNarrationText(" \n "), false);
  assert.equal(requiresImageKeyframe("wan22-fm-lightning-q8"), true);
  assert.equal(requiresImageKeyframe("wan22-fm-lightning-q8-dual-gpu"), true);
  assert.equal(supportsPerShotLoras("wan22-fm-lightning-q8-dual-gpu"), true);
  assert.equal(supportsPerShotLoras("wan-vace-1.3b"), false);
  assert.equal(requiresImageKeyframe("wan-t2v-1.3b"), false);
});

test("user-provided c08 and c18 character references default to direct I2V conditioning", () => {
  assert.equal(defaultKeyframeMode({ metadata: { sourceBeat: { id: "c08" } } }), "source");
  assert.equal(defaultKeyframeMode({ metadata: { sourceBeat: { id: "c18" } } }), "source");
  assert.equal(defaultKeyframeMode({ metadata: { sourceBeat: { id: "c09" } } }), "continuity");
  assert.equal(defaultKeyframeMode({
    metadata: { renderKeyframeMode: "continuity", sourceBeat: { id: "c08" } },
  }), "continuity");
});

test("recognizes six ordered timecoded prompt lines without source-beat metadata", () => {
  const prompt = Array.from({ length: 6 }, (_, second) =>
    `(At ${second} seconds: static, subject, action, lighting)`).join("\n");

  assert.equal(isTimecodedPrompt(prompt), true);
  assert.equal(isTimecodedPrompt(prompt.replace("(At 3 seconds:", "(At 4 seconds:")), false);
  assert.equal(isTimecodedPrompt(prompt.split("\n").slice(1).join("\n")), false);
  assert.equal(isTimecodedPrompt("A normal visual prompt."), false);
});

test("dialogue turns route tagged speakers to their character voices", () => {
  const characters = [
    { id: "faiz-id", name: "Faiz", voiceProfile: "en-US-GuyNeural" },
    { id: "juveriya-id", name: "Juveriya", voiceProfile: "hi-IN-SwaraNeural" },
    { id: "ammi-id", name: "Ammi", voiceProfile: "ur-IN-GulNeural" },
  ];
  const narration = "Faiz: Are you ready?\nJuveriya: Give me a moment.\nAmmi: Call me when you get home.";
  const speakers = dialogueCharacters(narration, characters);
  const turns = parseDialogueTurns(
    narration,
    speakers,
    "en-US-AriaNeural",
  );
  assert.deepEqual(turns, [
    { speakerId: "faiz-id", speakerName: "Faiz", text: "Are you ready?", voiceProfile: "en-US-GuyNeural" },
    { speakerId: "juveriya-id", speakerName: "Juveriya", text: "Give me a moment.", voiceProfile: "hi-IN-SwaraNeural" },
    { speakerId: "ammi-id", speakerName: "Ammi", text: "Call me when you get home.", voiceProfile: "ur-IN-GulNeural" },
  ]);
  assert.deepEqual(speakers.map((character) => character.id), ["faiz-id", "juveriya-id", "ammi-id"]);
  assert.throws(() => parseDialogueTurns("Line\n".repeat(9)), /at most 8/);
});
