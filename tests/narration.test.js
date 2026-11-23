"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { EDGE_TTS_VOICES, NarrationService } = require("../gateway/narration");

test("Edge TTS is the only narration engine and includes English, Hindi, and Urdu voices", () => {
  const voices = new NarrationService({ env: {} }).voices();
  assert.ok(voices.length > 0);
  assert.ok(voices.every((voice) => voice.engine === "edge-tts"));
  assert.ok(voices.some((voice) => voice.id === "en-US-AriaNeural"));
  assert.ok(voices.some((voice) => voice.id === "en-IN-NeerjaNeural"));
  assert.ok(voices.some((voice) => voice.id === "en-IN-PrabhatNeural"));
  assert.ok(voices.some((voice) => voice.id === "hi-IN-SwaraNeural"));
  assert.ok(voices.some((voice) => voice.id === "ur-PK-UzmaNeural"));
  assert.equal(EDGE_TTS_VOICES.has("en-US-AriaNeural"), true);
  assert.equal(EDGE_TTS_VOICES.has("en-IN-PrabhatNeural"), true);
});

test("Edge TTS uses the project uv environment unless explicitly overridden", () => {
  const service = new NarrationService({ env: {} });
  const projectPython = path.join(
    __dirname,
    "..",
    ".venv",
    process.platform === "win32" ? "Scripts" : "bin",
    process.platform === "win32" ? "python.exe" : "python",
  );
  assert.equal(service.python, fs.existsSync(projectPython) ? projectPython : "python");
  assert.equal(new NarrationService({ env: { EDGE_TTS_PYTHON: "custom-python" } }).python, "custom-python");
});

test("Edge TTS synthesizes with the selected voice and reports measured duration", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "edge-tts-test-"));
  const calls = [];
  const service = new NarrationService({
    env: { EDGE_TTS_PYTHON: "python-test", FFPROBE_PATH: "ffprobe-test" },
    execFileImpl: async (executable, args) => {
      calls.push({ executable, args });
      if (executable === "python-test") {
        fs.writeFileSync(args[4], Buffer.from("synthetic-mp3"));
        return { stdout: "", stderr: "" };
      }
      return { stdout: "2.75\n", stderr: "" };
    },
  });

  try {
    const audio = await service.synthesize({ text: "  A synthetic voice test.  ", voice: "en-US-AriaNeural" });
    assert.equal(audio.durationSeconds, 2.75);
    assert.equal(audio.mime, "audio/mpeg");
    assert.equal(audio.bytes, Buffer.byteLength("synthetic-mp3"));
    assert.equal(calls[0].args[0], "-c");
    assert.match(calls[0].args[1], /edge_tts\.Communicate/);
    assert.deepEqual(calls[0].args.slice(2, 4), ["A synthetic voice test.", "en-US-AriaNeural"]);
    assert.equal(fs.existsSync(audio.filePath), true);
    audio.cleanup();
    assert.equal(fs.existsSync(path.dirname(audio.filePath)), false);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("Edge TTS rejects unsupported voices and surfaces online-service failures", async () => {
  const service = new NarrationService({
    env: { EDGE_TTS_PYTHON: "python-test" },
    execFileImpl: async () => { throw new Error("service unavailable"); },
  });
  await assert.rejects(
    service.synthesize({ text: "Hello.", voice: "unknown-voice" }),
    /Choose a supported Edge TTS voice/,
  );
  await assert.rejects(
    service.synthesize({ text: "Hello.", voice: "en-US-AriaNeural" }),
    /Edge TTS failed.*service unavailable/,
  );
});
