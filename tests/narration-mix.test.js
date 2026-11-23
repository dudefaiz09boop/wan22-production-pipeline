"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { concatenateNarrationFiles } = require("../gateway/narration-mix");

test("narration mixer preserves turn order and inserts a short pause", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "narration-mix-test-"));
  const inputs = [path.join(directory, "first.mp3"), path.join(directory, "second.wav")];
  const output = path.join(directory, "dialogue.wav");
  inputs.forEach((input, index) => fs.writeFileSync(input, `audio-${index}`));
  let commandArgs;
  try {
    const result = await concatenateNarrationFiles(inputs, output, {
      execFileImpl: async (_command, args) => {
        commandArgs = args;
        fs.writeFileSync(args[args.length - 1], "mixed-audio");
      },
    });

    assert.deepEqual(result, { filePath: output, bytes: Buffer.byteLength("mixed-audio") });
    assert.ok(commandArgs.indexOf(inputs[0]) < commandArgs.indexOf(inputs[1]));
    assert.match(commandArgs[commandArgs.indexOf("-filter_complex") + 1], /anullsrc=.*d=0\.25/);
    assert.match(commandArgs[commandArgs.indexOf("-filter_complex") + 1], /concat=n=3:v=0:a=1/);
    assert.equal(commandArgs[commandArgs.indexOf("-c:a") + 1], "pcm_s16le");
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("narration mixer rejects missing inputs and invalid turn counts", async () => {
  await assert.rejects(concatenateNarrationFiles(["missing.wav", "missing-too.wav"], "out.wav"), /missing/);
  await assert.rejects(concatenateNarrationFiles(["one.wav"], "out.wav"), /requires 2-8/);
});
