"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const { CharacterStore } = require("../gateway/characters");
const { ProjectVault } = require("../gateway/project-vault");
const { StoryContext, trimTo } = require("../gateway/story-context");

function makeWorld() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-story-"));
  const db = new DatabaseSync(":memory:");
  const vault = new ProjectVault({ db, projectsRoot: path.join(root, "projects"), logger: { error() {} } });
  const characters = new CharacterStore({ db, dataDir: root, logger: { error() {} } });
  const story = new StoryContext({ db, characters, vault, logger: { error() {} } });

  const project = vault.createProject({
    name: "Harbour Lights",
    stylePrompt: "muted teal, 35mm film grain",
    negativePrompt: "oversaturated",
  });
  const episode = vault.createEpisode(project.id, { number: 1, title: "First Light", synopsis: "Mira returns home." });
  const location = vault.addLocation(project.id, {
    name: "The jetty", description: "weathered planks, a single lamp",
  });
  const previous = vault.createScene(project.id, episode.id, {
    number: 1, title: "Arrival", description: "Mira steps off the ferry in the rain",
  });
  const current = vault.createScene(project.id, episode.id, {
    number: 2,
    title: "The Jetty",
    description: "Mira finds the compass on the jetty",
    timeOfDay: "dusk",
    locationId: location.id,
  });
  const shot = vault.createShot(project.id, current.id, {
    number: 1, title: "Wide", description: "wide shot of the jetty",
    camera: "slow dolly in", metadata: { characters: ["placeholder"] },
  });
  const mira = characters.create({
    projectId: project.id,
    name: "Mira Voss",
    role: "protagonist",
    appearance: "silver hair, amber eyes",
    outfit: "navy flight jacket",
  });
  const jonah = characters.create({
    projectId: "global",
    name: "Jonah Reed",
    appearance: "broad shoulders, grey beard",
  });
  vault.addProp(project.id, { name: "brass compass", description: "Mira's heirloom" });
  const reference = characters.addReference(mira.id, {
    kind: "face",
    label: "front portrait",
    data: Buffer.from("face").toString("base64"),
    mime: "image/png",
  });
  // The shot's metadata is written before the characters exist, so point it at
  // the real ids now.
  vault.updateShot(shot.id, { metadata: { characters: [mira.id, jonah.id] } });

  return { root, db, vault, characters, story, project, episode, previous, current, shot, mira, jonah, reference };
}

test("resolving a shot loads style, characters, continuity, location and references", () => {
  const world = makeWorld();
  try {
    const context = world.story.resolveShot(world.shot.id, { basePrompt: "slow push in" });
    assert.match(context.prompt, /^slow push in/);
    assert.match(context.prompt, /project style: muted teal/);
    assert.match(context.prompt, /Mira Voss/);
    assert.match(context.prompt, /outfit: navy flight jacket/);
    assert.match(context.prompt, /Jonah Reed/);
    assert.match(context.prompt, /weathered planks/);
    assert.match(context.prompt, /time of day: dusk/);
    assert.match(context.prompt, /brass compass/);
    assert.match(context.prompt, /Mira steps off the ferry/);
    assert.equal(context.negativePrompt, "oversaturated");
    assert.equal(context.chain.shotId, world.shot.id);
    assert.equal(context.chain.sceneId, world.current.id);
    assert.deepEqual(context.characters.map((character) => character.name), ["Mira Voss", "Jonah Reed"]);
    assert.equal(context.references.length, 1);
    assert.equal(context.references[0].characterId, world.mira.id);
    assert.equal(context.references[0].path, world.reference.filePath);
    assert.ok(context.timeline.length >= 0);
  } finally {
    world.db.close();
    fs.rmSync(world.root, { recursive: true, force: true });
  }
});

test("a scene-specific character state overrides the project default", () => {
  const world = makeWorld();
  try {
    world.vault.setCharacterState(world.project.id, {
      characterId: world.mira.id,
      outfit: "day dress",
    });
    world.vault.setCharacterState(world.project.id, {
      characterId: world.mira.id,
      sceneId: world.current.id,
      outfit: "oilskin coat",
      appearanceNotes: "soaked hair",
    });
    const context = world.story.resolveShot(world.shot.id);
    assert.match(context.prompt, /current outfit: oilskin coat/);
    assert.match(context.prompt, /current state: soaked hair/);
    assert.equal(context.characters[0].outfit, "oilskin coat");
  } finally {
    world.db.close();
    fs.rmSync(world.root, { recursive: true, force: true });
  }
});

test("resolve works from an episode or scene id and normalises the chain", () => {
  const world = makeWorld();
  try {
    const fromEpisode = world.story.resolve({ episodeId: world.episode.id });
    assert.equal(fromEpisode.chain.episodeId, world.episode.id);
    assert.equal(fromEpisode.chain.sceneId, null);
    assert.match(fromEpisode.prompt, /project style/);

    const fromScene = world.story.resolve({ sceneId: world.current.id });
    assert.equal(fromScene.chain.sceneId, world.current.id);
    assert.match(fromScene.prompt, /Mira finds the compass/);
  } finally {
    world.db.close();
    fs.rmSync(world.root, { recursive: true, force: true });
  }
});

test("explicit characterIds win over shot metadata", () => {
  const world = makeWorld();
  try {
    const context = world.story.resolveShot(world.shot.id, { characterIds: [world.jonah.id] });
    assert.deepEqual(context.characters.map((character) => character.id), [world.jonah.id]);
    assert.ok(!context.prompt.includes("Mira Voss"));
  } finally {
    world.db.close();
    fs.rmSync(world.root, { recursive: true, force: true });
  }
});

test("an explicit empty characterIds list suppresses episode defaults", () => {
  const world = makeWorld();
  try {
    const context = world.story.resolveShot(world.shot.id, { characterIds: [] });
    assert.deepEqual(context.characters, []);
    assert.ok(!context.prompt.includes("Mira Voss"));
    assert.ok(!context.prompt.includes("Jonah Reed"));
  } finally {
    world.db.close();
    fs.rmSync(world.root, { recursive: true, force: true });
  }
});

test("the resolved prompt never exceeds its cap", () => {
  const world = makeWorld();
  try {
    const context = world.story.resolveShot(world.shot.id, {
      basePrompt: "x".repeat(5000),
      maxChars: 300,
    });
    assert.equal(context.prompt.length, 300);
    assert.ok(context.prompt.endsWith("…"));
    assert.equal(trimTo("abcdef", 4), "abc…");
    assert.equal(trimTo("abc", 4), "abc");
  } finally {
    world.db.close();
    fs.rmSync(world.root, { recursive: true, force: true });
  }
});

test("unknown projects and shots fail with a clear 404", () => {
  const world = makeWorld();
  try {
    assert.throws(() => world.story.resolve({ projectId: "missing" }), /Project not found/);
    assert.throws(() => world.story.resolveShot("missing-shot"), /Shot not found/);
    assert.throws(() => world.story.resolve({}), /projectId is required/);
  } finally {
    world.db.close();
    fs.rmSync(world.root, { recursive: true, force: true });
  }
});
