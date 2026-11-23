"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const { ProjectVault } = require("../gateway/project-vault");
const { MODEL_ID } = require("../gateway/huggingface-llm");
const {
  applyNegativePromptDefault,
  createEpisodeFromPlan,
  generateStoryPlan,
  loadExplicitYamlPlan,
  loadSharedNegativePrompt,
  parseJsonObject,
  resolveYamlImportCast,
  storyContentChanged,
  validateStoryPlan,
} = require("../gateway/story-studio");

function makeProject() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-story-"));
  const db = new DatabaseSync(":memory:");
  const vault = new ProjectVault({ db, projectsRoot: path.join(root, "projects") });
  return { root, db, vault, project: vault.createProject({ name: "Blue Hour" }) };
}

function plan() {
  return {
    title: "The Last Lantern",
    synopsis: "A lighthouse keeper follows a strange light across a quiet island.",
    scenes: [{
      title: "The Shore",
      description: "At dusk, Mira discovers a second light moving beneath the waves.",
      timeOfDay: "dusk",
      shots: [{
        title: "The moving light",
        visualPrompt: "Wide shot, a lone lighthouse keeper on a rocky coast at dusk, one blue light gliding below the waves.",
        negativePrompt: "text, watermark, extra people",
        camera: "slow push in",
        narration: "At the edge of the island, Mira saw the sea answer her lantern.",
      }],
    }],
  };
}

test("story plan parser accepts JSON fences and rejects malformed model output", () => {
  assert.deepEqual(parseJsonObject('```json\n{"title":"Test"}\n```'), { title: "Test" });
  assert.throws(() => parseJsonObject("I could not produce a plan"), /did not return a JSON plan/);
});

test("story plans enforce shot and prompt limits and allocate the requested runtime", () => {
  const validated = validateStoryPlan(plan(), { targetSeconds: 60 });
  assert.equal(validated.shotCount, 1);
  assert.equal(validated.scenes[0].shots[0].durationSeconds, 60);
  assert.throws(() => validateStoryPlan(plan(), { targetSeconds: 401 }), /target_seconds/);
  assert.throws(() => validateStoryPlan({
    ...plan(),
    scenes: Array.from({ length: 13 }, (_, index) => ({ ...plan().scenes[0], title: `Scene ${index}` })),
  }, { targetSeconds: 60 }), /1-12 scenes/);
});

test("story plans retain long negative prompts and shared defaults only fill blanks", () => {
  const longPrompt = "avoid " + "detail, ".repeat(170).trimEnd();
  const raw = {
    ...plan(),
    scenes: [{
      ...plan().scenes[0],
      shots: [
        { ...plan().scenes[0].shots[0], negativePrompt: longPrompt },
        { ...plan().scenes[0].shots[0], negativePrompt: "" },
      ],
    }],
  };
  const validated = validateStoryPlan(raw, { targetSeconds: 60 });
  const result = applyNegativePromptDefault(validated, "shared default");

  assert.equal(result.scenes[0].shots[0].negativePrompt, longPrompt);
  assert.equal(result.scenes[0].shots[1].negativePrompt, "shared default");
  assert.throws(() => validateStoryPlan({
    ...plan(),
    scenes: [{
      ...plan().scenes[0],
      shots: [{ ...plan().scenes[0].shots[0], negativePrompt: "x".repeat(4097) }],
    }],
  }, { targetSeconds: 60 }), /negativePrompt must be at most 4096/);
});

test("story plans can assign one or multiple selected characters per shot", () => {
  const characters = [
    { id: "mira-id", name: "Mira" },
    { id: "jon-id", name: "Jon" },
  ];
  const raw = plan();
  raw.scenes[0].shots = [
    { ...raw.scenes[0].shots[0], characters: ["Mira"] },
    { ...raw.scenes[0].shots[0], title: "Both characters", characters: ["Mira", "Jon"] },
  ];

  const validated = validateStoryPlan(raw, { targetSeconds: 60, availableCharacters: characters });
  assert.deepEqual(validated.scenes[0].shots[0].characters, ["Mira"]);
  assert.deepEqual(validated.scenes[0].shots[1].characters, ["Mira", "Jon"]);
  assert.throws(
    () => validateStoryPlan({
      ...raw,
      scenes: [{
        ...raw.scenes[0],
        shots: [{ ...raw.scenes[0].shots[0], characters: ["Unknown"] }],
      }],
    }, { targetSeconds: 60, availableCharacters: characters }),
    /must match the selected cast/,
  );
});

test("YAML imports complete the selected cast from saved project characters", () => {
  const raw = plan();
  raw.scenes[0].shots = [
    { ...raw.scenes[0].shots[0], characters: ["Mira"] },
    { ...raw.scenes[0].shots[0], title: "Jon appears", characters: ["Jon"] },
  ];
  const mira = { id: "mira-id", name: "Mira", projectId: "project-1" };
  const jon = { id: "jon-id", name: "Jon", projectId: "project-1" };

  const resolved = resolveYamlImportCast(raw, [mira], [mira, jon]);
  assert.deepEqual(resolved, [mira, jon]);
  assert.doesNotThrow(() => validateStoryPlan(raw, {
    targetSeconds: 60,
    availableCharacters: resolved,
  }));
  assert.throws(
    () => resolveYamlImportCast(raw, [mira], [mira]),
    /No saved cast profiles found for imported characters: Jon/,
  );
});

test("YAML import resolves characters appearing after the first shot", () => {
  const imported = loadExplicitYamlPlan();
  const visibleNames = [...new Set(imported.scenes.flatMap((scene) => scene.shots)
    .flatMap((shot) => shot.characters || []))];
  const savedCharacters = visibleNames.map((name) => ({
    id: `${name.toLowerCase()}-id`,
    name,
    projectId: "project-1",
  }));
  const firstShotNames = new Set(imported.scenes[0].shots[0].characters);
  const initiallySelected = savedCharacters.filter((character) => firstShotNames.has(character.name));
  const resolved = resolveYamlImportCast(imported, initiallySelected, savedCharacters);

  assert.deepEqual(resolved.map((character) => character.name), visibleNames);
  assert.doesNotThrow(() => validateStoryPlan(imported, {
    targetSeconds: imported.targetSeconds,
    title: imported.title,
    availableCharacters: resolved,
  }));
});

test("story draft generation uses Llama-3-8B-Lexi and applies the selected story style", async () => {
  const { root, db, vault, project } = makeProject();
  try {
    let request;
    let storyPrompt;
    const result = await generateStoryPlan({
      brief: "A lighthouse keeper follows a light under the sea.",
      targetSeconds: 60,
      project,
      genre: "mystery",
      tone: "suspenseful",
      endingStyle: "twist",
      characters: [{ name: "Mira", role: "lighthouse keeper", appearance: "silver hair", outfit: "navy coat" }],
      huggingFace: {
        available: true,
        resolveModel: (model) => model || MODEL_ID,
        chat: async (options) => {
          request = options;
          storyPrompt = options.messages[1].content;
          return {
            ok: true,
            json: async () => ({ choices: [{ message: { content: JSON.stringify(plan()) } }] }),
          };
        },
      },
    });
    assert.equal(request.model, MODEL_ID);
    assert.equal(result.title, "The Last Lantern");
    assert.equal(result.genre, "mystery");
    assert.equal(result.tone, "suspenseful");
    assert.equal(result.endingStyle, "twist");
    assert.equal(result.scenes[0].shots[0].visualPrompt.includes("lighthouse keeper"), true);
    assert.match(storyPrompt, /Genre: mystery\. Tone: suspenseful\. Ending: twist\./);
    assert.match(storyPrompt, /Mira: lighthouse keeper, silver hair, navy coat/);
    assert.match(storyPrompt, /exact selected character name/);
    assert.match(storyPrompt, /one turn per line/);
    assert.match(storyPrompt, /clearly adults/);
    assert.match(storyPrompt, /explicit adult erotica/);
    assert.match(storyPrompt, /on-screen/);
    assert.match(storyPrompt, /explicit language/);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("story draft reports the Numes model that answered after failover", async () => {
  const { root, db, project } = makeProject();
  const requestedModel = "dealignai/glm-5.3-flash-uncensored-fp8:free";
  const servedModel = "nvidia/nemotron-3-ultra-550b-a55b:free";
  try {
    const result = await generateStoryPlan({
      brief: "A lighthouse keeper follows a light under the sea.",
      targetSeconds: 60,
      project,
      provider: "numes",
      llm: {
        available: true,
        resolveModel: (model) => model || requestedModel,
        chat: async () => ({
          ok: true,
          numesModel: servedModel,
          json: async () => ({
            choices: [{ message: { content: JSON.stringify(plan()) } }],
          }),
        }),
      },
    });
    assert.equal(result.model, servedModel);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("story writing rejects unknown providers and names the missing credential", async () => {
  const { root, db, project } = makeProject();
  try {
    await assert.rejects(
      generateStoryPlan({
        brief: "A lighthouse keeper follows a light under the sea.",
        targetSeconds: 60,
        project,
        provider: "ollama",
      }),
      /does not support the "ollama" provider/,
    );
    await assert.rejects(
      generateStoryPlan({
        brief: "A lighthouse keeper follows a light under the sea.",
        targetSeconds: 60,
        project,
        provider: "huggingface",
        huggingFace: { available: false },
      }),
      /Configure HF_TOKEN/,
    );
    await assert.rejects(
      generateStoryPlan({
        brief: "A lighthouse keeper follows a light under the sea.",
        targetSeconds: 60,
        project,
        provider: "numes",
        llm: { available: false },
      }),
      /Configure NUMES_API_KEY/,
    );
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("creating a story persists the episode, scene, shot prompts, and narration atomically", () => {
  const { root, db, vault, project } = makeProject();
  try {
    const validated = validateStoryPlan(plan(), { targetSeconds: 60 });
    const episode = createEpisodeFromPlan(vault, project.id, validated, {
      brief: "A lighthouse keeper follows a light under the sea.",
      provider: "huggingface",
      characterIds: ["mira-id"],
      voiceProfile: "en-GB-SoniaNeural",
      genre: "mystery",
      tone: "suspenseful",
      endingStyle: "twist",
    });
    assert.equal(vault.listEpisodes(project.id).length, 1);
    assert.equal(episode.scenes.length, 1);
    const [shot] = vault.listShots(episode.scenes[0].id);
    assert.equal(shot.metadata.visualPrompt, plan().scenes[0].shots[0].visualPrompt);
    assert.equal(shot.metadata.narration, plan().scenes[0].shots[0].narration);
    assert.deepEqual(shot.metadata.characters, ["mira-id"]);
    assert.equal(shot.metadata.voiceProfile, "en-GB-SoniaNeural");
    assert.deepEqual(episode.metadata.characters, ["mira-id"]);
    assert.equal(episode.metadata.voiceProfile, "en-GB-SoniaNeural");
    assert.equal(episode.metadata.genre, "mystery");
    assert.equal(episode.metadata.tone, "suspenseful");
    assert.equal(episode.metadata.endingStyle, "twist");
    assert.equal(shot.durationSeconds, 60);

    const unchangedSave = {
      description: shot.description,
      camera: shot.camera,
      metadata: {
        ...shot.metadata,
        characters: [...shot.metadata.characters],
      },
    };
    assert.equal(storyContentChanged(shot, unchangedSave, episode), false);
    assert.equal(storyContentChanged(shot, {
      ...unchangedSave,
      metadata: { ...unchangedSave.metadata, characters: ["another-character"] },
    }, episode), true);

    const legacyShot = {
      ...shot,
      metadata: Object.fromEntries(
        Object.entries(shot.metadata).filter(([key]) => !["characters", "voiceProfile"].includes(key)),
      ),
    };
    assert.equal(storyContentChanged(legacyShot, unchangedSave, episode), false);

    assert.throws(() => createEpisodeFromPlan(vault, "missing-project", validated), /Project not found/);
    assert.equal(vault.listEpisodes(project.id).length, 1);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("creating a story persists per-shot character selections, including two-person shots", () => {
  const { root, db, vault, project } = makeProject();
  try {
    const characters = [
      { id: "mira-id", name: "Mira" },
      { id: "jon-id", name: "Jon" },
    ];
    const raw = plan();
    raw.scenes[0].shots = [
      { ...raw.scenes[0].shots[0], characters: ["Mira"] },
      { ...raw.scenes[0].shots[0], title: "Both characters", characters: ["Mira", "Jon"] },
    ];
    const validated = validateStoryPlan(raw, { targetSeconds: 60, availableCharacters: characters });
    const episode = createEpisodeFromPlan(vault, project.id, validated, {
      characterIds: characters.map((character) => character.id),
      selectedCharacters: characters,
    });
    const shots = vault.listShots(episode.scenes[0].id);

    assert.deepEqual(shots[0].metadata.characters, ["mira-id"]);
    assert.deepEqual(shots[1].metadata.characters, ["mira-id", "jon-id"]);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("production YAML imports all explicit beats and the shared negative prompt", () => {
  const plan = loadExplicitYamlPlan();
  const expectedNegative = fs.readFileSync(
    path.join(__dirname, "..", "negatives", "artifact_neg.txt"),
    "utf8",
  ).replace(/\r\n?/g, "\n").trim();

  assert.equal(plan.title, "Midnight Heat");
  assert.equal(plan.shotCount, 70);
  assert.equal(plan.targetSeconds, 344);
  assert.equal(plan.defaultNegativePrompt, expectedNegative);
  assert.equal(plan.scenes.length, 12);
  assert.equal(plan.scenes[0].title, "Hallway");
  assert.equal(plan.scenes[0].shots[0].sourceBeat.id, "c01");
  assert.ok(plan.starterBrief.length > 0);
  assert.ok(plan.starterBrief.length <= 6000);
  assert.ok(plan.starterBrief.includes("All characters are adults."));
  assert.ok(plan.scenes.flatMap((scene) => scene.shots)
    .every((shot) => shot.negativePrompt === expectedNegative));
});

test("Film Studio defaults load the shared negative prompt independently of story YAML", () => {
  const expectedNegative = fs.readFileSync(
    path.join(__dirname, "..", "negatives", "artifact_neg.txt"),
    "utf8",
  ).replace(/\r\n?/g, "\n").trim();

  assert.equal(loadSharedNegativePrompt(), expectedNegative);
});

test("imported source-beat metadata is retained on the editable shot", () => {
  const { root, db, vault, project } = makeProject();
  try {
    const sourceBeat = {
      id: "c01",
      scene: 1,
      build: "fm",
      startFrame: "startframes/c01.png",
      loras: [],
      chain: true,
      sampler: "euler_simple",
      steps: [2, 3],
      cfg: 1,
      duration: 5,
      width: 832,
      height: 480,
      noise: 2,
      light: "tube_flicker",
      mark_state: "none",
      thread: "",
      profile: "draft",
      anchors: {
        face: "Faiz and Juveriya",
        body: "adult posture",
        wardrobe: "tracked wardrobe",
        prop: "hallway prop",
        light: "tube flicker",
        geography: "narrow hallway",
      },
      audio_mode: "none",
    };
    const raw = plan();
    raw.scenes[0].shots[0].sourceBeat = sourceBeat;
    const validated = validateStoryPlan(raw, { targetSeconds: 60 });
    const episode = createEpisodeFromPlan(vault, project.id, validated, {
      sourceFile: "docs/beats.yaml",
    });

    const [shot] = vault.listShots(episode.scenes[0].id);

    assert.deepEqual(shot.metadata.sourceBeat, sourceBeat);
    assert.equal(episode.metadata.sourceFile, "docs/beats.yaml");
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("imported source beats enforce the noise-duration rule and chained identity anchors", () => {
  const { root, db, vault, project } = makeProject();
  try {
    const raw = plan();
    const first = {
      id: "c01",
      scene: 1,
      build: "fm",
      startFrame: "startframes/c01.png",
      loras: [],
      chain: true,
      sampler: "euler_simple",
      steps: [2, 3],
      cfg: 1,
      duration: 5,
      width: 832,
      height: 480,
      noise: 2,
      light: "tube_flicker",
      mark_state: "none",
      thread: "",
      profile: "draft",
      anchors: {
        face: "Faiz and Juveriya",
        body: "adult posture",
        wardrobe: "tracked wardrobe",
        prop: "hallway prop",
        light: "tube flicker",
        geography: "narrow hallway",
      },
      audio_mode: "none",
    };
    const second = {
      ...first,
      id: "c02",
      startFrame: "startframes/c02.png",
      anchors: { ...first.anchors, wardrobe: "different outfit" },
    };
    raw.scenes[0].shots[0].sourceBeat = first;
    raw.scenes[0].shots.push({ ...raw.scenes[0].shots[0], sourceBeat: second });
    assert.throws(
      () => validateStoryPlan(raw, { targetSeconds: 60 }),
      /sourceBeat anchors\.wardrobe must match the previous beat when chain is true/,
    );

    second.anchors.wardrobe = first.anchors.wardrobe;
    second.duration = 4;
    assert.throws(
      () => validateStoryPlan(raw, { targetSeconds: 60 }),
      /sourceBeat duration must match meta\.duration \(5\)/,
    );

    second.duration = 5;
    second.audio_mode = "s2v";
    assert.throws(
      () => validateStoryPlan(raw, { targetSeconds: 60 }),
      /s2v requires an audio sidecar/,
    );
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
