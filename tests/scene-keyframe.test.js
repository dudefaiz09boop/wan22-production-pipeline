const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {
  assertSceneKeyframeReady,
  attachCompletedSceneKeyframe,
  buildSceneKeyframeRequest,
} = require("../gateway/scene-keyframe");

function makeShot(overrides = {}) {
  return {
    id: "shot-1",
    projectId: "project-1",
    title: "Meet in the station",
    description: "Two travelers meet beneath the platform clock.",
    keyframeAssetId: "raw-character-photo",
    metadata: {
      sourceBeat: { id: "c01", startFrame: "startframes/c01.png", chain: false },
    },
    ...overrides,
  };
}

test("scene rebuild uses start frame for composition and character photos for identity only", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "scene-keyframe-build-"));
  const startframes = path.join(root, "startframes");
  fs.mkdirSync(startframes);
  const scenePath = path.join(startframes, "c01.png");
  const characterPath = path.join(root, "character.png");
  const sceneBytes = Buffer.from("scene composition");
  const characterBytes = Buffer.from("character identity");
  fs.writeFileSync(scenePath, sceneBytes);
  fs.writeFileSync(characterPath, characterBytes);
  const requestedAssets = [];
  const vault = {
    getAsset(id) {
      requestedAssets.push(id);
      return null;
    },
  };
  const storyContext = {
    resolve(options) {
      assert.equal(options.shotId, "shot-1");
      return {
        prompt: "A station platform beneath a clock.",
        negativePrompt: "blurred faces",
        references: [{
          path: characterPath,
          label: "Mira",
          characterId: "character-1",
        }],
      };
    },
  };

  try {
    const request = await buildSceneKeyframeRequest({
      vault,
      storyContext,
      shot: makeShot(),
      videoProfile: { width: 832, height: 480 },
      projectRoot: root,
    });
    assert.equal(request.sceneReferenceSource, "yaml-start-frame");
    assert.equal(request.characterReferenceCount, 1);
    assert.equal(request.referenceImages.length, 2);
    assert.equal(request.referenceImages[0].data, sceneBytes.toString("base64"));
    assert.equal(request.referenceImages[0].label, "scene composition and continuity");
    assert.equal(request.referenceImages[1].data, characterBytes.toString("base64"));
    assert.match(request.prompt, /Image 2 is the identity reference for Mira/);
    assert.match(request.prompt, /do not copy their original background/);
    assert.deepEqual(requestedAssets, []);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("scene rebuild rejects more identity references than the Krea-2 input limit allows", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "scene-keyframe-limit-"));
  const startframes = path.join(root, "startframes");
  fs.mkdirSync(startframes);
  fs.writeFileSync(path.join(startframes, "c01.png"), Buffer.from("scene"));
  const referencePath = path.join(root, "character.png");
  fs.writeFileSync(referencePath, Buffer.from("character"));
  const storyContext = {
    resolve: () => ({
      prompt: "A scene with three characters.",
      negativePrompt: "",
      references: ["Mira", "Jon", "Kai"].map((label, index) => ({
        path: referencePath,
        label,
        characterId: `character-${index}`,
      })),
    }),
  };

  try {
    await assert.rejects(
      buildSceneKeyframeRequest({
        vault: { getAsset: () => null },
        storyContext,
        shot: makeShot(),
        videoProfile: { width: 832, height: 480 },
        projectRoot: root,
      }),
      (error) => error.status === 413 && /Reduce the references to 2 or fewer/.test(error.message),
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("chained scene rebuild uses the previous video's final frame instead of the shot's stale keyframe", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "scene-keyframe-chain-"));
  const previousVideoPath = path.join(root, "previous.webm");
  const existingScenePath = path.join(root, "existing-scene.png");
  const characterPath = path.join(root, "character.png");
  fs.writeFileSync(previousVideoPath, Buffer.from("previous video"));
  fs.writeFileSync(existingScenePath, Buffer.from("stale scene frame"));
  fs.writeFileSync(characterPath, Buffer.from("character"));
  const sceneAsset = {
    id: "existing-scene",
    projectId: "project-1",
    category: "scenes",
    filePath: existingScenePath,
    mime: "image/png",
    metadata: { keyframeRole: "scene" },
  };
  const previousShot = {
    id: "shot-0",
    sceneId: "scene-0",
    number: 1,
    segmentAssetId: "previous-video",
    metadata: { globalNumber: 1 },
  };
  const shot = makeShot({
    sceneId: "scene-1",
    number: 2,
    metadata: {
      globalNumber: 2,
      sourceBeat: { id: "c02", startFrame: "startframes/c02.png", chain: true },
      sceneKeyframeAssetId: "existing-scene",
    },
  });
  const vault = {
    getAsset(id) {
      if (id === "existing-scene") return sceneAsset;
      if (id === "previous-video") return { filePath: previousVideoPath, mime: "video/webm" };
      return null;
    },
    getScene(id) {
      return id === "scene-1" ? { id, episodeId: "episode-1" } : null;
    },
    listScenes() {
      return [{ id: "scene-0", episodeId: "episode-1" }, { id: "scene-1", episodeId: "episode-1" }];
    },
    listShots(sceneId) {
      if (sceneId === "scene-0") return [previousShot];
      return sceneId === "scene-1" ? [shot] : [];
    },
  };
  const storyContext = {
    resolve: () => ({
      prompt: "Continue the scene.",
      negativePrompt: "",
      references: [{ path: characterPath, label: "Mira", characterId: "character-1" }],
    }),
  };
  const execFileImpl = async (_command, args) => {
    fs.writeFileSync(args[args.length - 1], Buffer.from("previous final frame"));
  };

  try {
    const request = await buildSceneKeyframeRequest({
      vault,
      storyContext,
      shot,
      videoProfile: { width: 832, height: 480 },
      projectRoot: root,
      execFileImpl,
    });
    assert.equal(request.sceneReferenceSource, "previous-video");
    assert.equal(
      request.referenceImages[0].data,
      Buffer.from("previous final frame").toString("base64"),
    );
    assert.equal(request.referenceImages[0].label, "scene composition and continuity");
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("video rendering requires an attached generated scene keyframe for referenced characters", () => {
  const shot = makeShot({
    keyframeAssetId: "raw-character-photo",
    metadata: {},
  });
  const context = { references: [{ characterId: "character-1" }] };
  assert.throws(
    () => assertSceneKeyframeReady({ vault: { getAsset: () => null }, shot, context }),
    (error) => error.status === 409 && /cannot be sent directly/.test(error.message),
  );
  assert.doesNotThrow(() => assertSceneKeyframeReady({
    vault: { getAsset: () => null },
    shot,
    context,
    allowDirectReference: true,
  }));
  for (const keyframeSource of ["yaml-start-frame", "previous-video"]) {
    assert.doesNotThrow(() => assertSceneKeyframeReady({
      vault: { getAsset: () => null },
      shot,
      context,
      keyframeSource,
    }));
  }
  assert.throws(
    () => assertSceneKeyframeReady({
      vault: { getAsset: () => null },
      shot,
      context,
      keyframeSource: "uploaded-still",
    }),
    (error) => error.status === 409 && /cannot be sent directly/.test(error.message),
  );

  const asset = {
    id: "scene-frame",
    projectId: shot.projectId,
    category: "scenes",
    sourceRef: `shot-scene-keyframe:${shot.id}:task-1`,
    metadata: { keyframeRole: "scene" },
  };
  shot.keyframeAssetId = asset.id;
  shot.metadata.sceneKeyframeAssetId = asset.id;
  assert.doesNotThrow(() => assertSceneKeyframeReady({
    vault: { getAsset: () => asset },
    shot,
    context,
  }));
});

test("only the completed reconstruction task for a shot can be attached", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "scene-keyframe-attach-"));
  const outputDir = path.join(root, "downloads");
  fs.mkdirSync(outputDir);
  const imagePath = path.join(outputDir, "candidate.png");
  fs.writeFileSync(imagePath, Buffer.from("generated scene"));
  const shot = makeShot({
    metadata: {
      sceneKeyframeTaskId: "task-1",
      sceneKeyframeSource: "yaml-start-frame",
    },
  });
  let insertedAsset;
  let updatedShot;
  const vault = {
    addAssetFromPath(options) {
      insertedAsset = options;
      return { id: "scene-frame", ...options };
    },
    updateShot(_id, patch) {
      updatedShot = patch;
      return { ...shot, ...patch };
    },
  };
  const kaggle = {
    getJob: () => ({
      id: "task-1",
      kind: "image",
      status: "completed",
      model: "krea2",
      result: { images: [{ imagePath }] },
    }),
  };

  try {
    const attached = attachCompletedSceneKeyframe({
      vault,
      kaggle,
      shot,
      taskId: "task-1",
      outputDir,
    });
    assert.equal(insertedAsset.category, "scenes");
    assert.equal(insertedAsset.sourceRef, "shot-scene-keyframe:shot-1:task-1");
    assert.equal(updatedShot.keyframeAssetId, "scene-frame");
    assert.equal(updatedShot.metadata.sceneKeyframeAssetId, "scene-frame");
    assert.equal(attached.asset.id, "scene-frame");
    assert.equal(attached.shot.status, "keyframe_ready");

    assert.throws(
      () => attachCompletedSceneKeyframe({
        vault,
        kaggle,
        shot,
        taskId: "other-task",
        outputDir,
      }),
      { status: 409 },
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
