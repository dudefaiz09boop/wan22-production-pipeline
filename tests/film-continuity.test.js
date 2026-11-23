const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {
  extractVideoLastFrame,
  previousShotInEpisode,
  resolveFilmShotKeyframe,
  sourceBeatFramePath,
} = require("../gateway/film-continuity");

function makeEpisode(shotsByScene) {
  const scenes = Object.keys(shotsByScene).map((id, index) => ({
    id,
    episodeId: "episode",
    number: index + 1,
  }));
  const shots = new Map(Object.entries(shotsByScene));
  return {
    getScene(id) {
      return scenes.find((scene) => scene.id === id) || null;
    },
    listScenes(episodeId) {
      return scenes.filter((scene) => scene.episodeId === episodeId);
    },
    listShots(sceneId) {
      return shots.get(sceneId) || [];
    },
  };
}

test("YAML start frames resolve only to the matching in-project cNN file", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "film-start-frame-test-"));
  const startframes = path.join(root, "startframes");
  fs.mkdirSync(startframes);
  const file = path.join(startframes, "c02.png");
  fs.writeFileSync(file, Buffer.from("image"));
  try {
    assert.equal(sourceBeatFramePath({ id: "c02", startFrame: "startframes/c02.png" }, root), file);
    assert.throws(
      () => sourceBeatFramePath({ id: "c01", startFrame: "startframes/c02.png" }, root),
      { status: 400 },
    );
    assert.throws(
      () => sourceBeatFramePath({ id: "c02", startFrame: "startframes/../.env" }, root),
      { status: 400 },
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("previous beat lookup follows global story order across scenes", () => {
  const first = { id: "first", sceneId: "one", number: 1, metadata: { globalNumber: 1 } };
  const second = { id: "second", sceneId: "two", number: 1, metadata: { globalNumber: 2 } };
  const third = { id: "third", sceneId: "two", number: 2, metadata: { globalNumber: 3 } };
  const vault = makeEpisode({ one: [first], two: [third, second] });

  assert.equal(previousShotInEpisode(vault, third).id, "second");
  assert.equal(previousShotInEpisode(vault, first), null);
});

test("chained beats use the previous video's final frame and require order", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "film-chain-test-"));
  const videoPath = path.join(root, "previous.mp4");
  const sceneFramePath = path.join(root, "reconstructed.png");
  const uploadedStillPath = path.join(root, "uploaded-still.png");
  const stillPath = path.join(root, "startframes");
  fs.mkdirSync(stillPath);
  fs.writeFileSync(videoPath, Buffer.from("video"));
  fs.writeFileSync(sceneFramePath, Buffer.from("rebuilt scene"));
  fs.writeFileSync(uploadedStillPath, Buffer.from("uploaded still"));
  fs.writeFileSync(path.join(stillPath, "c01.png"), Buffer.from("first still"));
  fs.writeFileSync(path.join(stillPath, "c02.png"), Buffer.from("still"));
  const first = {
    id: "first",
    projectId: "project",
    sceneId: "one",
    number: 1,
    segmentAssetId: "video-asset",
    metadata: {
      globalNumber: 1,
      sourceBeat: { id: "c01", startFrame: "startframes/c01.png", chain: true },
    },
  };
  const second = {
    id: "second",
    projectId: "project",
    sceneId: "two",
    number: 1,
    metadata: { globalNumber: 2, sourceBeat: { id: "c02", startFrame: "startframes/c02.png", chain: true } },
  };
  const vault = {
    ...makeEpisode({ one: [first], two: [second] }),
    getAsset(id) {
      if (id === "video-asset") return { projectId: "project", filePath: videoPath, mime: "video/mp4" };
      if (id === "scene-image") return { projectId: "project", filePath: sceneFramePath, mime: "image/png" };
      if (id === "uploaded-still") {
        return { projectId: "project", filePath: uploadedStillPath, mime: "image/png", metadata: {} };
      }
      return null;
    },
  };
  let outputPath;
  const execFileImpl = async (_file, args) => {
    outputPath = args[args.length - 1];
    fs.writeFileSync(outputPath, Buffer.from("last frame"));
  };

  try {
    const result = await resolveFilmShotKeyframe({
      vault,
      shot: second,
      profile: { width: 480, height: 320 },
      projectRoot: root,
      execFileImpl,
    });
    assert.equal(result.keyframeBase64, Buffer.from("last frame").toString("base64"));
    assert.equal(result.keyframeSource, "previous-video");
    assert.equal(fs.existsSync(path.dirname(outputPath)), false);

    second.metadata.sceneKeyframeAssetId = "scene-image";
    const rebuilt = await resolveFilmShotKeyframe({
      vault,
      shot: second,
      profile: { width: 480, height: 320 },
      projectRoot: root,
      execFileImpl,
    });
    assert.equal(rebuilt.keyframeBase64, Buffer.from("rebuilt scene").toString("base64"));
    assert.equal(rebuilt.keyframeSource, "reconstructed-scene");
    const directSource = await resolveFilmShotKeyframe({
      vault,
      shot: second,
      profile: { width: 480, height: 320 },
      projectRoot: root,
      keyframeMode: "source",
    });
    assert.equal(directSource.keyframeBase64, Buffer.from("still").toString("base64"));
    assert.equal(directSource.keyframeSource, "yaml-start-frame");
    second.keyframeAssetId = "uploaded-still";
    const uploadedSource = await resolveFilmShotKeyframe({
      vault,
      shot: second,
      profile: { width: 480, height: 320 },
      projectRoot: root,
      keyframeMode: "source",
    });
    assert.equal(uploadedSource.keyframeBase64, Buffer.from("uploaded still").toString("base64"));
    assert.equal(uploadedSource.keyframeSource, "uploaded-still");
    delete second.keyframeAssetId;
    delete second.metadata.sceneKeyframeAssetId;

    second.metadata.sourceBeat.chain = false;
    const freshStart = await resolveFilmShotKeyframe({
      vault,
      shot: second,
      profile: { width: 480, height: 320 },
      projectRoot: root,
    });
    assert.equal(freshStart.keyframeSource, "yaml-start-frame");

    const episodeStart = await resolveFilmShotKeyframe({
      vault,
      shot: first,
      profile: { width: 480, height: 320 },
      projectRoot: root,
    });
    assert.equal(episodeStart.keyframeSource, "yaml-start-frame");

    second.metadata.sourceBeat.chain = true;
    first.segmentAssetId = null;
    await assert.rejects(
      resolveFilmShotKeyframe({
        vault,
        shot: second,
        profile: { width: 480, height: 320 },
        projectRoot: root,
      }),
      { status: 409 },
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("last-frame extraction seeks from the end and produces an inline image", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "film-last-frame-test-"));
  const videoPath = path.join(root, "source.mp4");
  fs.writeFileSync(videoPath, Buffer.from("video"));
  let outputPath;
  try {
    const result = await extractVideoLastFrame(videoPath, {
      width: 1280,
      height: 720,
      execFileImpl: async (_file, args) => {
        assert.ok(args.includes("-sseof"));
        assert.ok(args.includes("-1"));
        assert.match(args[args.indexOf("-vf") + 1], /^reverse,scale=/);
        outputPath = args[args.length - 1];
        fs.writeFileSync(outputPath, Buffer.from("jpeg"));
      },
    });
    assert.equal(result.keyframeBase64, Buffer.from("jpeg").toString("base64"));
    assert.equal(result.keyframeFilename, "continuity-handoff.jpg");
    assert.equal(fs.existsSync(path.dirname(outputPath)), false);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
