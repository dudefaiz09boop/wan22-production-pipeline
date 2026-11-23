"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const { ProjectVault, sanitizeFilename, CATEGORY_DIRECTORIES } = require("../gateway/project-vault");

function makeVault() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-vault-"));
  const db = new DatabaseSync(":memory:");
  const vault = new ProjectVault({ db, projectsRoot: path.join(root, "projects"), logger: { error() {} } });
  return { root, db, vault };
}

test("createProject uses a readable story folder and keeps beat videos at its root", () => {
  const { root, db, vault } = makeVault();
  try {
    const project = vault.createProject({
      name: "Harbour Lights",
      description: "a slow-burn coastal series",
      stylePrompt: "muted teal, 35mm",
    });
    assert.equal(project.name, "Harbour Lights");
    assert.equal(project.stylePrompt, "muted teal, 35mm");
    assert.equal(path.basename(project.rootPath), "harbour-lights");
    assert.equal(vault.projectDirectory(project.id), project.rootPath);
    assert.equal(vault.projectDirectory(project.id, "videos"), project.rootPath);
    for (const [category, directory] of Object.entries(CATEGORY_DIRECTORIES)) {
      if (category === "videos") {
        assert.equal(fs.existsSync(path.join(project.rootPath, directory)), false);
        continue;
      }
      assert.ok(
        fs.existsSync(path.join(project.rootPath, directory)),
        `${directory} should exist`,
      );
    }
    assert.equal(vault.listProjects().length, 1);
    assert.equal(vault.updateProject(project.id, { stylePrompt: "warm amber" }).stylePrompt, "warm amber");
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("story folders avoid existing paths and reserved Windows names", () => {
  const { root, db, vault } = makeVault();
  try {
    fs.mkdirSync(path.join(root, "projects", "same-story"), { recursive: true });
    const collision = vault.createProject({ name: "Same Story" });
    const reserved = vault.createProject({ name: "CON" });

    assert.equal(path.basename(collision.rootPath), "same-story-1");
    assert.equal(path.basename(reserved.rootPath), "story-con");
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("series, episodes, scenes and shots keep their order and hierarchy", () => {
  const { root, db, vault } = makeVault();
  try {
    const series = vault.createSeries({ name: "Coast" });
    const project = vault.createProject({ name: "Harbour Lights", seriesId: series.id });
    assert.equal(vault.getProject(project.id).seriesId, series.id);

    const episode2 = vault.createEpisode(project.id, { number: 2, title: "Undertow" });
    const episode1 = vault.createEpisode(project.id, { number: 1, title: "First Light" });
    assert.deepEqual(vault.listEpisodes(project.id).map((entry) => entry.id), [episode1.id, episode2.id]);

    const scene2 = vault.createScene(project.id, episode1.id, { number: 2, title: "The Jetty" });
    const scene1 = vault.createScene(project.id, episode1.id, {
      number: 1, title: "Arrival", description: "Mira steps off the ferry",
    });
    assert.deepEqual(vault.listScenes(episode1.id).map((entry) => entry.id), [scene1.id, scene2.id]);

    const shotB = vault.createShot(project.id, scene1.id, { number: 2, title: "Close up" });
    const shotA = vault.createShot(project.id, scene1.id, { number: 1, title: "Wide", camera: "slow dolly" });
    assert.deepEqual(vault.listShots(scene1.id).map((entry) => entry.id), [shotA.id, shotB.id]);
    assert.equal(vault.getShot(shotA.id).camera, "slow dolly");

    const updated = vault.updateShot(shotA.id, { status: "keyframe_ready", keyframeAssetId: "asset-kf" });
    assert.equal(updated.status, "keyframe_ready");
    assert.equal(updated.keyframeAssetId, "asset-kf");

    vault.removeEpisode(episode1.id);
    assert.equal(vault.getScene(scene1.id), null);
    assert.equal(vault.getShot(shotA.id), null);
    assert.equal(vault.listEpisodes(project.id).length, 1);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("assets are written into their category and summarised", () => {
  const { root, db, vault } = makeVault();
  try {
    const project = vault.createProject({ name: "Harbour Lights" });
    const image = vault.addAsset({
      projectId: project.id,
      category: "images",
      name: "keyframe-1.png",
      data: Buffer.from("png-bytes").toString("base64"),
      mime: "image/png",
      sourceRef: "keyframe:1",
      metadata: { shotId: "shot-1" },
    });
    assert.ok(fs.existsSync(image.filePath));
    assert.equal(path.basename(path.dirname(image.filePath)), "Images");
    assert.equal(image.bytes, 9);
    assert.deepEqual(image.metadata, { shotId: "shot-1" });

    vault.addAsset({
      projectId: project.id,
      category: "audio",
      name: "line.wav",
      data: Buffer.from("wav").toString("base64"),
      mime: "audio/wav",
    });
    const summary = vault.summary(project.id);
    assert.equal(summary.assets.images.count, 1);
    assert.equal(summary.assets.audio.count, 1);
    assert.equal(summary.assets.videos.count, 0);
    assert.equal(summary.counts.episodes, 0);

    const images = vault.listAssets({ projectId: project.id, category: "images" });
    assert.equal(images.length, 1);
    assert.throws(
      () => vault.addAsset({ projectId: project.id, category: "secrets", name: "x", data: "AAAA" }),
      /category/,
    );
    assert.throws(() => vault.addAsset({ projectId: project.id, category: "images", name: "x", data: "nope!!" }), /base64/);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("addAssetFromPath copies once per sourceRef and deleteAsset removes the file", () => {
  const { root, db, vault } = makeVault();
  try {
    const project = vault.createProject({ name: "Harbour Lights" });
    const source = path.join(root, "output.webm");
    fs.writeFileSync(source, "video-bytes");

    const first = vault.addAssetFromPath({
      projectId: project.id,
      category: "videos",
      filePath: source,
      name: "segment-1.webm",
      mime: "video/webm",
      sourceRef: "compute:job-1",
    });
    const second = vault.addAssetFromPath({
      projectId: project.id,
      category: "videos",
      filePath: source,
      name: "segment-1.webm",
      mime: "video/webm",
      sourceRef: "compute:job-1",
    });
    assert.equal(first.id, second.id);
    assert.ok(fs.existsSync(source), "the compute output should be copied, not moved");
    assert.equal(path.dirname(first.filePath), project.rootPath);
    const files = fs.readdirSync(project.rootPath)
      .filter((name) => fs.statSync(path.join(project.rootPath, name)).isFile());
    assert.equal(files.length, 1);

    vault.deleteAsset(first.id);
    assert.equal(fs.existsSync(first.filePath), false);
    assert.equal(vault.getAsset(first.id), null);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("existing UUID-root projects retain their Videos directory", () => {
  const { root, db, vault } = makeVault();
  try {
    const project = vault.createProject({ name: "Existing Story" });
    const legacyRoot = path.join(root, "projects", project.id);
    fs.mkdirSync(path.join(legacyRoot, "Videos"), { recursive: true });
    db.prepare("UPDATE vault_projects SET root_path = ? WHERE id = ?").run(legacyRoot, project.id);

    assert.equal(vault.projectDirectory(project.id, "videos"), path.join(legacyRoot, "Videos"));
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("deleting an attached video detaches it from its shot", () => {
  const { root, db, vault } = makeVault();
  try {
    const project = vault.createProject({ name: "Harbour Lights" });
    const episode = vault.createEpisode(project.id, { number: 1, title: "First Light" });
    const scene = vault.createScene(project.id, episode.id, { number: 1, title: "The Coast" });
    const shot = vault.createShot(project.id, scene.id, { number: 1, title: "The lighthouse" });
    const video = vault.addAsset({
      projectId: project.id,
      category: "videos",
      name: "shot.webm",
      data: Buffer.from("video").toString("base64"),
      mime: "video/webm",
    });
    vault.updateShot(shot.id, { segmentAssetId: video.id, status: "video_ready" });

    vault.deleteAsset(video.id);

    const updated = vault.getShot(shot.id);
    assert.equal(updated.segmentAssetId, null);
    assert.equal(updated.status, "draft");
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("filenames are sanitised to a single safe path segment", () => {
  assert.equal(sanitizeFilename("../../evil.png"), "evil.png");
  assert.equal(sanitizeFilename("C:\\Users\\me\\shot 1.png"), "shot 1.png");
  assert.equal(sanitizeFilename(".."), "asset");
  assert.equal(sanitizeFilename("bad:name?.png"), "badname.png");
  assert.equal(sanitizeFilename("\u0000null.png"), "null.png");
});

test("story memory records locations, props, relationships, character state and timeline", () => {
  const { root, db, vault } = makeVault();
  try {
    const project = vault.createProject({ name: "Harbour Lights" });
    const episode = vault.createEpisode(project.id, { title: "First Light" });
    const scene = vault.createScene(project.id, episode.id, { title: "Arrival" });

    const location = vault.addLocation(project.id, { name: "Ferry terminal", description: "rain-slick concrete" });
    const prop = vault.addProp(project.id, { name: "brass compass", description: "Mira's heirloom" });
    vault.addRelationship(project.id, {
      fromCharacterId: "mira", toCharacterId: "jonah", kind: "sibling", description: "estranged",
    });
    const state = vault.setCharacterState(project.id, {
      characterId: "mira", sceneId: scene.id, outfit: "oilskin coat", appearanceNotes: "soaked hair",
    });
    const updatedState = vault.setCharacterState(project.id, {
      characterId: "mira", sceneId: scene.id, outfit: "dry wool coat",
    });
    assert.equal(updatedState.id, state.id);
    assert.equal(updatedState.outfit, "dry wool coat");

    vault.addTimelineEntry(project.id, { sceneId: scene.id, position: 1, label: "Mira arrives" });
    assert.equal(vault.listLocations(project.id)[0].id, location.id);
    assert.equal(vault.listProps(project.id)[0].id, prop.id);
    assert.equal(vault.listRelationships(project.id)[0].kind, "sibling");
    assert.equal(vault.listCharacterStates(project.id, { characterId: "mira" }).length, 1);
    assert.equal(vault.listTimeline(project.id)[0].label, "Mira arrives");
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("removeProject keeps media on disk and clears project records", () => {
  const { root, db, vault } = makeVault();
  try {
    const project = vault.createProject({ name: "Harbour Lights" });
    const asset = vault.addAsset({
      projectId: project.id,
      category: "images",
      name: "keep.png",
      data: Buffer.from("x").toString("base64"),
    });
    vault.createEpisode(project.id, { title: "First Light" });
    const result = vault.removeProject(project.id);
    assert.equal(result.filesKept, true);
    assert.ok(fs.existsSync(asset.filePath));
    assert.equal(vault.getProject(project.id), null);
    assert.equal(vault.listEpisodes(project.id).length, 0);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
