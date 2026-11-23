"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const { CharacterStore } = require("../gateway/characters");

function makeStore() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "local-ai-characters-"));
  const db = new DatabaseSync(":memory:");
  const store = new CharacterStore({ db, dataDir: root, logger: { error() {} } });
  return { root, db, store };
}

test("creates, reads, updates and slugs characters scoped to a project", () => {
  const { root, db, store } = makeStore();
  try {
    const character = store.create({
      projectId: "proj-1",
      name: "Mira Voss",
      role: "protagonist",
      appearance: "silver hair, amber eyes",
      outfit: "navy flight jacket",
      stylePrompt: "cinematic portrait, soft rim light",
    });
    assert.equal(character.slug, "mira-voss");
    assert.equal(character.projectId, "proj-1");
    assert.equal(character.appearance, "silver hair, amber eyes");
    assert.equal(store.get(character.id).name, "Mira Voss");

    const updated = store.update(character.id, { outfit: "black trench coat", voiceProfile: "mira-voice" });
    assert.equal(updated.outfit, "black trench coat");
    assert.equal(updated.voiceProfile, "mira-voice");

    assert.throws(
      () => store.create({ projectId: "proj-1", name: "Mira Voss" }),
      /already exists/,
    );
    const otherProject = store.create({ projectId: "proj-2", name: "Mira Voss" });
    assert.notEqual(otherProject.id, character.id);

    const projectList = store.list({ projectId: "proj-1" });
    assert.equal(projectList.length, 1);
    const withGlobal = store.create({ projectId: "global", name: "Narrator" });
    assert.equal(store.list({ projectId: "proj-1" }).length, 2);
    assert.equal(store.list({ projectId: "proj-1", includeGlobal: false }).length, 1);
    assert.equal(store.get(withGlobal.id).projectId, "global");
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("stores reference images on disk, keeps one primary per kind, and cleans up", () => {
  const { root, db, store } = makeStore();
  try {
    const character = store.create({ projectId: "proj-1", name: "Mira" });
    const first = store.addReference(character.id, {
      kind: "face",
      label: "front",
      data: Buffer.from("face-one").toString("base64"),
      mime: "image/png",
      isPrimary: true,
    });
    const second = store.addReference(character.id, {
      kind: "face",
      label: "profile",
      data: Buffer.from("face-two").toString("base64"),
      mime: "image/png",
      isPrimary: true,
    });
    assert.ok(fs.existsSync(first.filePath));
    assert.ok(first.filePath.startsWith(path.join(root, "characters", character.id)));
    const faces = store.listReferences(character.id, { kind: "face" });
    assert.equal(faces.length, 2);
    assert.equal(faces.filter((reference) => reference.isPrimary).length, 1);
    assert.equal(faces.find((reference) => reference.isPrimary).id, second.id);

    const prompt = store.describeForPrompt(character.id);
    assert.match(prompt, /Mira/);
    assert.match(prompt, /face: profile/);

    const resolved = store.referencesForPrompt(character.id, { limit: 1 });
    assert.equal(resolved.length, 1);
    assert.equal(resolved[0].id, second.id);

    store.removeReference(character.id, second.id);
    assert.equal(fs.existsSync(second.filePath), false);
    assert.equal(store.listReferences(character.id).length, 1);

    store.remove(character.id);
    assert.equal(store.get(character.id), null);
    assert.equal(fs.existsSync(first.filePath), false);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("reclassifies references and keeps a single primary face per character", () => {
  const { root, db, store } = makeStore();
  try {
    const character = store.create({ projectId: "proj-1", name: "Mira" });
    const primaryFace = store.addReference(character.id, {
      kind: "face",
      label: "portrait.png",
      data: Buffer.from("face").toString("base64"),
      mime: "image/png",
      isPrimary: true,
    });
    const sheet = store.addReference(character.id, {
      kind: "face",
      label: "angles.png",
      data: Buffer.from("sheet").toString("base64"),
      mime: "image/png",
    });
    const body = store.addReference(character.id, {
      kind: "face",
      label: "standing.png",
      data: Buffer.from("body").toString("base64"),
      mime: "image/png",
    });
    const secondFace = store.addReference(character.id, {
      kind: "face",
      label: "face-2.png",
      data: Buffer.from("face-2").toString("base64"),
      mime: "image/png",
    });

    store.updateReference(character.id, sheet.id, { kind: "character_sheet", isPrimary: false });
    store.updateReference(character.id, body.id, { kind: "full_body", isPrimary: false });
    store.updateReference(character.id, secondFace.id, { kind: "face", isPrimary: true });

    const references = store.listReferences(character.id);
    assert.equal(references.find((reference) => reference.id === sheet.id).kind, "character_sheet");
    assert.equal(references.find((reference) => reference.id === body.id).kind, "full_body");
    assert.equal(references.find((reference) => reference.id === primaryFace.id).isPrimary, false);
    assert.equal(references.find((reference) => reference.id === secondFace.id).isPrimary, true);
    assert.equal(references.find((reference) => reference.id === sheet.id).filePath, sheet.filePath);
    assert.throws(
      () => store.updateReference(character.id, sheet.id, { kind: "unknown", isPrimary: false }),
      /kind must be one of/,
    );
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("rejects invalid reference payloads and unknown characters", () => {
  const { root, db, store } = makeStore();
  try {
    const character = store.create({ projectId: "proj-1", name: "Mira" });
    assert.throws(() => store.addReference("missing", { data: "AAAA" }), /not found/);
    assert.throws(() => store.addReference(character.id, {}), /filePath or base64 data/);
    assert.throws(() => store.addReference(character.id, { data: "not base64!!" }), /base64/);
    assert.throws(() => store.create({ projectId: "proj-1", name: "   " }), /name/);
    assert.throws(() => store.update(character.id, { name: "" }), /name/);
    assert.equal(store.describeForPrompt("missing"), "");
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("describeForPrompt is stable and capped", () => {
  const { root, db, store } = makeStore();
  try {
    const character = store.create({
      projectId: "proj-1",
      name: "Mira",
      role: "lead",
      appearance: "a".repeat(300),
      outfit: "b".repeat(300),
      personality: "c".repeat(300),
      stylePrompt: "d".repeat(300),
    });
    const first = store.describeForPrompt(character.id, { maxChars: 80 });
    const second = store.describeForPrompt(character.id, { maxChars: 80 });
    assert.equal(first, second);
    assert.ok(first.length <= 80);
    assert.match(first, /^Mira/);
  } finally {
    db.close();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
