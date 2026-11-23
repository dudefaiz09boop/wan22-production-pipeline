"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { parseKaggleDatasetList } = require("../scripts/kaggle-dataset-list");

test("Kaggle dataset list parser accepts the empty-list message and JSON arrays", () => {
  assert.deepEqual(parseKaggleDatasetList("No datasets found\r\n"), []);
  assert.deepEqual(parseKaggleDatasetList('[{"ref":"owner/cache"}]'), [
    { ref: "owner/cache" },
  ]);
});

test("Kaggle dataset list parser rejects unexpected CLI output", () => {
  assert.throws(() => parseKaggleDatasetList("Kaggle API unavailable"), /Could not parse/);
  assert.throws(() => parseKaggleDatasetList('{"ref":"owner/cache"}'), /must be a JSON array/);
});
