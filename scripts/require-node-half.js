#!/usr/bin/env node
"use strict";

// The Node/Studio half of this project is not part of this checkout: there is
// no gateway/ directory and none of the Node helper scripts under scripts/.
// Without this guard, `npm test` fails with a bare
// "Cannot find module '../gateway/ai-horde'" from whichever test file loads
// first, which reads like a broken test rather than a missing subtree.
//
// Fail once, up front, naming every missing path, so the cause is obvious.

const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");

const required = [
  "gateway/server.js",
  "gateway/model-store-server.js",
  "scripts/test-film-studio-playwright.js",
  "scripts/cache-krea2-image-model.js",
  "scripts/cache-wan22-video-models.js",
  "scripts/kaggle-dataset-list.js",
  "scripts/fetch-wan22-q8.js",
];

const missing = required.filter((rel) => !fs.existsSync(path.join(root, rel)));

if (missing.length === 0) {
  process.exit(0);
}

console.error(
  [
    "The Node/Studio half of this project is missing from this checkout.",
    "",
    "Missing paths:",
    ...missing.map((rel) => `  ${rel}`),
    "",
    "The JavaScript tests, the `check` script and the Node helper scripts",
    "cannot run without them. The Python pipeline is unaffected.",
    "",
    "Use the Python entry points instead:",
    "  npm run test:python",
    "  npm run validate",
    "  npm run check:credentials",
    "",
    "Or restore the gateway/ subtree and the Node scripts, then re-run.",
  ].join("\n"),
);
process.exit(1);
