"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { DatabaseSync } = require("node:sqlite");
const { KaggleComputeManager } = require("../gateway/compute-manager");
const queue = require("../gateway/worker-queue");

function makeDb() {
  return new DatabaseSync(":memory:");
}

function makeSession(db, overrides = {}) {
  return queue.createSession(db, {
    id: overrides.id || "session-1",
    token: overrides.token || "a".repeat(64),
    kind: overrides.kind || "image",
    model: overrides.model || "qwen-image",
    remoteId: overrides.remoteId || "user/kernel",
    idleTimeoutSeconds: 900,
    maxRuntimeSeconds: 21600,
  });
}

function makeManager(db, { kaggle = {}, env = {}, vault = null } = {}) {
  const calls = { started: [], pumped: 0 };
  const provider = {
    enabled: true,
    startWorker: async ({ model, kind }) => {
      calls.started.push({ model, kind });
      return { id: `new-${calls.started.length}`, kind, model, status: "starting" };
    },
    pump: async () => { calls.pumped += 1; },
    getJob: () => null,
    ...kaggle,
  };
  const manager = new KaggleComputeManager({
    kaggle: provider,
    db,
    env: { KAGGLE_MAX_GPU_WORKERS: "1", KAGGLE_WORKER_STALE_SECONDS: "300", ...env },
    logger: { error() {}, warn() {} },
    vault,
  });
  return { manager, provider, calls };
}

test("probe reports capacity per kind without mutating sessions", () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "img", kind: "image" });
    const { manager } = makeManager(db);
    const probe = manager.probe();
    assert.equal(probe.enabled, true);
    assert.equal(probe.capacity.maxGpuWorkers, 1);
    assert.equal(probe.capacity.active, 1);
    assert.equal(probe.capacity.free, 0);
    assert.equal(probe.workers.length, 1);
    assert.equal(probe.workers[0].kind, "image");
    assert.deepEqual(probe.stale, []);
  } finally {
    db.close();
  }
});

test("status refreshes worker capacity after a previously cached probe", () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "img", kind: "image" });
    const { manager } = makeManager(db);
    manager.probe();

    queue.markStopped(db, "img", "stopped");
    const status = manager.status();

    assert.equal(status.capacity.active, 0);
    assert.equal(status.capacity.free, 1);
    assert.equal(status.available, true);
    assert.equal(status.workers[0].status, "failed");
  } finally {
    db.close();
  }
});

test("ensureWorker re-attaches to a live session instead of creating another", async () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "live", kind: "video", model: "wan-t2v-1.3b" });
    const { manager, calls } = makeManager(db);
    const result = await manager.ensureWorker({ kind: "video", model: "wan-t2v-1.3b" });
    assert.equal(result.created, false);
    assert.equal(result.session.id, "live");
    assert.equal(calls.started.length, 0);
  } finally {
    db.close();
  }
});

test("ensureWorker checks the model-store bridge before reusing a live worker", async () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "live", kind: "video", model: "wan-t2v-1.3b" });
    let bridgeChecks = 0;
    const { manager, calls } = makeManager(db, {
      kaggle: { prepareModelStoreBridge: async () => { bridgeChecks += 1; } },
    });

    const result = await manager.ensureWorker({ kind: "video", model: "wan-t2v-1.3b" });

    assert.equal(result.created, false);
    assert.equal(bridgeChecks, 1);
    assert.equal(calls.started.length, 0);
  } finally {
    db.close();
  }
});

test("ensureWorker counts a stopping worker against capacity without reusing it", async () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "stopping", kind: "image", model: "qwen-image" });
    queue.requestStop(db, "stopping");
    const { manager, calls } = makeManager(db);

    await assert.rejects(
      manager.ensureWorker({ kind: "image", model: "qwen-image" }),
      /slots are in use/,
    );
    assert.equal(calls.started.length, 0);
  } finally {
    db.close();
  }
});

test("ensureWorker creates the worker on demand and enforces the GPU slot cap", async () => {
  const db = makeDb();
  try {
    const { manager, calls } = makeManager(db);
    const first = await manager.ensureWorker({ kind: "video" });
    assert.equal(first.created, true);
    assert.deepEqual(calls.started[0], { model: undefined, kind: "video" });

    makeSession(db, { id: "existing", kind: "image", model: "qwen-image" });
    await assert.rejects(
      async () => manager.ensureWorker({ kind: "image", model: "another-checkpoint" }),
      /slots are in use/,
    );
  } finally {
    db.close();
  }
});

test("recover closes a stale session, fails its queued jobs and re-pumps", async () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "stale", kind: "video", model: "wan" });
    db.prepare("UPDATE worker_sessions SET status = 'busy', created_at = datetime('now', '-2 hours'), last_seen_at = datetime('now', '-1 hour') WHERE id = 'stale'").run();
    queue.enqueueJob(db, { id: "job-1", sessionId: "stale", kind: "video", model: "wan", requestJson: {} });
    const { manager, calls } = makeManager(db);

    const summary = await manager.recover();
    assert.deepEqual(summary.recoveredSessions, ["stale"]);
    assert.deepEqual(summary.failedJobs, ["job-1"]);
    assert.equal(calls.pumped, 1);
    assert.equal(queue.getSession(db, "stale").status, "failed");
    assert.equal(db.prepare("SELECT status FROM compute_jobs WHERE id = 'job-1'").get().status, "failed");
  } finally {
    db.close();
  }
});

test("recover leaves a fresh worker and its queued job untouched", async () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "fresh", kind: "video", model: "wan" });
    queue.enqueueJob(db, { id: "job-fresh", sessionId: "fresh", kind: "video", model: "wan", requestJson: {} });
    const { manager } = makeManager(db);
    const summary = await manager.recover();
    assert.deepEqual(summary.recoveredSessions, []);
    assert.equal(db.prepare("SELECT status FROM compute_jobs WHERE id = 'job-fresh'").get().status, "queued");
  } finally {
    db.close();
  }
});

test("recover fails jobs whose owning session was already closed", async () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "closed", kind: "image", model: "m" });
    queue.markStopped(db, "closed", "stopped earlier");
    queue.enqueueJob(db, { id: "orphan", sessionId: "closed", kind: "image", model: "m", requestJson: {} });
    const { manager } = makeManager(db);
    const summary = await manager.recover();
    assert.deepEqual(summary.staleJobSessions, ["closed"]);
    assert.deepEqual(summary.failedJobs, ["orphan"]);
  } finally {
    db.close();
  }
});

test("syncJob copies a completed video output into the vault idempotently", () => {
  const db = makeDb();
  try {
    const saved = [];
    const vault = {
      addAssetFromPath: (args) => {
        saved.push(args);
        return { id: `asset-${saved.length}`, ...args };
      },
    };
    const { manager } = makeManager(db, {
      vault,
      kaggle: {
        getJob: () => ({
          id: "job-9",
          provider: "kaggle-worker",
          kind: "video",
          model: "wan-t2v-1.3b",
          status: "completed",
          result: {
            videoPath: "C:/tmp/output.webm",
            mime: "video/webm",
            width: 480,
            height: 320,
            fps: 8,
            frames: 9,
            seed: 3,
          },
        }),
      },
    });
    const outcome = manager.syncJob("job-9", { projectId: "proj-1" });
    assert.equal(outcome.synced, true);
    assert.equal(saved.length, 1);
    assert.equal(saved[0].category, "videos");
    assert.equal(saved[0].sourceRef, "compute:job-9");
    assert.equal(saved[0].projectId, "proj-1");
    assert.equal(saved[0].metadata.computeJobId, "job-9");
  } finally {
    db.close();
  }
});

test("syncJob is a no-op for unfinished jobs and missing projects", () => {
  const db = makeDb();
  try {
    const vault = { addAssetFromPath: () => { throw new Error("should not be called"); } };
    const unfinished = makeManager(db, {
      vault,
      kaggle: { getJob: () => ({ id: "job-1", kind: "image", status: "running", result: null }) },
    });
    const outcome = unfinished.manager.syncJob("job-1", { projectId: "p" });
    assert.equal(outcome.synced, false);
    assert.equal(outcome.reason, "not-completed");

    const completed = makeManager(db, {
      vault,
      kaggle: {
        getJob: () => ({
          id: "job-2", kind: "image", status: "completed", result: { imagePath: "C:/tmp/a.png" },
        }),
      },
    });
    assert.equal(completed.manager.syncJob("job-2").reason, "no-project");
  } finally {
    db.close();
  }
});

test("queueSnapshot summarises queued work and the next job to run", () => {
  const db = makeDb();
  try {
    makeSession(db, { id: "s", kind: "video", model: "wan" });
    queue.enqueueJob(db, { id: "j-low", sessionId: "s", kind: "video", model: "wan", requestJson: {}, priority: 0 });
    queue.enqueueJob(db, { id: "j-high", sessionId: "s", kind: "video", model: "wan", requestJson: {}, priority: 9 });
    const { manager } = makeManager(db);
    const snapshot = manager.queueSnapshot();
    assert.equal(snapshot.oldestQueued.id, "j-high");
    assert.equal(snapshot.totals[0].count, 2);
    assert.equal(snapshot.capacity.active, 1);
  } finally {
    db.close();
  }
});
