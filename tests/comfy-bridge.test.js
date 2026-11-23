const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const {
  createKaggleComfyBridge,
  nodeInfo,
  parseImageWorkflow,
  parseVideoWorkflow,
} = require("../gateway/comfy-bridge");

const workflow = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "comfy", "kaggle-image-api-workflow.json"), "utf8"));
const videoWorkflow = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "comfy", "kaggle-video-api-workflow.json"), "utf8"));

test("registered Kaggle image workflow maps to one bounded image request", () => {
  assert.deepEqual(parseImageWorkflow(workflow), {
    generatorId: "1",
    saveNodeId: "2",
    prompt: "a small cabin beside a lake, soft morning light",
    negativePrompt: "blurry, low quality, text, watermark",
    steps: 20,
    seed: 123456,
  });
  assert.deepEqual(Object.keys(nodeInfo()), ["KaggleImageGenerate", "SaveImage", "KaggleVideoGenerate", "SaveVideo"]);
  assert.throws(() => createKaggleComfyBridge({ kaggle: {}, host: "0.0.0.0" }), /loopback address/);
});

test("registered Wan video workflow maps to a short bounded video task", () => {
  assert.deepEqual(parseVideoWorkflow(videoWorkflow), {
    generatorId: "1",
    saveNodeId: "2",
    prompt: "A cinematic wide shot of a tiny explorer crossing a misty alien desert at sunrise",
    negativePrompt: "text, subtitles, watermark, flicker, distorted motion",
    videoLength: 2,
    fps: 8,
    steps: 8,
    seed: 123456,
  });
  assert.throws(() => parseVideoWorkflow({
    ...videoWorkflow,
    "3": { class_type: "UnsupportedNode", inputs: {} },
  }), /registered KaggleVideoGenerate/);
  assert.throws(() => parseVideoWorkflow({
    ...videoWorkflow,
    "1": { ...videoWorkflow["1"], inputs: { ...videoWorkflow["1"].inputs, video_length: 60 } },
  }), /video_length/);
});

test("Kaggle ComfyUI bridge submits the workflow and serves completed image history", async () => {
  const imagePath = path.join(os.tmpdir(), `local-ai-comfy-${Date.now()}.png`);
  const id = "32b0a64b-82fe-409b-ae1a-44d98ed0cdd0";
  fs.writeFileSync(imagePath, Buffer.from("png-test"));
  let task = { id, kind: "image", status: "queued", result: null };
  let submitted;
  const kaggle = {
    async submitImage(request) {
      submitted = request;
      return { id, status: "queued" };
    },
    getJob(taskId) {
      return taskId === id ? task : null;
    },
    listJobs() {
      return [task];
    },
  };
  const bridge = createKaggleComfyBridge({
    kaggle,
    port: 0,
    logger: { log() {} },
  });

  test("Kaggle ComfyUI bridge submits and serves a generated WEBM video", async () => {
    const videoPath = path.join(os.tmpdir(), `local-ai-comfy-${Date.now()}.webm`);
    const id = "32b0a64b-82fe-409b-ae1a-44d98ed0cdd1";
    fs.writeFileSync(videoPath, Buffer.from("webm-test"));
    let task = { id, kind: "video", status: "queued", result: null };
    let submitted;
    const kaggle = {
      async submitVideo(request) {
        submitted = request;
        return { id, status: "queued" };
      },
      getJob(taskId) {
        return taskId === id ? task : null;
      },
      listJobs() {
        return [task];
      },
    };
    const bridge = createKaggleComfyBridge({
      kaggle,
      port: 0,
      logger: { log() {} },
    });
    try {
      await bridge.ready;
      const baseUrl = `http://127.0.0.1:${bridge.server.address().port}`;
      const submission = await fetch(`${baseUrl}/prompt`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: videoWorkflow, client_id: "maestro-video-test" }),
      });
      assert.equal(submission.status, 200);
      assert.deepEqual(submitted, {
        prompt: videoWorkflow["1"].inputs.prompt,
        negativePrompt: videoWorkflow["1"].inputs.negative_prompt,
        videoLength: 2,
        fps: 8,
        steps: 8,
        seed: 123456,
      });
      const queued = await fetch(`${baseUrl}/history/${id}`).then((response) => response.json());
      assert.equal(queued[id].status.status_str, "in_progress");
      task = { ...task, status: "completed", result: { videoPath } };
      const completed = await fetch(`${baseUrl}/history/${id}`).then((response) => response.json());
      assert.deepEqual(completed[id].outputs["2"].videos, [{
        filename: "maestro-video.webm",
        subfolder: id,
        type: "output",
      }]);
      const videoResponse = await fetch(`${baseUrl}/view?filename=maestro-video.webm&subfolder=${id}&type=output`);
      assert.equal(videoResponse.status, 200);
      assert.equal(videoResponse.headers.get("content-type"), "video/webm");
      assert.equal(await videoResponse.text(), "webm-test");
    } finally {
      await bridge.stop();
      fs.rmSync(videoPath, { force: true });
    }
  });
  try {
    await bridge.ready;
    const baseUrl = `http://127.0.0.1:${bridge.server.address().port}`;
    const submission = await fetch(`${baseUrl}/prompt`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt: workflow, client_id: "maestro-test-client" }),
    });
    assert.equal(submission.status, 200);
    assert.deepEqual(submitted, {
      prompt: workflow["1"].inputs.prompt,
      negativePrompt: workflow["1"].inputs.negative_prompt,
      steps: workflow["1"].inputs.steps,
      seed: workflow["1"].inputs.seed,
    });
    const queued = await fetch(`${baseUrl}/history/${id}`).then((response) => response.json());
    assert.equal(queued[id].status.status_str, "in_progress");
    task = { ...task, status: "completed", result: { imagePath } };
    const completed = await fetch(`${baseUrl}/history/${id}`).then((response) => response.json());
    assert.deepEqual(completed[id].outputs["2"].images, [{
      filename: "unicorn-dragon.png",
      subfolder: id,
      type: "output",
    }]);
    const imageResponse = await fetch(`${baseUrl}/view?filename=unicorn-dragon.png&subfolder=${id}&type=output`);
    assert.equal(imageResponse.status, 200);
    assert.equal(await imageResponse.text(), "png-test");
    const invalid = await fetch(`${baseUrl}/prompt`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt: { ...workflow, "3": { class_type: "UnsupportedNode", inputs: {} } } }),
    });
    assert.equal(invalid.status, 400);
  } finally {
    await bridge.stop();
    fs.rmSync(imagePath, { force: true });
  }
});
