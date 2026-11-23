const assert = require("node:assert/strict");
const test = require("node:test");
const {
  parseCivitaiReference,
  fetchModelMetadata,
  searchCivitaiLoras,
  classifyType,
  sanitizeFilename,
} = require("../gateway/civitai");
const {
  assessLoraCompatibility,
  buildLoraPayload,
  compareBaseModels,
  compatibleImportedLoras,
  isGenericImageBaseModel,
} = require("../gateway/public/civitai-compatibility");

test("parseCivitaiReference accepts a numeric model ID", () => {
  assert.deepEqual(parseCivitaiReference("667086"), { modelId: 667086, versionId: null });
});

test("parseCivitaiReference extracts the model ID from a Civitai URL", () => {
  assert.deepEqual(
    parseCivitaiReference("https://civitai.com/models/667086/nsfw-master"),
    { modelId: 667086, versionId: null },
  );
});

test("parseCivitaiReference extracts an explicit modelVersionId", () => {
  assert.deepEqual(
    parseCivitaiReference("https://civitai.red/models/3031081/ultra-realistic?modelVersionId=3446495"),
    { modelId: 3031081, versionId: 3446495 },
  );
});

test("Civitai LoRA compatibility matches known base families and flags Wan task mismatches", () => {
  assert.equal(compareBaseModels("SDXL 1.0", "SDXL"), "match");
  assert.equal(compareBaseModels("Wan Video 2.2 I2V-A14B", "Wan Video 2.2 I2V-A14B"), "match");
  assert.equal(compareBaseModels("Wan Video 2.2 T2V-A14B", "Wan Video 2.2 I2V-A14B"), "review");
  assert.equal(compareBaseModels("Wan Video 2.1 T2V", "Wan Video 2.2 I2V-A14B"), "review");
  assert.equal(compareBaseModels("Wan Video", "Wan Video 2.2 I2V-A14B"), "review");

  const result = assessLoraCompatibility(
    { baseModel: "Wan Video 2.2 T2V-A14B" },
    [{ kind: "checkpoint", modelName: "WAN I2V", filename: "wan.gguf", status: "complete", baseModel: "Wan Video 2.2 I2V-A14B" }],
  );
  assert.equal(result.status, "review");
  assert.equal(result.review[0].modelName, "WAN I2V");
});

test("Civitai LoRA compatibility reports missing or unknown checkpoint metadata", () => {
  assert.equal(
    assessLoraCompatibility({ baseModel: "SDXL 1.0" }, []).status,
    "no-checkpoints",
  );
  assert.equal(
    assessLoraCompatibility({ baseModel: "unknown" }, [{ kind: "checkpoint", baseModel: "SDXL" }]).status,
    "unknown",
  );
  assert.equal(
    assessLoraCompatibility(
      { baseModel: "SD 1.5" },
      [{ kind: "checkpoint", baseModel: "SDXL" }],
    ).status,
    "unmatched",
  );
});

test("generic image LoRAs require a completed exact SD-family base-model match", () => {
  const loras = [
    { kind: "lora", status: "complete", filename: "sdxl.safetensors", baseModel: "SDXL 1.0" },
    { kind: "lora", status: "downloading", filename: "pending.safetensors", baseModel: "SDXL 1.0" },
    { kind: "lora", status: "complete", filename: "sd15.safetensors", baseModel: "SD 1.5" },
    { kind: "lora", status: "complete", filename: "wan.safetensors", baseModel: "Wan Video 2.2 I2V-A14B" },
  ];
  assert.equal(isGenericImageBaseModel("Pony"), true);
  assert.equal(isGenericImageBaseModel("Wan Video 2.2 I2V-A14B"), false);
  assert.deepEqual(
    compatibleImportedLoras(loras, "SDXL"),
    [loras[0]],
  );
  assert.deepEqual(compatibleImportedLoras(loras, "Wan Video 2.2 I2V-A14B"), []);
});

test("Civitai LoRA payload maps one bounded strength across up to four selections", () => {
  assert.deepEqual(buildLoraPayload(["a.safetensors", "b.safetensors"], 0.7), {
    loras: ["a.safetensors", "b.safetensors"],
    loraStrengths: [0.7, 0.7],
  });
  assert.deepEqual(buildLoraPayload([], "invalid"), { loras: [], loraStrengths: [] });
  assert.throws(() => buildLoraPayload(["a", "a"], 1), /only be selected once/);
  assert.throws(() => buildLoraPayload(["a", "b", "c", "d", "e"], 1), /no more than 4/);
  assert.throws(() => buildLoraPayload(["a"], 2.1), /between 0 and 2/);
});

test("parseCivitaiReference rejects unparseable input", () => {
  assert.throws(() => parseCivitaiReference(""), /required/);
  assert.throws(() => parseCivitaiReference("not a url"), /Could not parse/);
  assert.throws(() => parseCivitaiReference("https://civitai.com/images/123"), /models\/<id>/);
});

test("classifyType accepts Checkpoint and LoRA and rejects everything else", () => {
  assert.equal(classifyType("Checkpoint"), "checkpoint");
  assert.equal(classifyType("LORA"), "lora");
  assert.equal(classifyType("LoCon"), "lora");
  assert.throws(() => classifyType("TextualInversion"), (error) => {
    assert.equal(error.status, 400);
    return true;
  });
});

function fakeCivitaiFetch(model) {
  return async () => ({
    ok: true,
    status: 200,
    json: async () => model,
  });
}

test("fetchModelMetadata resolves the primary file of the requested (or latest) version", async () => {
  const model = {
    name: "Ultra Realistic by Stable Yogi Krea2",
    type: "Checkpoint",
    modelVersions: [
      {
        id: 3446495,
        name: "v1.0 INT8 Extended",
        baseModel: "Krea 2",
        files: [
          { id: 3345679, name: "krea2.safetensors", sizeKB: 13175734.6328125, primary: true, downloadUrl: "https://civitai.com/api/download/models/3446495?fileId=3345679" },
        ],
      },
    ],
  };
  const metadata = await fetchModelMetadata({ modelId: 3031081, fetchImpl: fakeCivitaiFetch(model) });
  assert.equal(metadata.kind, "checkpoint");
  assert.equal(metadata.baseModel, "Krea 2");
  assert.equal(metadata.filename, "krea2.safetensors");
  assert.equal(metadata.sizeBytes, Math.round(13175734.6328125 * 1024));
  assert.equal(metadata.versionId, 3446495);
});

test("searchCivitaiLoras uses the server-side token, model filters, and cursor pagination", async () => {
  let requestUrl = "";
  let requestHeaders = {};
  const result = await searchCivitaiLoras({
    query: "Wan lighting",
    baseModel: "Wan Video 2.2 I2V-A14B",
    cursor: "opaque|cursor",
    sort: "Newest",
    apiKey: "test-civitai-key",
    fetchImpl: async (url, options) => {
      requestUrl = url;
      requestHeaders = options.headers;
      return {
        ok: true,
        json: async () => ({
          items: [{
            id: 33,
            name: "Wan lighting LoRA",
            creator: { username: "creator" },
            stats: { downloadCount: 123 },
            modelVersions: [{ id: 44, name: "v2", baseModel: "Wan Video 2.2 I2V-A14B" }],
          }],
          metadata: { nextCursor: "next|cursor" },
        }),
      };
    },
  });

  const url = new URL(requestUrl);
  assert.equal(url.pathname, "/api/v1/models");
  assert.equal(url.searchParams.get("types"), "LORA");
  assert.equal(url.searchParams.get("query"), "Wan lighting");
  assert.equal(url.searchParams.get("baseModels"), "Wan Video 2.2 I2V-A14B");
  assert.equal(url.searchParams.get("cursor"), "opaque|cursor");
  assert.equal(url.searchParams.get("page"), null);
  assert.equal(requestHeaders.Authorization, "Bearer test-civitai-key");
  assert.equal(result.items[0].url, "https://civitai.com/models/33?modelVersionId=44");
  assert.equal(result.items[0].downloads, 123);
  assert.equal(result.nextCursor, "next|cursor");
});

test("searchCivitaiLoras requires a configured key and validates API paging", async () => {
  await assert.rejects(searchCivitaiLoras(), (error) => {
    assert.equal(error.status, 503);
    return true;
  });
  await assert.rejects(searchCivitaiLoras({ apiKey: "key", limit: 51 }), (error) => {
    assert.equal(error.status, 400);
    return true;
  });
});

test("Civitai metadata accepts GGUF checkpoints but rejects unsupported GGUF LoRAs", async () => {
  const checkpoint = {
    name: "Wan 2.2 GGUF",
    type: "Checkpoint",
    modelVersions: [{
      id: 2,
      baseModel: "Wan Video",
      files: [{ id: 3, name: "wan-high.gguf", sizeKB: 1024, primary: true, downloadUrl: "https://civitai.example/model" }],
    }],
  };
  const metadata = await fetchModelMetadata({
    modelId: 1,
    versionId: 2,
    fetchImpl: fakeCivitaiFetch(checkpoint),
  });
  assert.equal(metadata.filename, "wan-high.gguf");

  const lora = {
    ...checkpoint,
    type: "LORA",
  };
  await assert.rejects(
    fetchModelMetadata({ modelId: 1, versionId: 2, fetchImpl: fakeCivitaiFetch(lora) }),
    /GGUF LoRA files are not supported/,
  );
});

test("fetchModelMetadata rejects a model with no files", async () => {
  const model = { name: "empty", type: "Checkpoint", modelVersions: [{ id: 1, files: [] }] };
  await assert.rejects(
    fetchModelMetadata({ modelId: 1, fetchImpl: fakeCivitaiFetch(model) }),
    /no downloadable files/,
  );
});

test("fetchModelMetadata surfaces a 404 for an unknown model", async () => {
  await assert.rejects(
    fetchModelMetadata({
      modelId: 999999,
      fetchImpl: async () => ({ ok: false, status: 404 }),
    }),
    (error) => {
      assert.equal(error.status, 404);
      return true;
    },
  );
});

test("sanitizeFilename passes through a safe basename", () => {
  assert.equal(sanitizeFilename("model.safetensors"), "model.safetensors");
  assert.equal(sanitizeFilename("lora_v1.ckpt"), "lora_v1.ckpt");
  assert.equal(sanitizeFilename("weights.pth"), "weights.pth");
});

test("sanitizeFilename strips Unix directory traversal", () => {
  assert.equal(sanitizeFilename("../../../etc/passwd.safetensors"), "passwd.safetensors");
  assert.equal(sanitizeFilename("a/b/../../model.ckpt"), "model.ckpt");
  assert.equal(sanitizeFilename("./models/./deep/file.safetensors"), "file.safetensors");
});

test("sanitizeFilename strips Windows directory traversal", () => {
  assert.equal(sanitizeFilename("..\\..\\windows\\evil.safetensors"), "evil.safetensors");
  assert.equal(sanitizeFilename("a\\b\\..\\model.pt"), "model.pt");
});

test("sanitizeFilename removes null bytes", () => {
  assert.equal(sanitizeFilename("model\0.safetensors"), "model.safetensors");
});

test("sanitizeFilename rejects dot and dot-dot", () => {
  assert.throws(() => sanitizeFilename("."), /Unsafe Civitai filename/);
  assert.throws(() => sanitizeFilename(".."), /Unsafe Civitai filename/);
  assert.throws(() => sanitizeFilename(""), /Unsafe Civitai filename/);
});

test("sanitizeFilename rejects unsafe extensions", () => {
  assert.throws(() => sanitizeFilename("evil.exe"), /Unsupported Civitai file extension/);
  assert.throws(() => sanitizeFilename("script.js"), /Unsupported Civitai file extension/);
  assert.throws(() => sanitizeFilename("noextension"), /Unsupported Civitai file extension/);
});

test("sanitizeFilename accepts all allowed model extensions", () => {
  for (const ext of [".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"]) {
    assert.equal(sanitizeFilename(`model${ext}`), `model${ext}`);
  }
});

test("fetchModelMetadata neutralizes path traversal by stripping to basename", async () => {
  const model = {
    name: "Traversal",
    type: "Checkpoint",
    modelVersions: [
      {
        id: 1,
        baseModel: "SD 1.5",
        files: [
          {
            id: 99,
            name: "../../../windows/evil.safetensors",
            sizeKB: 1024,
            primary: true,
            downloadUrl: "https://civitai.com/api/download/models/1?fileId=99",
          },
        ],
      },
    ],
  };
  const metadata = await fetchModelMetadata({ modelId: 1, fetchImpl: fakeCivitaiFetch(model) });
  assert.equal(metadata.filename, "evil.safetensors");
});

test("fetchModelMetadata rejects a traversal path whose basename has an unsafe extension", async () => {
  const model = {
    name: "Traversal",
    type: "Checkpoint",
    modelVersions: [
      {
        id: 1,
        baseModel: "SD 1.5",
        files: [
          {
            id: 99,
            name: "../../../windows/evil.exe",
            sizeKB: 1024,
            primary: true,
            downloadUrl: "https://civitai.com/api/download/models/1?fileId=99",
          },
        ],
      },
    ],
  };
  await assert.rejects(
    fetchModelMetadata({ modelId: 1, fetchImpl: fakeCivitaiFetch(model) }),
    /Unsupported Civitai file extension/,
  );
});

test("fetchModelMetadata rejects a Civitai file with an unsafe extension", async () => {
  const model = {
    name: "BadExt",
    type: "Checkpoint",
    modelVersions: [
      {
        id: 1,
        baseModel: "SD 1.5",
        files: [
          {
            id: 99,
            name: "evil.exe",
            sizeKB: 1024,
            primary: true,
            downloadUrl: "https://civitai.com/api/download/models/1?fileId=99",
          },
        ],
      },
    ],
  };
  await assert.rejects(
    fetchModelMetadata({ modelId: 1, fetchImpl: fakeCivitaiFetch(model) }),
    /Unsupported Civitai file extension/,
  );
});