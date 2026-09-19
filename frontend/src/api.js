// Every call the UI makes. In development Vite proxies /api to FastAPI.

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).detail || "";
    } catch (error) {
      detail = "";
    }
    throw new Error(detail || path + " failed with " + response.status);
  }
  return response.json();
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).detail || "";
    } catch (error) {
      detail = "";
    }
    throw new Error(detail || "request failed with " + response.status);
  }
  return response.json();
}

function withCorpus(path, corpusId) {
  if (!corpusId || corpusId === "demo") {
    return path;
  }
  const joiner = path.includes("?") ? "&" : "?";
  return path + joiner + "corpus_id=" + encodeURIComponent(corpusId);
}

export function getHealth() {
  return getJson("/api/health");
}

export function getCorpora() {
  return getJson("/api/corpora");
}

export function getTree(corpusId) {
  return getJson(withCorpus("/api/tree", corpusId));
}

export function getNode(nodeId, corpusId) {
  return getJson(withCorpus("/api/node/" + encodeURIComponent(nodeId), corpusId));
}

export function getEvalResults() {
  return getJson("/api/eval");
}

export function getJob(jobId) {
  return getJson("/api/job/" + encodeURIComponent(jobId));
}

export function compareModes(payload) {
  return postJson("/api/compare", payload);
}

// Sends the PDFs and returns { corpus_id, job_id, ... }. The build runs in the
// background; poll getJob until its state is "done".
export async function uploadCorpus(fileList, name, fast) {
  const form = new FormData();
  for (const file of fileList) {
    form.append("files", file);
  }
  form.append("name", name || "");
  form.append("fast", fast ? "true" : "false");

  const response = await fetch("/api/corpus/upload", { method: "POST", body: form });
  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).detail || "";
    } catch (error) {
      detail = "";
    }
    throw new Error(detail || "upload failed with " + response.status);
  }
  return response.json();
}

// Shared reader for both streaming endpoints.
async function readEventStream(path, payload, onEvent) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || response.body === null) {
    let detail = "";
    try {
      detail = (await response.json()).detail || "";
    } catch (error) {
      detail = "";
    }
    throw new Error(detail || "stream failed with " + response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const result = await reader.read();
    if (result.done) {
      break;
    }
    buffer = buffer + decoder.decode(result.value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const raw of events) {
      const line = raw.trim();
      if (!line.startsWith("data:")) {
        continue;
      }
      const payloadText = line.slice(5).trim();
      if (payloadText === "[DONE]") {
        return;
      }
      let parsed;
      try {
        parsed = JSON.parse(payloadText);
      } catch (error) {
        continue;
      }
      onEvent(parsed);
    }
  }
}

export function askStream(payload, handlers) {
  return readEventStream("/api/ask/stream", payload, function (parsed) {
    if (parsed.event === "meta" && handlers.onMeta) {
      handlers.onMeta(parsed);
    } else if (parsed.event === "text" && handlers.onText) {
      handlers.onText(parsed.text);
    } else if (parsed.event === "done" && handlers.onDone) {
      handlers.onDone(parsed);
    }
  });
}

// Both retrievers at once. Passages arrive first (milliseconds), then the two
// answers stream in one after the other.
export function compareStream(payload, handlers) {
  return readEventStream("/api/compare/stream", payload, function (parsed) {
    if (parsed.event === "meta" && handlers.onSideMeta) {
      handlers.onSideMeta(parsed.side, parsed);
    } else if (parsed.event === "retrieval_done" && handlers.onRetrievalDone) {
      handlers.onRetrievalDone(parsed);
    } else if (parsed.event === "text" && handlers.onSideText) {
      handlers.onSideText(parsed.side, parsed.text);
    } else if (parsed.event === "side_done" && handlers.onSideDone) {
      handlers.onSideDone(parsed.side, parsed);
    }
  });
}
