// Every call the UI makes. In development Vite proxies /api to FastAPI.

async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(path + " failed with " + response.status);
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
    const detail = await response.text();
    throw new Error(detail || "request failed");
  }
  return response.json();
}

export function getHealth() {
  return getJson("/api/health");
}

export function getTree() {
  return getJson("/api/tree");
}

export function getNode(nodeId) {
  return getJson("/api/node/" + encodeURIComponent(nodeId));
}

export function getEvalResults() {
  return getJson("/api/eval");
}

export function compareModes(payload) {
  return postJson("/api/compare", payload);
}

export async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body: form });
  if (!response.ok) {
    throw new Error("upload failed");
  }
  return response.json();
}

// Reads the server-sent event stream from /api/ask/stream and calls the
// handlers as things arrive: the retrieval first, then the answer text.
export async function askStream(payload, handlers) {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || response.body === null) {
    throw new Error("stream failed with " + response.status);
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
      if (parsed.event === "meta" && handlers.onMeta) {
        handlers.onMeta(parsed);
      } else if (parsed.event === "text" && handlers.onText) {
        handlers.onText(parsed.text);
      } else if (parsed.event === "done" && handlers.onDone) {
        handlers.onDone(parsed);
      }
    }
  }
}
