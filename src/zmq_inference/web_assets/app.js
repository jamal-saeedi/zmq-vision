"use strict";

const el = (id) => document.getElementById(id);
const form = el("request");
const imageInput = el("image");
const taskInputs = [...form.querySelectorAll('input[name="operation"]')];
const modes = {
  infer: "Run classification",
  generate: "Generate text",
  describe: "Ask about image",
};
const taskNames = {infer: "Classify", generate: "Generate", describe: "Ask image"};
const history = [];
let imageUrl = "";
let serverReady = false;
let requestPending = false;

function updateMode() {
  const operation = form.elements.operation.value;
  el("image-fields").hidden = operation === "generate";
  el("text-fields").hidden = operation === "infer";
  el("prompt").required = operation !== "infer";
  el("submit").textContent = modes[operation];
  el("submit").disabled =
    !serverReady || requestPending || !taskInputs.some((input) => input.checked && !input.disabled);
}

function updateConnection(ready, operations = null) {
  serverReady = ready;
  const connection = el("connection");
  connection.dataset.ready = String(ready);
  connection.textContent = ready ? "ZeroMQ ready" : "Server unavailable";
  el("task-note").hidden = ready;
  el("vlm-setup").hidden = ready && operations !== null && operations.includes("describe");
  for (const input of taskInputs) {
    input.disabled = operations !== null && !operations.includes(input.value);
  }
  if (taskInputs.some((input) => input.checked && input.disabled)) {
    const next = taskInputs.find((input) => input.value === "describe" && !input.disabled)
      || taskInputs.find((input) => !input.disabled);
    if (next) next.checked = true;
  }
  updateMode();
}

function updateImage(dataUrl, label) {
  imageUrl = dataUrl;
  const preview = el("preview");
  preview.hidden = !dataUrl;
  if (dataUrl) {
    preview.src = dataUrl;
    preview.alt = label;
  } else {
    preview.removeAttribute("src");
  }
  el("image-state").textContent = label;
}

function setStatus(text, error = false) {
  el("status").textContent = text;
  el("status").dataset.error = String(error);
}

function clearResponse() {
  for (const id of ["answer", "predictions", "metrics", "details"]) {
    el(id).hidden = true;
  }
  el("predictions").replaceChildren();
  el("metrics").replaceChildren();
}

function addMetric(label, value, identifier = false) {
  const entry = document.createElement("div");
  const term = document.createElement("dt");
  const definition = document.createElement("dd");
  term.textContent = label;
  definition.textContent = value;
  if (identifier) entry.className = "request-id";
  entry.append(term, definition);
  el("metrics").append(entry);
}

function bar(value, max, label) {
  const meter = document.createElement("meter");
  meter.max = max;
  meter.value = value;
  meter.setAttribute("aria-label", label);
  return meter;
}

function addHistory(operation, elapsed) {
  history.unshift({operation, elapsed});
  history.splice(8);
  const max = Math.max(1, ...history.map((run) => run.elapsed));
  const list = el("history");
  list.replaceChildren();
  for (const run of history) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const time = document.createElement("span");
    name.textContent = taskNames[run.operation];
    time.textContent = `${run.elapsed.toFixed(1)} ms`;
    item.append(name, bar(run.elapsed, max, `${name.textContent}: ${time.textContent}`), time);
    list.append(item);
  }
  el("history-empty").hidden = true;
  list.hidden = false;
}

function showResponse(data, operation) {
  addMetric("Server time", `${data.elapsed_ms.toFixed(1)} ms`);
  if (data.generation) {
    el("answer").textContent = data.generation.text || "The model returned an empty answer.";
    el("answer").hidden = false;
    addMetric("Prompt tokens", data.generation.prompt_tokens);
    addMetric("Output tokens", data.generation.completion_tokens);
    addMetric("Finish reason", data.generation.finish_reason);
  }
  if (data.predictions?.length) {
    for (const prediction of data.predictions) {
      const item = document.createElement("li");
      const label = document.createElement("div");
      const name = document.createElement("span");
      const confidence = document.createElement("span");
      label.className = "bar-label";
      name.textContent = prediction.label;
      confidence.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
      label.append(name, confidence);
      item.append(label, bar(prediction.confidence, 1, `${prediction.label}: ${confidence.textContent}`));
      el("predictions").append(item);
    }
    el("predictions").hidden = false;
  }
  addMetric("Request ID", data.request_id, true);
  el("metrics").hidden = false;
  el("raw").textContent = JSON.stringify(data, null, 2);
  el("details").hidden = false;
  addHistory(operation, data.elapsed_ms);
  setStatus("Response received.");
}

async function checkConnection() {
  const connection = el("connection");
  el("check").disabled = true;
  connection.textContent = "Checking connection";
  try {
    const response = await fetch("/api/health", {cache: "no-store"});
    const data = await response.json();
    updateConnection(response.ok, Array.isArray(data.operations) ? data.operations : null);
  } catch {
    updateConnection(false);
  } finally {
    el("check").disabled = false;
  }
}

form.addEventListener("change", (event) => {
  if (event.target.name === "operation") {
    updateMode();
  }
});
el("check").addEventListener("click", checkConnection);
el("sample").addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 224;
  const context = canvas.getContext("2d");
  context.fillStyle = "#b33b33";
  context.fillRect(0, 0, 224, 224);
  context.fillStyle = "#ffd25d";
  context.beginPath();
  context.arc(112, 112, 55, 0, Math.PI * 2);
  context.fill();
  imageInput.value = "";
  updateImage(canvas.toDataURL("image/png"), "Sample image with a yellow circle on red.");
});
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file || file.size > 5 * 1024 * 1024) {
    updateImage("", file ? "Choose an image under 5 MiB." : "Choose a local image or use the sample.");
    return;
  }
  const reader = new FileReader();
  updateImage("", "Reading image…");
  reader.onload = () => {
    if (imageInput.files[0] === file) updateImage(reader.result, file.name);
  };
  reader.onerror = () => {
    if (imageInput.files[0] === file) updateImage("", "Could not read the image.");
  };
  reader.readAsDataURL(file);
});
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (el("submit").disabled) return;
  const operation = form.elements.operation.value;
  const request = {operation};
  clearResponse();
  if (operation !== "generate") {
    if (!imageUrl) {
      setStatus("Choose an image first.", true);
      return;
    }
    request.payload = imageUrl.split(",", 2)[1];
  }
  if (operation !== "infer") {
    request.prompt = el("prompt").value.trim();
    request.max_tokens = Number(el("tokens").value);
  }
  requestPending = true;
  updateMode();
  setStatus("Running inference…");
  try {
    const response = await fetch("/api/request", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(request),
    });
    const data = await response.json();
    if (!response.ok) {
      if (operation === "describe" && data.error?.includes("does not support")) {
        el("vlm-setup").hidden = false;
        el("vlm-setup").open = true;
        throw new Error("The connected backend cannot answer image questions. Use the SmolVLM setup under Task.");
      }
      throw new Error(data.error || "The request failed.");
    }
    showResponse(data, operation);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    requestPending = false;
    updateMode();
  }
});

updateMode();
checkConnection();
