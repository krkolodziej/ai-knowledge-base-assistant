const API_PREFIX = "/api/v1";

const state = {
  documents: [],
  pendingTraceTimer: null,
  pendingTraceIndex: 0,
};

const elements = {
  healthStatus: document.querySelector("#healthStatus"),
  documentForm: document.querySelector("#documentForm"),
  documentTitle: document.querySelector("#documentTitle"),
  contentType: document.querySelector("#contentType"),
  documentContent: document.querySelector("#documentContent"),
  sourceLabel: document.querySelector("#sourceLabel"),
  createButton: document.querySelector("#createButton"),
  refreshButton: document.querySelector("#refreshButton"),
  documentList: document.querySelector("#documentList"),
  documentCount: document.querySelector("#documentCount"),
  questionForm: document.querySelector("#questionForm"),
  questionInput: document.querySelector("#questionInput"),
  topK: document.querySelector("#topK"),
  askButton: document.querySelector("#askButton"),
  answerBlock: document.querySelector("#answerBlock"),
  answerText: document.querySelector("#answerText"),
  tracePanel: document.querySelector("#tracePanel"),
  traceStats: document.querySelector("#traceStats"),
  traceList: document.querySelector("#traceList"),
  sourceList: document.querySelector("#sourceList"),
  toast: document.querySelector("#toast"),
};

const TRACE_LABELS = {
  question_embedding_generated: "Question embedding generated",
  similar_chunks_retrieved: "Similar chunks retrieved",
  prompt_built: "Prompt built",
  answer_generated: "Answer generated",
  generation_skipped_no_sources: "Generation skipped",
};

const PENDING_TRACE_STEPS = [
  "Generating question embedding",
  "Searching pgvector chunks",
  "Building prompt",
  "Asking Ollama",
];

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_PREFIX}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const errorBody = await response.json();
      if (errorBody.detail) {
        detail = Array.isArray(errorBody.detail)
          ? errorBody.detail.map((item) => item.msg || JSON.stringify(item)).join(", ")
          : errorBody.detail;
      }
    } catch {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  return response.json();
}

function setBusy(button, isBusy, label) {
  button.disabled = isBusy;
  if (label) {
    button.textContent = isBusy ? label.busy : label.idle;
  }
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", isError);
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timeoutId);
  showToast.timeoutId = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 4200);
}

async function checkHealth() {
  try {
    await requestJson("/health");
    elements.healthStatus.textContent = "API online";
    elements.healthStatus.classList.add("is-ok");
    elements.healthStatus.classList.remove("is-error");
  } catch (error) {
    elements.healthStatus.textContent = "API offline";
    elements.healthStatus.classList.add("is-error");
    elements.healthStatus.classList.remove("is-ok");
  }
}

async function loadDocuments() {
  const data = await requestJson("/documents");
  state.documents = data.items;
  renderDocuments();
}

function renderDocuments() {
  elements.documentCount.textContent = String(state.documents.length);
  elements.documentList.innerHTML = "";

  if (state.documents.length === 0) {
    const emptyState = document.createElement("div");
    emptyState.className = "empty-state";
    emptyState.textContent = "No documents yet.";
    elements.documentList.append(emptyState);
    return;
  }

  for (const documentItem of state.documents) {
    const item = document.createElement("article");
    item.className = "document-item";

    const title = document.createElement("h3");
    title.textContent = documentItem.title;

    const meta = document.createElement("div");
    meta.className = "document-meta";
    meta.innerHTML = `
      <span>${documentItem.content_type}</span>
      <span>${new Date(documentItem.created_at).toLocaleString()}</span>
      <span>${documentItem.id}</span>
    `;

    const actions = document.createElement("div");
    actions.className = "document-actions";

    const status = document.createElement("span");
    status.className = `status-label ${documentItem.status}`;
    status.textContent = documentItem.status;

    const indexButton = document.createElement("button");
    indexButton.className = "index-button";
    indexButton.type = "button";
    indexButton.textContent = documentItem.status === "indexed" ? "Re-index" : "Index";
    indexButton.addEventListener("click", () => indexDocument(documentItem.id, indexButton));

    actions.append(status, indexButton);
    item.append(title, meta, actions);
    elements.documentList.append(item);
  }
}

async function createDocument(event) {
  event.preventDefault();
  setBusy(elements.createButton, true, { idle: "Add document", busy: "Adding" });

  const source = elements.sourceLabel.value.trim();
  const payload = {
    title: elements.documentTitle.value.trim(),
    content: elements.documentContent.value.trim(),
    content_type: elements.contentType.value,
    metadata: source ? { source } : {},
  };

  try {
    await requestJson("/documents", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.documentForm.reset();
    await loadDocuments();
    showToast("Document added.");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(elements.createButton, false, { idle: "Add document", busy: "Adding" });
  }
}

async function indexDocument(documentId, button) {
  setBusy(button, true, { idle: button.textContent, busy: "Indexing" });

  try {
    const result = await requestJson(`/documents/${documentId}/index`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    await loadDocuments();
    showToast(`Indexed ${result.chunks_indexed} chunks.`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(button, false, { idle: "Index", busy: "Indexing" });
  }
}

async function askQuestion(event) {
  event.preventDefault();
  setBusy(elements.askButton, true, { idle: "Ask question", busy: "Asking" });
  startPendingTrace();

  try {
    const result = await requestJson("/questions", {
      method: "POST",
      body: JSON.stringify({
        question: elements.questionInput.value.trim(),
        top_k: Number(elements.topK.value),
      }),
    });
    stopPendingTrace();
    renderAnswer(result);
  } catch (error) {
    stopPendingTrace();
    markPendingTraceFailed();
    showToast(error.message, true);
  } finally {
    setBusy(elements.askButton, false, { idle: "Ask question", busy: "Asking" });
  }
}

function renderAnswer(result) {
  elements.answerBlock.hidden = false;
  elements.answerText.textContent = result.answer;
  elements.sourceList.innerHTML = "";
  renderTrace(result.trace);

  if (result.sources.length === 0) {
    const emptyState = document.createElement("div");
    emptyState.className = "empty-state";
    emptyState.textContent = "No sources returned.";
    elements.sourceList.append(emptyState);
    return;
  }

  result.sources.forEach((source, index) => {
    const item = document.createElement("article");
    item.className = "source-item";

    const title = document.createElement("h3");
    title.textContent = `Source ${index + 1}`;

    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.innerHTML = `
      <span>chunk ${source.chunk_index}</span>
      <span>similarity ${source.similarity.toFixed(3)}</span>
      <span>distance ${source.distance.toFixed(3)}</span>
    `;

    const content = document.createElement("p");
    content.textContent = source.content;

    item.append(title, meta, content);
    elements.sourceList.append(item);
  });
}

function startPendingTrace() {
  stopPendingTrace();
  state.pendingTraceIndex = 0;
  elements.answerBlock.hidden = false;
  elements.answerText.textContent = "";
  elements.sourceList.innerHTML = "";
  elements.tracePanel.hidden = false;
  elements.traceStats.innerHTML = "";
  elements.traceList.innerHTML = "";

  PENDING_TRACE_STEPS.forEach((step, index) => {
    const item = document.createElement("li");
    item.className = index === 0 ? "trace-step is-active" : "trace-step";
    item.textContent = step;
    elements.traceList.append(item);
  });

  state.pendingTraceTimer = window.setInterval(advancePendingTrace, 900);
}

function advancePendingTrace() {
  if (state.pendingTraceIndex >= PENDING_TRACE_STEPS.length - 1) {
    return;
  }

  const steps = [...elements.traceList.querySelectorAll(".trace-step")];
  steps[state.pendingTraceIndex].classList.remove("is-active");
  steps[state.pendingTraceIndex].classList.add("is-complete");
  state.pendingTraceIndex += 1;
  steps[state.pendingTraceIndex].classList.add("is-active");
}

function stopPendingTrace() {
  if (state.pendingTraceTimer) {
    window.clearInterval(state.pendingTraceTimer);
    state.pendingTraceTimer = null;
  }
}

function markPendingTraceFailed() {
  const activeStep = elements.traceList.querySelector(".trace-step.is-active");
  if (activeStep) {
    activeStep.classList.remove("is-active");
    activeStep.classList.add("is-error");
  }
}

function renderTrace(trace) {
  elements.tracePanel.hidden = false;
  elements.traceStats.innerHTML = "";
  elements.traceList.innerHTML = "";

  const stats = [
    ["top_k", trace.top_k],
    ["chunks", trace.retrieved_chunks],
    ["context chars", trace.context_characters],
  ];

  stats.forEach(([label, value]) => {
    const item = document.createElement("div");
    item.className = "trace-stat";
    item.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    elements.traceStats.append(item);
  });

  trace.steps.forEach((step) => {
    const item = document.createElement("li");
    item.className = "trace-step is-complete";
    item.textContent = TRACE_LABELS[step] || step;
    elements.traceList.append(item);
  });
}

elements.documentForm.addEventListener("submit", createDocument);
elements.refreshButton.addEventListener("click", async () => {
  try {
    await loadDocuments();
    showToast("Documents refreshed.");
  } catch (error) {
    showToast(error.message, true);
  }
});
elements.questionForm.addEventListener("submit", askQuestion);

checkHealth();
loadDocuments().catch((error) => showToast(error.message, true));
