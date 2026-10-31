/* Pensive Console — vanilla JS client for the admin API. */
"use strict";

const TYPE_COLORS = {
  semantic: "#00ed64",
  episodic: "#0498ec",
  taxonomic: "#fbbf24",
  procedural: "#b45af2",
};

const state = {
  user_id: "default",
  mem_type: null, // null = all types
  types: [],
};

/* ---------- helpers ---------- */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch (_) {
    /* no body */
  }
  if (!response.ok) {
    const detail = body && body.detail ? String(body.detail) : response.statusText;
    throw new Error(detail);
  }
  return body;
}

let toastTimer = null;
function toast(message, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = "toast" + (isError ? " error" : "");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 3600);
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
  );
}

function fmtWhen(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 19).replace("T", " ");
  return date.toLocaleString();
}

/* ---------- tabs ---------- */

document.getElementById("tabs").addEventListener("click", (event) => {
  const button = event.target.closest(".tab");
  if (!button) return;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === button));
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${button.dataset.tab}`);
  });
  if (button.dataset.tab === "memories") loadMemories();
  if (button.dataset.tab === "prompt") loadPrompt();
  if (button.dataset.tab === "agents") loadAgents();
});

/* ---------- memories ---------- */

async function loadTypes() {
  const data = await api("/api/admin/memory-types");
  state.types = data.types;
  const grid = document.getElementById("type-grid");
  grid.innerHTML = "";
  const all = typeCard(null, "All types", "Every memory for this user across the four types");
  all.dataset.type = "all";
  all.classList.toggle("selected", state.mem_type === null);
  grid.appendChild(all);
  for (const meta of state.types) {
    grid.appendChild(typeCard(meta.type, cap(meta.type), meta.description));
  }
}

function cap(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function typeCard(type, name, description) {
  const card = document.createElement("div");
  card.className = "type-card" + (state.mem_type === type ? " selected" : "");
  card.dataset.type = type || "all";
  const color = type ? TYPE_COLORS[type] : "#f9fbfa";
  card.innerHTML = `
    <div class="type-name">
      <span class="type-dot" style="background:${color}"></span>${esc(name)}
    </div>
    <div class="type-desc">${esc(description)}</div>
  `;
  card.addEventListener("click", () => {
    state.mem_type = type;
    document.querySelectorAll(".type-card").forEach((c) => {
      c.classList.toggle("selected", c.dataset.type === (type || "all"));
    });
    loadMemories();
  });
  return card;
}

async function loadMemories() {
  const user_id = document.getElementById("user-id").value.trim() || "default";
  state.user_id = user_id;
  const params = new URLSearchParams({ user_id });
  if (state.mem_type) params.set("mem_type", state.mem_type);
  try {
    const data = await api(`/api/admin/memories?${params}`);
    renderMemories(data.memories);
    updateCounts(data.counts);
  } catch (error) {
    document.getElementById("memory-list").innerHTML =
      `<div class="empty">Could not load memories: ${esc(error.message)}</div>`;
  }
}

function setCardCount(card, text) {
  let el = card.querySelector(".type-count");
  if (!el) {
    el = document.createElement("div");
    el.className = "type-count";
    card.appendChild(el);
  }
  el.textContent = text;
}

function updateCounts(counts) {
  // Counts come from the API computed UNFILTERED, so panel badges stay
  // correct no matter which type filter is currently active.
  const map = counts || {};
  let total = 0;
  for (const value of Object.values(map)) total += value;
  const allCard = document.querySelector('.type-card[data-type="all"]');
  if (allCard) setCardCount(allCard, total === 1 ? "1 memory total" : `${total} memories total`);
  for (const meta of state.types) {
    const count = map[meta.type] ?? 0;
    const card = document.querySelector(`.type-card[data-type="${meta.type}"]`);
    if (card) setCardCount(card, count === 1 ? "1 memory" : `${count} memories`);
  }
}

function renderMemories(rows) {
  const list = document.getElementById("memory-list");
  if (!rows.length) {
    list.innerHTML = `<div class="empty">No memories found${state.mem_type ? ` of type ${state.mem_type}` : ""} for user “${esc(state.user_id)}”.</div>`;
    return;
  }
  list.innerHTML = "";
  for (const row of rows) {
    list.appendChild(memoryRow(row));
  }
}

function memoryRow(row) {
  const div = document.createElement("div");
  div.className = "memory-row";
  div.style.setProperty("--type-color", TYPE_COLORS[row.type] || "#00ed64");

  const meta = state.types.find((t) => t.type === row.type);
  const subtitle = meta ? meta.description : "";

  div.innerHTML = `
    <div class="row-head">
      <span class="badge" style="background:${TYPE_COLORS[row.type] || "#00ed64"}22;color:${TYPE_COLORS[row.type] || "#00ed64"}">${esc(row.type)}</span>
      <span class="label" title="${esc(subtitle)}">${esc(row.label || row.key || "")}</span>
      <span class="when">${fmtWhen(row.updated_at || row.recorded_at)}</span>
    </div>
    <p class="content">${esc(row.content)}</p>
    <div class="row-actions">
      <button class="btn ghost small" data-action="edit">Edit</button>
      <button class="btn danger small" data-action="delete">Prune</button>
    </div>
  `;

  div.querySelector('[data-action="edit"]').addEventListener("click", () => {
    div.querySelector(".row-actions").remove();
    const textarea = document.createElement("textarea");
    textarea.rows = 3;
    textarea.value = row.content;
    const actions = document.createElement("div");
    actions.className = "row-actions";
    const save = document.createElement("button");
    save.className = "btn primary small";
    save.textContent = "Save & re-embed";
    const cancel = document.createElement("button");
    cancel.className = "btn ghost small";
    cancel.textContent = "Cancel";
    actions.append(save, cancel);
    div.insertBefore(textarea, div.querySelector(".memory-row > .content").nextSibling ?? null);
    div.querySelector(".content").replaceWith(textarea);
    div.appendChild(actions);
    cancel.addEventListener("click", () => loadMemories());
    save.addEventListener("click", async () => {
      save.disabled = true;
      try {
        await updateMemory(row.type, row.key, textarea.value);
        toast("Memory corrected and re-embedded");
        loadMemories();
      } catch (error) {
        toast(error.message, true);
        save.disabled = false;
      }
    });
  });

  div.querySelector('[data-action="delete"]').addEventListener("click", async () => {
    if (!confirm(`Prune this ${row.type} memory?\n\n${row.content}`)) return;
    try {
      await api(
        `/api/admin/memories/${encodeURIComponent(row.type)}/${encodeURIComponent(row.key)}?user_id=${encodeURIComponent(state.user_id)}`,
        { method: "DELETE" },
      );
      toast("Memory pruned");
      loadMemories();
    } catch (error) {
      toast(error.message, true);
    }
  });

  return div;
}

async function updateMemory(type, key, content) {
  return api(
    `/api/admin/memories/${encodeURIComponent(type)}/${encodeURIComponent(key)}?user_id=${encodeURIComponent(state.user_id)}`,
    {
      method: "PUT",
      body: JSON.stringify({ content }),
    },
  );
}

document.getElementById("user-id").addEventListener("change", loadMemories);
document.getElementById("refresh-memories").addEventListener("click", loadMemories);

/* ---------- system prompt ---------- */

async function loadPrompt() {
  try {
    const data = await api("/api/admin/system-prompt");
    document.getElementById("system-prompt").value = data.text;
    document.getElementById("prompt-status").textContent = "";
  } catch (error) {
    toast(error.message, true);
  }
}

document.getElementById("save-prompt").addEventListener("click", async () => {
  const textarea = document.getElementById("system-prompt");
  const status = document.getElementById("prompt-status");
  try {
    await api("/api/admin/system-prompt", {
      method: "PUT",
      body: JSON.stringify({ text: textarea.value }),
    });
    status.textContent = "Saved — applies to the main context.";
    toast("System prompt updated");
  } catch (error) {
    toast(error.message, true);
  }
});

/* ---------- agents ---------- */

async function loadAgents() {
  const list = document.getElementById("agent-list");
  try {
    const data = await api("/api/admin/agents");
    list.innerHTML = "";
    if (!data.agents.length) {
      list.innerHTML = `<div class="empty">No test agents yet. Create one above.</div>`;
      return;
    }
    for (const agent of data.agents) {
      list.appendChild(agentCard(agent));
    }
  } catch (error) {
    list.innerHTML = `<div class="empty">Could not load agents: ${esc(error.message)}</div>`;
  }
}

function toolCount(tools) {
  return Array.isArray(tools) ? tools.length : 0;
}

function agentCard(agent) {
  const div = document.createElement("div");
  div.className = "agent-card";
  const promptPreview = agent.system_prompt ? `“${esc(agent.system_prompt.slice(0, 140))}${agent.system_prompt.length > 140 ? "…" : ""}”` : "<em>no custom system prompt</em>";
  div.innerHTML = `
    <div class="agent-head">
      <span class="agent-name">${esc(agent.name)}</span>
      <span class="agent-meta">id ${esc(agent.id)} · ${toolCount(agent.tools)} tool(s) · memory scope agent:${esc(agent.id)}</span>
      <button class="btn danger small" data-action="delete">Delete</button>
    </div>
    <div class="agent-prompt">${promptPreview}</div>
    <div class="run-box">
      <input type="text" placeholder="Task to run, e.g. Write a one-line release note for the fix" />
      <button class="btn primary small" data-action="run">Run in isolation</button>
    </div>
    <div class="run-result" hidden></div>
  `;

  const resultBox = div.querySelector(".run-result");
  const taskInput = div.querySelector(".run-box input");
  const runButton = div.querySelector('[data-action="run"]');

  runButton.addEventListener("click", async () => {
    const task = taskInput.value.trim();
    if (!task) {
      toast("Enter a task first", true);
      return;
    }
    runButton.disabled = true;
    resultBox.hidden = false;
    resultBox.innerHTML = `<p class="decisions">Running agent ${esc(agent.name)} in isolation…</p>`;
    try {
      const result = await api(`/api/admin/agents/${encodeURIComponent(agent.id)}/run`, {
        method: "POST",
        body: JSON.stringify({ task }),
      });
      const decisionLines = (result.decisions || [])
        .map((d) => `• ${cap(String(d.step || "").replaceAll("_", " "))}: ${d.decision} — ${d.reason}`)
        .join("<br>");
      const toolCalls = result.tool_calls && result.tool_calls.length
        ? `<details><summary>${result.tool_calls.length} tool call(s) requested</summary>${esc(JSON.stringify(result.tool_calls, null, 2))}</details>`
        : "";
      resultBox.innerHTML = `
        <p class="answer">${esc(result.answer || "(no answer text)")}</p>
        ${toolCalls}
        <p class="decisions">${decisionLines || "no decisions recorded"}</p>
      `;
    } catch (error) {
      resultBox.innerHTML = `<p class="decisions" style="color:var(--danger)">${esc(error.message)}</p>`;
    } finally {
      runButton.disabled = false;
    }
  });

  div.querySelector('[data-action="delete"]').addEventListener("click", async () => {
    if (!confirm(`Delete agent “${agent.name}”? Its isolated memories remain in MongoDB.`)) return;
    try {
      await api(`/api/admin/agents/${encodeURIComponent(agent.id)}`, { method: "DELETE" });
      toast("Agent deleted");
      loadAgents();
    } catch (error) {
      toast(error.message, true);
    }
  });

  return div;
}

document.getElementById("create-agent").addEventListener("click", async () => {
  const name = document.getElementById("agent-name").value.trim();
  const system_prompt = document.getElementById("agent-prompt").value;
  const toolsRaw = document.getElementById("agent-tools").value.trim();
  if (!name) {
    toast("Agent name is required", true);
    return;
  }
  let tools = [];
  if (toolsRaw) {
    try {
      tools = JSON.parse(toolsRaw);
    } catch (error) {
      toast(`Tools must be valid JSON: ${error.message}`, true);
      return;
    }
  }
  try {
    await api("/api/admin/agents", {
      method: "POST",
      body: JSON.stringify({ name, system_prompt, tools }),
    });
    toast(`Agent “${name}” created`);
    document.getElementById("agent-name").value = "";
    document.getElementById("agent-prompt").value = "";
    document.getElementById("agent-tools").value = "";
    loadAgents();
  } catch (error) {
    toast(error.message, true);
  }
});

/* ---------- boot ---------- */

loadTypes().then(loadMemories).catch((error) => toast(error.message, true));