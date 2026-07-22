(function () {
  "use strict";
  const config = window.LABELER_CONFIG;
  const canvas = document.getElementById("canvas");
  const frame = document.getElementById("canvas-frame");
  const stage = document.getElementById("stage");
  const ctx = canvas.getContext("2d");
  const state = { item: null, image: new Image(), boxes: [], selected: -1, activeClass: config.classes[0], history: [], pointer: null, bootstrapCache: {}, carryEnabled: true };
  const $ = (id) => document.getElementById(id);
  const DEFAULT_QUEUE = "two_model_validation";
  const queueName = new URLSearchParams(location.search).get("queue") || DEFAULT_QUEUE;
  const initialItemId = new URLSearchParams(location.search).get("id") || "";
  const requestedView = new URLSearchParams(location.search).get("view") || "undecided";
  const views = new Set(["undecided", "labeled", "labeled_sus", "unlabeled_sus", "clean", "duplicate", "all"]);
  const viewName = views.has(requestedView) ? requestedView : "undecided";
  const isChronologicalQueueReview = queueName === DEFAULT_QUEUE && viewName === "labeled";
  const isLabeledSusView = viewName === "labeled_sus";
  const isUnlabeledSusView = viewName === "unlabeled_sus";
  const isSusView = isLabeledSusView || isUnlabeledSusView;

  function cloneBoxes() { return state.boxes.map((box) => ({ ...box })); }
  function remember() { state.history.push(cloneBoxes()); if (state.history.length > 40) state.history.shift(); }
  function canonicalCardLabel(label) {
    if (label === undefined || label === null) return undefined;
    let text = String(label).trim();
    if (!text || text.toLowerCase() === "joker") return undefined;
    if (text.slice(0, 2) === "10") text = "T" + text.slice(2);
    if (text.length === 2) text = text[0].toUpperCase() + text[1].toLowerCase();
    return text.length === 2 && config.cardRanks.includes(text[0]) && config.cardSuits.includes(text[1]) ? text : undefined;
  }
  function normalize(box) {
    const x1 = Math.min(box.x1, box.x2), y1 = Math.min(box.y1, box.y2);
    const out = { ...box, x1, y1, x2: Math.max(box.x1, box.x2), y2: Math.max(box.y1, box.y2) };
    const canonical = out.class === config.cardLabelClass ? canonicalCardLabel(out.label) : undefined;
    if (canonical) out.label = canonical; else delete out.label;
    return out;
  }
  function clampPoint(point) { return { x: Math.max(0, Math.min(canvas.width, point.x)), y: Math.max(0, Math.min(canvas.height, point.y)) }; }
  function pointerToImage(event) {
    const rect = canvas.getBoundingClientRect();
    return clampPoint({ x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height });
  }
  function hitTest(point) {
    const rect = canvas.getBoundingClientRect();
    const tolerance = Math.max(5, 9 * canvas.width / rect.width);
    for (let index = state.boxes.length - 1; index >= 0; index--) {
      const box = normalize(state.boxes[index]);
      if (state.selected === index) {
        const handles = { nw: [box.x1, box.y1], ne: [box.x2, box.y1], sw: [box.x1, box.y2], se: [box.x2, box.y2] };
        for (const [handle, xy] of Object.entries(handles)) if (Math.abs(point.x - xy[0]) <= tolerance && Math.abs(point.y - xy[1]) <= tolerance) return { index, mode: handle };
      }
      if (point.x >= box.x1 - tolerance && point.x <= box.x2 + tolerance && point.y >= box.y1 - tolerance && point.y <= box.y2 + tolerance) return { index, mode: "move" };
    }
    return null;
  }
  function draw() {
    if (!state.image.complete || !canvas.width) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
    state.boxes.forEach((raw, index) => {
      const box = normalize(raw), color = config.colors[box.class] || "#ffffff";
      const isFace = box.class === config.cardLabelClass;
      const isSelected = index === state.selected;
      const missingCard = isFace && !(draftRank(box) && draftSuit(box));
      ctx.strokeStyle = missingCard ? "#f59e0b" : color;
      ctx.lineWidth = isSelected ? 4 : (missingCard ? 3 : 2);
      ctx.fillStyle = missingCard ? "#f59e0b22" : color + "22";
      ctx.fillRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      if (!isFace) {
        ctx.font = "bold 16px system-ui"; ctx.fillStyle = color;
        ctx.fillText(box.class, box.x1 + 4, Math.max(16, box.y1 + 16));
      } else {
        drawCardBadge(box, isSelected);
      }
      if (isSelected) {
        ctx.fillStyle = missingCard ? "#f59e0b" : color;
        [[box.x1, box.y1], [box.x2, box.y1], [box.x1, box.y2], [box.x2, box.y2]].forEach(([x, y]) => ctx.fillRect(x - 5, y - 5, 10, 10));
      }
    });
    if (state.pointer && state.pointer.mode === "draw") {
      const box = normalize({ x1: state.pointer.start.x, y1: state.pointer.start.y, x2: state.pointer.current.x, y2: state.pointer.current.y });
      ctx.strokeStyle = config.colors[state.activeClass]; ctx.setLineDash([6, 4]); ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1); ctx.setLineDash([]);
    }
    refreshCardPicker();
    refreshFaceCardList();
  }
  function drawCardBadge(box, selected) {
    const rank = draftRank(box), suit = draftSuit(box);
    const complete = !!(rank && suit);
    const symbol = suit ? (config.cardSuitSymbols[suit] || suit) : "?";
    const text = complete ? (rank + symbol) : ((rank || "?") + (suit ? symbol : "?"));
    const boxW = Math.max(1, box.x2 - box.x1), boxH = Math.max(1, box.y2 - box.y1);
    const fontSize = Math.max(18, Math.min(56, Math.round(Math.min(boxW, boxH) * 0.42), Math.round(canvas.width / 18)));
    ctx.font = `bold ${fontSize}px system-ui`;
    const padX = fontSize * 0.35, padY = fontSize * 0.18;
    const badgeWidth = ctx.measureText(text).width + padX * 2, badgeHeight = fontSize + padY * 2;
    const bx = Math.min(Math.max(0, box.x1 + (boxW - badgeWidth) / 2), Math.max(0, canvas.width - badgeWidth));
    let by = box.y1 - badgeHeight - 6;
    if (by < 0) by = Math.min(box.y1 + 6, Math.max(0, canvas.height - badgeHeight));
    ctx.fillStyle = complete ? "rgba(11,16,32,0.92)" : "rgba(120,53,15,0.95)";
    ctx.fillRect(bx, by, badgeWidth, badgeHeight);
    ctx.strokeStyle = selected ? "#fbbf24" : (complete ? "#e2e8f0" : "#fbbf24");
    ctx.lineWidth = selected ? 3 : 2;
    ctx.strokeRect(bx, by, badgeWidth, badgeHeight);
    ctx.textBaseline = "top";
    ctx.fillStyle = complete
      ? ((suit === "h" || suit === "d") ? "#f87171" : "#f8fafc")
      : "#fde68a";
    ctx.fillText(text, bx + padX, by + padY);
    ctx.textBaseline = "alphabetic";
  }
  function selectedCardBox() {
    const box = state.selected >= 0 ? state.boxes[state.selected] : null;
    return box && box.class === config.cardLabelClass ? box : null;
  }
  function refreshCardPicker() {
    const picker = $("card-picker");
    if (!picker) return;
    const box = selectedCardBox();
    picker.dataset.disabled = box ? "false" : "true";
    const rank = box ? draftRank(box) : "", suit = box ? draftSuit(box) : "";
    const rankText = rank || "&mdash;", suitText = suit ? `<span class="suit-${suit}">${config.cardSuitSymbols[suit] || suit}</span>` : "&mdash;";
    $("card-current").innerHTML = box
      ? (rank && suit
          ? `Card: <b>${rank}${suitText}</b>`
          : `Rank <b>${rankText}</b> &nbsp; Suit <b>${suitText}</b> <span style="color:#fbbf24">(incomplete)</span>`)
      : "Select a face_card box";
    document.querySelectorAll(".rank-button").forEach((b) => b.classList.toggle("active", !!box && b.dataset.rank === rank));
    document.querySelectorAll(".suit-button").forEach((b) => b.classList.toggle("active", !!box && b.dataset.suit === suit));
  }
  function faceCardLabelHtml(box) {
    const rank = draftRank(box), suit = draftSuit(box);
    if (rank && suit) {
      return `${rank}<span class="suit-${suit}">${config.cardSuitSymbols[suit] || suit}</span>`;
    }
    return '<span class="face-missing">??</span>';
  }
  function refreshFaceCardList() {
    const list = $("face-card-list");
    if (!list) return;
    const faces = state.boxes
      .map((box, index) => ({ box, index }))
      .filter((item) => item.box.class === config.cardLabelClass);
    if (!faces.length) {
      list.innerHTML = '<div class="face-card-empty">No face cards on this frame</div>';
      return;
    }
    list.innerHTML = faces.map(({ box, index }) => {
      const missing = !(draftRank(box) && draftSuit(box));
      const selected = index === state.selected;
      return `<button type="button" class="face-chip${missing ? " missing" : ""}${selected ? " active" : ""}" data-index="${index}">${faceCardLabelHtml(box)}</button>`;
    }).join("");
    list.querySelectorAll(".face-chip").forEach((button) => {
      button.addEventListener("click", () => {
        state.selected = Number(button.dataset.index);
        setActiveClass(config.cardLabelClass, false);
        draw();
      });
    });
  }
  function draftRank(box) { return box._r !== undefined ? box._r : (box.label ? box.label.slice(0, 1) : ""); }
  function draftSuit(box) { return box._s !== undefined ? box._s : (box.label ? box.label.slice(1, 2) : ""); }
  function setCardPart(kind, value) {
    const box = selectedCardBox();
    if (!box) return;
    remember();
    let rank = draftRank(box), suit = draftSuit(box);
    if (kind === "rank") rank = value; else suit = value;
    box._r = rank; box._s = suit;
    if (rank && suit) box.label = rank + suit; else delete box.label;
    draw();
  }
  function clearCardLabel() {
    const box = selectedCardBox();
    if (!box) return;
    if (!box.label && !box._r && !box._s) return;
    remember(); delete box.label; box._r = ""; box._s = ""; draw();
  }
  function setActiveClass(name, relabelSelected) {
    state.activeClass = name;
    if (relabelSelected && state.selected >= 0 && state.boxes[state.selected]) {
      remember(); state.boxes[state.selected].class = name;
      if (name !== "face_card") { const b = state.boxes[state.selected]; delete b.label; delete b._r; delete b._s; }
      draw();
    }
    document.querySelectorAll(".class-button").forEach((button) => button.classList.toggle("active", button.dataset.class === name));
  }
  function updateProgress() {
    const progressParams = new URLSearchParams();
    if (queueName) progressParams.set("queue", queueName);
    if (isLabeledSusView) progressParams.set("status", "labeled_sus");
    if (isUnlabeledSusView) progressParams.set("status", "unlabeled_sus");
    const qs = progressParams.toString();
    const url = `/api/progress${qs ? `?${qs}` : ""}`;
    fetch(url).then((response) => response.json()).then((data) => {
      $("progress").textContent = `${data.labeled} labeled | ${data.clean} clean | ${data.duplicate} duplicate | ${data.undecided} undecided | ${data.total} total`;
      const queueEl = $("queue-progress");
      if (isLabeledSusView && typeof data.labeled_sus === "number") {
        queueEl.hidden = false;
        queueEl.textContent = `Labeled sus: ${data.labeled_sus} still flagged for re-review`;
      } else if (isUnlabeledSusView && typeof data.unlabeled_sus === "number") {
        queueEl.hidden = false;
        queueEl.textContent = `Unlabeled sus: ${data.unlabeled_sus} still flagged for review`;
      } else if (data.queue) {
        const q = data.queue;
        queueEl.hidden = false;
        queueEl.textContent = `Priority "${q.name}": ${q.labeled + q.clean + q.duplicate} done | ${q.undecided} left | ${q.total} total`;
      } else {
        queueEl.hidden = true;
      }
    });
  }
  function fitFrame() {
    if (!state.item || !canvas.width || !canvas.height || frame.hidden) return;
    const style = getComputedStyle(stage), horizontalPadding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight), verticalPadding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    const availableWidth = Math.max(1, stage.clientWidth - horizontalPadding), availableHeight = Math.max(1, stage.clientHeight - verticalPadding);
    const scale = Math.min(availableWidth / canvas.width, availableHeight / canvas.height);
    frame.style.width = `${Math.max(1, Math.floor(canvas.width * scale))}px`;
    frame.style.height = `${Math.max(1, Math.floor(canvas.height * scale))}px`;
  }
  function clearAll() {
    if (!state.item || !state.boxes.length) return;
    remember(); state.boxes = []; state.selected = -1; draw();
    $("status").textContent = `${state.item.id} | cleared`;
  }
  function loadItem(item) {
    const carry = state.item && state.carryEnabled ? cloneBoxes() : [];
    state.item = item; state.boxes = (item && item.boxes ? item.boxes : []).map(normalize); state.selected = -1; state.history = []; state.pointer = null;
    if (!item) { $("status").textContent = "No image available"; frame.hidden = true; stage.querySelector(".empty").hidden = false; return; }
    const susNote = Array.isArray(item.sus_reasons) && item.sus_reasons.length
      ? ` | sus: ${item.sus_reasons.join("; ")}`
      : "";
    $("status").textContent = `${item.id} | ${item.status}${susNote}`; frame.hidden = false; stage.querySelector(".empty").hidden = true;
    state.image.onload = function () { canvas.width = state.image.naturalWidth; canvas.height = state.image.naturalHeight; fitFrame(); draw(); };
    state.image.src = `${item.image_url}?v=${Date.now()}`; updateProgress();
    if (item.status === "undecided" && !item.boxes.length) {
      if (carry.length) {
        state.boxes = carry.map((box) => normalize({ ...box })); state.history = []; draw();
        $("status").textContent = `${item.id} | carried ${state.boxes.length} labels from previous frame (Shift+C clears)`;
      } else {
        bootstrap(item.id);
      }
    }
  }
  function bootstrap(fileId) {
    if (state.bootstrapCache[fileId]) { state.boxes = state.bootstrapCache[fileId].map(normalize); draw(); return; }
    $("status").textContent = `${fileId} | loading YOLO detections...`;
    // One setup: always bootstrap Model 1 regions + Model 2 card names.
    getJson(`/api/bootstrap/${encodeURIComponent(fileId)}?source=two_model`).then((data) => {
      if (!state.item || state.item.id !== fileId || state.boxes.length) return;
      state.bootstrapCache[fileId] = data.boxes || []; state.boxes = state.bootstrapCache[fileId].map(normalize); state.history = []; draw();
      const susNote = Array.isArray(state.item.sus_reasons) && state.item.sus_reasons.length
        ? ` | sus: ${state.item.sus_reasons.join("; ")}`
        : "";
      $("status").textContent = `${fileId} | YOLO prelabels loaded (${state.boxes.length})${susNote}`;
    }).catch(showError);
  }
  function getJson(url) { return fetch(url).then((response) => response.json()).then((data) => { if (data.error) throw new Error(data.error); return data; }); }
  function browseParams(extra = {}) {
    const params = new URLSearchParams({ status: viewName, queue: queueName, ...extra });
    if (isChronologicalQueueReview) {
      params.set("scope", "queue");
      params.set("order", "labeled_at");
      params.set("wrap", "next");
      if (!state.item) params.set("start", "latest");
    }
    return params.toString();
  }
  function loadNext() {
    const current = state.item ? { id: state.item.id } : {};
    getJson(`/api/next?${browseParams(current)}`).then((data) => loadItem(data.item)).catch(showError);
  }
  function seek(direction) {
    getJson(`/api/seek?${browseParams({ dir: direction, id: state.item ? state.item.id : "" })}`).then((data) => loadItem(data.item)).catch(showError);
  }
  function save(status, thenNext) {
    if (!state.item) return;
    const boxes = state.boxes.map(normalize).filter((box) => box.x2 - box.x1 >= 2 && box.y2 - box.y1 >= 2);
    fetch("/api/annotate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: state.item.id, status, boxes: ["clean", "duplicate"].includes(status) ? [] : boxes }) }).then((response) => response.json()).then((data) => { if (data.error) throw new Error(data.error); if (thenNext) loadNext(); else loadItem(data); }).catch(showError);
  }
  function showError(error) { $("status").textContent = `Error: ${error.message}`; }
  function undo() { if (!state.history.length) return; state.boxes = state.history.pop(); state.selected = -1; draw(); }
  function skip() { seek("next"); }
  window.addEventListener("resize", () => { fitFrame(); draw(); });

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.item) return;
    canvas.setPointerCapture(event.pointerId);
    const point = pointerToImage(event), hit = hitTest(point); state.pointer = hit ? { ...hit, start: point, current: point, original: { ...state.boxes[hit.index] } } : { mode: "draw", start: point, current: point };
    if (hit) { state.selected = hit.index; remember(); } else state.selected = -1;
    draw();
  });
  canvas.addEventListener("pointermove", (event) => { if (!state.pointer) return; const point = pointerToImage(event), p = state.pointer; p.current = point;
    if (p.mode === "draw") return draw();
    const box = { ...p.original }, dx = point.x - p.start.x, dy = point.y - p.start.y;
    if (p.mode === "move") { box.x1 += dx; box.x2 += dx; box.y1 += dy; box.y2 += dy; }
    if (p.mode.includes("n")) box.y1 = point.y;
    if (p.mode.includes("s")) box.y2 = point.y;
    if (p.mode.includes("w")) box.x1 = point.x;
    if (p.mode.includes("e")) box.x2 = point.x;
    state.boxes[p.index] = normalize({ ...box, x1: Math.max(0, Math.min(canvas.width, box.x1)), x2: Math.max(0, Math.min(canvas.width, box.x2)), y1: Math.max(0, Math.min(canvas.height, box.y1)), y2: Math.max(0, Math.min(canvas.height, box.y2)) }); draw();
  });
  canvas.addEventListener("pointerup", (event) => { if (!state.pointer) return; const p = state.pointer; if (p.mode === "draw") { const point = pointerToImage(event), box = normalize({ class: state.activeClass, x1: p.start.x, y1: p.start.y, x2: point.x, y2: point.y }); if (box.x2 - box.x1 >= 2 && box.y2 - box.y1 >= 2) { remember(); state.boxes.push(box); state.selected = state.boxes.length - 1; } } state.pointer = null; draw(); });
  document.querySelectorAll(".class-button").forEach((button) => button.addEventListener("click", () => setActiveClass(button.dataset.class, state.selected >= 0)));
  document.querySelectorAll(".rank-button").forEach((button) => button.addEventListener("click", () => setCardPart("rank", button.dataset.rank)));
  document.querySelectorAll(".suit-button").forEach((button) => button.addEventListener("click", () => setCardPart("suit", button.dataset.suit)));
  $("card-clear").addEventListener("click", clearCardLabel);
  $("prev").addEventListener("click", () => seek("prev")); $("next").addEventListener("click", loadNext); $("save").addEventListener("click", () => save("labeled", true)); $("clean").addEventListener("click", () => save("clean", true)); $("duplicate").addEventListener("click", () => save("duplicate", true));
  $("clear").addEventListener("click", clearAll);
  $("carry-toggle").addEventListener("change", (event) => { state.carryEnabled = event.target.checked; });
  $("view-filter").value = viewName;
  $("view-filter").addEventListener("change", (event) => {
    const url = new URL(location.href);
    url.searchParams.set("view", event.target.value);
    url.searchParams.set("queue", queueName || DEFAULT_QUEUE);
    location.assign(url);
  });
  if (isChronologicalQueueReview) {
    $("prev").textContent = "Older label";
    $("next").textContent = "Newer label";
  } else {
    $("next").textContent = viewName === "undecided"
      ? "Next undecided"
      : (viewName === "labeled_sus"
        ? "Next labeled sus"
        : (viewName === "unlabeled_sus" ? "Next unlabeled sus" : `Next ${viewName}`));
  }
  if (isSusView) {
    const doneBtn = $("sus-done");
    doneBtn.hidden = false;
    doneBtn.addEventListener("click", () => {
      const countText = ($("queue-progress").textContent || "").trim();
      const labeled = isLabeledSusView;
      const ok = confirm(
        labeled
          ? "Mark all remaining Labeled sus frames as good?\n\nThis clears the sus queue. Labels stay as saved; Browse switches to Labeled.\n\n"
          : "Clear all remaining Unlabeled sus frames?\n\nThis only clears the sus queue. Frames stay undecided; no labels are written.\n\n"
        + (countText || "Current sus queue will be emptied.")
      );
      if (!ok) return;
      fetch(labeled ? "/api/labeled-sus/done" : "/api/unlabeled-sus/done", { method: "POST" })
        .then((response) => response.json())
        .then((data) => {
          if (data.error) throw new Error(data.error);
          const url = new URL(location.href);
          url.searchParams.set("view", labeled ? "labeled" : "undecided");
          url.searchParams.set("queue", queueName || DEFAULT_QUEUE);
          location.assign(url);
        })
        .catch(showError);
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea, select")) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.shiftKey && event.key.toLowerCase() === "c") { event.preventDefault(); clearAll(); return; }
    // When a face_card box is selected, keystrokes label its rank+suit and take
    // priority over the global class/shortcut keys they would otherwise collide with.
    if (selectedCardBox()) {
      const rankKey = event.key.toUpperCase();
      if (config.cardRanks.includes(rankKey)) { event.preventDefault(); setCardPart("rank", rankKey); return; }
      if (config.cardSuits.includes(event.key.toLowerCase())) { event.preventDefault(); setCardPart("suit", event.key.toLowerCase()); return; }
      if (event.key === "Backspace" || event.key === "Delete") { event.preventDefault(); remember(); state.boxes.splice(state.selected, 1); state.selected = -1; draw(); return; }
      if (event.key === "Escape") { event.preventDefault(); state.selected = -1; draw(); return; }
    }
    if (/^[1-9]$/.test(event.key) && Number(event.key) <= config.classes.length) setActiveClass(config.classes[Number(event.key) - 1], state.selected >= 0);
    else if ((event.key.toLowerCase() === "d" || event.key === "Backspace" || event.key === "Delete") && state.selected >= 0) { event.preventDefault(); remember(); state.boxes.splice(state.selected, 1); state.selected = -1; draw(); }
    else if (event.key.toLowerCase() === "z") undo();
    else if (event.key === "Enter") { event.preventDefault(); save("labeled", true); }
    else if (event.key.toLowerCase() === "k") save("clean", true);
    else if (event.key.toLowerCase() === "x") save("duplicate", true);
    else if (event.key.toLowerCase() === "s") skip();
    else if (event.key === "ArrowLeft") seek("prev");
    else if (event.key === "ArrowRight") seek("next");
  });
  setActiveClass(state.activeClass);
  if (initialItemId) getJson(`/api/item/${encodeURIComponent(initialItemId)}`).then(loadItem).catch(showError);
  else loadNext();
})();
