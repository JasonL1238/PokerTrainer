(function () {
  "use strict";
  const config = window.LABELER_CONFIG;
  const canvas = document.getElementById("canvas");
  const frame = document.getElementById("canvas-frame");
  const stage = document.getElementById("stage");
  const ctx = canvas.getContext("2d");
  const state = { item: null, image: new Image(), boxes: [], selected: -1, activeClass: config.classes[0], history: [], pointer: null, bootstrapCache: {} };
  const $ = (id) => document.getElementById(id);

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
      ctx.strokeStyle = color; ctx.lineWidth = index === state.selected ? 4 : 2;
      ctx.fillStyle = color + "22"; ctx.fillRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
      ctx.font = "bold 16px system-ui"; ctx.fillStyle = color; ctx.fillText(box.label ? `${box.class} (${box.label})` : box.class, box.x1 + 4, Math.max(16, box.y1 + 16));
      if (index === state.selected) {
        ctx.fillStyle = color;
        [[box.x1, box.y1], [box.x2, box.y1], [box.x1, box.y2], [box.x2, box.y2]].forEach(([x, y]) => ctx.fillRect(x - 5, y - 5, 10, 10));
      }
    });
    if (state.pointer && state.pointer.mode === "draw") {
      const box = normalize({ x1: state.pointer.start.x, y1: state.pointer.start.y, x2: state.pointer.current.x, y2: state.pointer.current.y });
      ctx.strokeStyle = config.colors[state.activeClass]; ctx.setLineDash([6, 4]); ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1); ctx.setLineDash([]);
    }
    refreshCardPicker();
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
    const label = box && box.label ? box.label : "";
    const rank = label.slice(0, 1), suit = label.slice(1, 2);
    $("card-current").innerHTML = box
      ? (label ? `Card: <b>${label}</b>` : "Pick a rank and suit")
      : "Select a face_card box";
    document.querySelectorAll(".rank-button").forEach((b) => b.classList.toggle("active", !!box && b.dataset.rank === rank));
    document.querySelectorAll(".suit-button").forEach((b) => b.classList.toggle("active", !!box && b.dataset.suit === suit));
  }
  function setCardPart(kind, value) {
    const box = selectedCardBox();
    if (!box) return;
    remember();
    let rank = box.label ? box.label.slice(0, 1) : "";
    let suit = box.label ? box.label.slice(1, 2) : "";
    if (kind === "rank") rank = value; else suit = value;
    if (rank && suit) box.label = rank + suit; else delete box.label;
    draw();
  }
  function clearCardLabel() {
    const box = selectedCardBox();
    if (!box || !box.label) return;
    remember(); delete box.label; draw();
  }
  function setActiveClass(name, relabelSelected) {
    state.activeClass = name;
    if (relabelSelected && state.selected >= 0 && state.boxes[state.selected]) {
      remember(); state.boxes[state.selected].class = name;
      if (name !== "face_card") delete state.boxes[state.selected].label;
      draw();
    }
    document.querySelectorAll(".class-button").forEach((button) => button.classList.toggle("active", button.dataset.class === name));
  }
  function updateProgress() {
    fetch("/api/progress").then((response) => response.json()).then((data) => { $("progress").textContent = `${data.labeled} labeled | ${data.clean} clean | ${data.duplicate} duplicate | ${data.undecided} undecided | ${data.total} total`; });
  }
  function fitFrame() {
    if (!state.item || !canvas.width || !canvas.height || frame.hidden) return;
    const style = getComputedStyle(stage), horizontalPadding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight), verticalPadding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    const availableWidth = Math.max(1, stage.clientWidth - horizontalPadding), availableHeight = Math.max(1, stage.clientHeight - verticalPadding);
    const scale = Math.min(availableWidth / canvas.width, availableHeight / canvas.height);
    frame.style.width = `${Math.max(1, Math.floor(canvas.width * scale))}px`;
    frame.style.height = `${Math.max(1, Math.floor(canvas.height * scale))}px`;
  }
  function loadItem(item) {
    state.item = item; state.boxes = (item && item.boxes ? item.boxes : []).map(normalize); state.selected = -1; state.history = []; state.pointer = null;
    if (!item) { $("status").textContent = "No image available"; frame.hidden = true; stage.querySelector(".empty").hidden = false; return; }
    $("status").textContent = `${item.id} | ${item.status}`; frame.hidden = false; stage.querySelector(".empty").hidden = true;
    state.image.onload = function () { canvas.width = state.image.naturalWidth; canvas.height = state.image.naturalHeight; fitFrame(); draw(); };
    state.image.src = `${item.image_url}?v=${Date.now()}`; updateProgress();
    if (item.status === "undecided" && !item.boxes.length) bootstrap(item.id);
  }
  function bootstrap(fileId) {
    if (state.bootstrapCache[fileId]) { state.boxes = state.bootstrapCache[fileId].map(normalize); draw(); return; }
    $("status").textContent = `${fileId} | loading YOLO detections...`;
    getJson(`/api/bootstrap/${encodeURIComponent(fileId)}`).then((data) => {
      if (!state.item || state.item.id !== fileId || state.boxes.length) return;
      state.bootstrapCache[fileId] = data.boxes || []; state.boxes = state.bootstrapCache[fileId].map(normalize); state.history = []; draw();
      $("status").textContent = `${fileId} | YOLO prelabels loaded (${state.boxes.length})`;
    }).catch(showError);
  }
  function getJson(url) { return fetch(url).then((response) => response.json()).then((data) => { if (data.error) throw new Error(data.error); return data; }); }
  function loadNext() { getJson("/api/next").then((data) => loadItem(data.item)).catch(showError); }
  function seek(direction) { getJson(`/api/seek?dir=${direction}&id=${encodeURIComponent(state.item ? state.item.id : "")}`).then((data) => loadItem(data.item)).catch(showError); }
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
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea, select")) return;
    if (/^[1-9]$/.test(event.key) && Number(event.key) <= config.classes.length) setActiveClass(config.classes[Number(event.key) - 1], state.selected >= 0);
    else if (event.key.toLowerCase() === "d" && state.selected >= 0) { remember(); state.boxes.splice(state.selected, 1); state.selected = -1; draw(); }
    else if (event.key.toLowerCase() === "z") undo();
    else if (event.key === "Enter") { event.preventDefault(); save("labeled", true); }
    else if (event.key.toLowerCase() === "k") save("clean", true);
    else if (event.key.toLowerCase() === "x") save("duplicate", true);
    else if (event.key.toLowerCase() === "s") skip();
    else if (event.key === "ArrowLeft") seek("prev");
    else if (event.key === "ArrowRight") seek("next");
  });
  setActiveClass(state.activeClass); loadNext();
})();
