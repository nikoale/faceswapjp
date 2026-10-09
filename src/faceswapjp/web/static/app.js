// faceswapjp studio — single-page UI. Talks only to the local server (127.0.0.1).

const TOKEN = document.querySelector('meta[name="fsj-token"]').content;
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const withToken = (url) => url + (url.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(TOKEN);
const fileUrl = (path) => withToken("/api/file?path=" + encodeURIComponent(path));

// ---------------------------------------------------------------------------- icons (inline, local)
const I = {
  face: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c1.6-4 4.5-6 8-6s6.4 2 8 6"/></svg>',
  film: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/></svg>',
  target: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="11" r="3"/><path d="M7.5 17c1-1.8 2.6-2.8 4.5-2.8s3.5 1 4.5 2.8"/></svg>',
  sparkle: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/></svg>',
  export: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12M7 10l5 5 5-5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.6 2.5L16 9.5"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.5h.01"/></svg>',
  info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.5h.01"/></svg>',
  layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
  list: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/></svg>',
  lock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="10" rx="2.5"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>',
  folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
  chipIcon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/></svg>',
  prev: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg>',
  next: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  share: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="3"/><path d="M10 9.5v5l4.5-2.5z"/></svg>',
  gauge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/><path d="M4 18h2M18 18h2M12 10V8"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="8" width="12" height="12" rx="2.5"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>',
  edit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16z"/><path d="M13.5 6.5l4 4"/></svg>',
};

// ---------------------------------------------------------------------------- state
const STEPS = [
  { id: "face", t: "使う顔", icon: I.face, title: "どの顔に差し替えますか？", desc: "新しく映したい顔を選ぶか、登録します。" },
  { id: "media", t: "素材", icon: I.film, title: "素材を読み込む", desc: "差し替えたい動画・写真を選びます。" },
  { id: "who", t: "置き換える人", icon: I.target, title: "誰を置き換えますか？", desc: "プレビュー上の顔をクリックして選びます。" },
  { id: "look", t: "仕上がり", icon: I.sparkle, title: "仕上がりを確認", desc: "元と差し替え後を見比べて調整します。" },
  { id: "export", t: "書き出し", icon: I.export, title: "書き出す", desc: "形式を選んでファイルに保存します。" },
];
const TYPE_LABEL = { self: "自分自身", consented_person: "出演者（同意あり）", synthetic: "AI 生成" };
const TYPE_SHORT = { self: "自分自身", consented_person: "出演者", synthetic: "AI 生成" };
const QUALITY = [["high", "高画質"], ["standard", "標準"], ["light", "軽量"]];
const ENCODER = [["auto", "自動"], ["software", "ソフトウェア"], ["hardware", "ハードウェア"]];
const DETAIL = [[128, "速さ優先"], [256, "高画質"], [512, "最高画質"]];
const MATTE = [["", "なし"], ["luma", "白黒マスク"], ["alpha", "透明付き素材"]];

const S = {
  env: null, project: null, identities: [], identity: null,
  target: null, frame: 0, step: 0,
  who: "pick", faces: [], facesFrame: -1, picked: [],
  look: { detail: 256, keep_front: true, sharpen: false, blend: 0.12, color: 0.5, strictness: 0.4, smoothing: 0.5, enhance_blend: 0.8 },
  out: { codec: "h264", quality: "standard", encoder: "auto", matte: "", watermark: false, in: null, outp: null, path: "" },
  preview: null, previewKey: "", view: "frame", split: 50,
  busy: "", busyKind: "", job: null, result: null, enter: true, liveShown: 0, facesFresh: false, justToggled: -1,
};

// ---------------------------------------------------------------------------- api
async function api(path, opts = {}) {
  const headers = { "X-FSJ-Token": TOKEN, ...(opts.headers || {}) };
  let body = opts.body;
  if (body && !(body instanceof FormData)) { headers["Content-Type"] = "application/json"; body = JSON.stringify(body); }
  const res = await fetch(path, { method: opts.method || (body ? "POST" : "GET"), headers, body });
  const data = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(data?.error || `エラーが発生しました (${res.status})`);
  return data;
}
async function attempt(fn) {
  try { return await fn(); } catch (e) { toast(e.message, "err"); return undefined; }
}

// ---------------------------------------------------------------------------- toasts
function toast(msg, kind = "info", ms = 6000) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `${kind === "ok" ? I.check : kind === "err" ? I.alert : I.info}<div>${esc(msg)}</div>`;
  $("#toasts").append(el);
  setTimeout(() => { el.style.transition = "opacity .3s"; el.style.opacity = "0"; setTimeout(() => el.remove(), 300); }, ms);
}

// ---------------------------------------------------------------------------- timecode
function tcFrames(tc, fps) {
  const drop = /[;,]/.test(tc);
  const [h, m, s, f] = tc.split(/[:;,.]/).map(Number);
  const nom = Math.round(fps);
  let frames = (h * 3600 + m * 60 + s) * nom + f;
  if (drop) { const d = nom === 30 ? 2 : 4; const tm = h * 60 + m; frames -= d * (tm - Math.floor(tm / 10)); }
  return frames;
}
function framesTc(frames, fps, drop) {
  const nom = Math.round(fps);
  if (drop) {
    const d = nom === 30 ? 2 : 4, per10 = nom * 600 - d * 9, perMin = nom * 60 - d;
    const tens = Math.floor(frames / per10), mod = frames % per10;
    frames += mod > d ? d * 9 * tens + d * Math.floor((mod - d) / perMin) : d * 9 * tens;
  }
  const p = (n) => String(n).padStart(2, "0");
  return `${p(Math.floor(frames / (nom * 3600)) % 24)}:${p(Math.floor(frames / (nom * 60)) % 60)}:${p(Math.floor(frames / nom) % 60)}${drop ? ";" : ":"}${p(frames % nom)}`;
}
function tcAt(n) {
  const t = S.target;
  if (!t || t.kind !== "video") return "";
  if (t.timecode) return framesTc(tcFrames(t.timecode, t.fps) + n, t.fps, /[;,]/.test(t.timecode));
  return framesTc(n, t.fps, false);
}
const fmtSec = (s) => s == null ? "–" : s < 60 ? `${Math.round(s)} 秒` : `${Math.floor(s / 60)} 分 ${Math.round(s % 60)} 秒`;

// ---------------------------------------------------------------------------- helpers
const cos = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const pickedIndex = (emb) => S.picked.findIndex((p) => cos(p.emb, emb) >= 0.6);
const frameUrl = (n, w = 1600) => withToken(`/api/targets/${S.target.id}/frame?n=${n}&w=${w}`);
const currentFormat = () => S.env.formats.find((f) => f.name === S.out.codec) || S.env.formats[0];
const stepDone = [
  () => !!S.identity,
  () => !!S.target,
  () => S.who === "all" || S.picked.length > 0,
  () => !!S.preview,
  () => !!S.result,
];
function payload(extra = {}) {
  return {
    project: S.project, identity: S.identity, target: S.target?.id, frame: S.frame,
    who: S.who, picked: S.picked.map((p) => p.emb), ...S.look,
    codec: S.out.codec, quality: S.out.quality, encoder: S.out.encoder, matte: S.out.matte || null,
    watermark: S.out.watermark, in_frame: S.out.in ?? 0, out_frame: S.out.outp, output: S.out.path || null, ...extra,
  };
}
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

// ---------------------------------------------------------------------------- render: rail
function renderRail() {
  const subs = [
    S.identities.find((i) => i.id === S.identity)?.label || "未選択",
    S.target ? S.target.name : "未選択",
    S.who === "all" ? "全員" : S.picked.length ? `${S.picked.length} 人` : "未選択",
    S.preview ? "確認済み" : "–",
    S.result ? "完了" : S.job ? "書き出し中…" : "–",
  ];
  $("#rail").innerHTML = `
    <div class="rail-title">ワークフロー</div>
    <div class="steps" style="--p:${S.step / (STEPS.length - 1)}">
    ${STEPS.map((st, i) => `
      <button class="step ${i === S.step ? "active" : ""} ${stepDone[i]() ? "done" : ""}" data-step="${i}">
        <span class="badge">${st.icon}</span>
        <span><div class="t">${i + 1}. ${st.t}</div><div class="s">${esc(subs[i])}</div></span>
      </button>`).join("")}
    </div>
    <div class="grow"></div>
    <button class="rail-link" data-open="batch">${I.layers}<span>まとめて処理</span></button>
    <button class="rail-link" data-open="log">${I.list}<span>同意の記録</span></button>
    <button class="rail-link" data-open="perf">${I.gauge}<span>速度チェック</span></button>
    <div class="rail-foot">${I.lock}<span>v${esc(S.env?.version || "")} · すべてローカルで処理</span></div>`;
  $$(".step", $("#rail")).forEach((b) => b.onclick = () => goStep(+b.dataset.step));
  $$("[data-open]", $("#rail")).forEach((b) => b.onclick = () => ({ batch: openBatch, log: openLog, perf: openPerf })[b.dataset.open]());
}

function goStep(i) {
  if (i !== S.step) S.enter = true;
  S.step = i;
  if (STEPS[i].id === "who" && S.target && S.facesFrame !== S.frame) findFaces();
  if (STEPS[i].id === "look" && S.target && !S.preview) runPreview();
  if (STEPS[i].id === "look" || STEPS[i].id === "export") { if (S.preview) S.view = "compare"; } else S.view = "frame";
  renderAll();
}

// ---------------------------------------------------------------------------- render: viewer
function renderViewer() {
  const v = $("#viewer");
  if (S.env?.missing_models?.length) {
    v.innerHTML = `<div class="drop"><div><div class="icon">${I.export}</div>
      <h2>はじめに準備が必要です</h2><p>顔の差し替えに使う AI のファイル（約 1.2 GB）をダウンロードします。最初の 1 回だけです。</p>
      <div class="row"><button class="btn primary" id="dl">ダウンロードする</button></div></div></div>`;
    $("#dl").onclick = downloadModels;
    return;
  }
  if (!S.target) {
    v.innerHTML = `<div class="drop" id="drop"><div>
      <div class="icon">${I.upload}</div>
      <h2>動画・写真をここにドロップ</h2>
      <p>mov / mp4 / mxf / png / jpg など。素材はこのパソコンから外に出ません。</p>
      <div class="row"><button class="btn primary" id="pick-file">${I.folder} ファイルを選ぶ</button></div>
    </div></div>`;
    $("#pick-file").onclick = () => $("#file-target").click();
    bindDrop($("#drop"));
    return;
  }
  const t = S.target;
  const j = S.job;
  if (j && j.kind === "render" && j.live > 0) {
    S.liveShown = j.live;
    v.innerHTML = `
      <div class="canvas" id="canvas" style="aspect-ratio:${t.width}/${t.height}">
        <img class="layer base live-frame" id="live-img" src="${withToken(`/api/jobs/${j.id}/live?v=${j.live}`)}" alt="">
        <div class="hud tl"><span class="rec"></span>RENDERING</div>
        <div class="hud tr" id="hud-frames"></div>
        <div class="hud bl" id="hud-tc"></div>
        <div class="hud br" id="hud-rate"></div>
        <div class="live-bar" id="live-bar"></div>
      </div>`;
    fitCanvas();
    updateLive(j);
    return;
  }
  const hasPrev = !!S.preview;
  const view = hasPrev ? S.view : "frame";
  const base = view === "frame" ? frameUrl(S.frame) : view === "after" ? withToken(S.preview.after) : view === "matte" ? withToken(S.preview.matte) : withToken(S.preview.before);
  const showFaces = STEPS[S.step].id === "who" && view === "frame";
  v.innerHTML = `
    ${hasPrev ? `<div class="view-tabs">
      ${[["frame", "元の映像"], ["compare", "比較"], ["after", "差し替え後"], ["matte", "差し替え範囲"]].map(([k, l]) =>
        `<button data-view="${k}" class="${view === k ? "on" : ""}">${l}</button>`).join("")}</div>` : ""}
    <div class="canvas ${S.enter ? "intro" : ""} ${S.facesFresh ? "fresh" : ""}" id="canvas" style="aspect-ratio:${t.width}/${t.height}; --split:${S.split}%">
      <img class="layer base" src="${base}" alt="">
      ${view === "compare" ? `<img class="layer after" src="${withToken(S.preview.after)}" alt="">
        <div class="split-handle" id="split"></div><span class="split-tag l">元</span><span class="split-tag r">差し替え後</span>` : ""}
      ${showFaces ? S.faces.map((f) => {
        const on = S.who === "all" || pickedIndex(f.emb) >= 0;
        const [x0, y0, x1, y1] = f.box;
        return `<div class="facebox ${on ? "on" : ""} ${f.i === S.justToggled ? "just" : ""} ${y0 < 0.07 ? "top" : ""}" data-face="${f.i}" style="left:${x0 * 100}%;top:${y0 * 100}%;width:${(x1 - x0) * 100}%;height:${(y1 - y0) * 100}%;animation-delay:${f.i * 60}ms"><span class="n">${String(f.i + 1).padStart(2, "0")}</span></div>`;
      }).join("") : ""}
      ${S.busyKind ? `<div class="fx ${S.busyKind}"></div><div class="fx-label"><div class="spinner"></div>${esc(S.busy)}</div>` : ""}
    </div>
    ${S.busy && !S.busyKind ? `<div class="busy"><div class="chip"><div class="spinner"></div>${esc(S.busy)}</div></div>` : ""}`;
  fitCanvas();
  S.facesFresh = false; S.justToggled = -1;
  $$("[data-view]", v).forEach((b) => b.onclick = () => { S.view = b.dataset.view; renderViewer(); });
  $$(".facebox", v).forEach((b) => b.onclick = (e) => { e.stopPropagation(); toggleFace(+b.dataset.face); });
  const canvas = $("#canvas");
  if (view === "compare") {
    const move = (e) => {
      const r = canvas.getBoundingClientRect();
      S.split = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100));
      canvas.style.setProperty("--split", S.split + "%");
    };
    canvas.onpointerdown = (e) => { canvas.setPointerCapture(e.pointerId); move(e); canvas.onpointermove = move; };
    canvas.onpointerup = () => { canvas.onpointermove = null; };
  }
}
function updateLive(j) {
  const t = S.target;
  if (!$("#live-img")) { renderViewer(); return; }
  if (j.live !== S.liveShown) {
    S.liveShown = j.live;
    const next = new Image();
    next.onload = () => { const img = $("#live-img"); if (img) img.src = next.src; };
    next.src = withToken(`/api/jobs/${j.id}/live?v=${j.live}`);
  }
  const pad = (n) => String(n).padStart(5, "0");
  $("#hud-frames").textContent = `F ${pad(j.done)} / ${pad(j.total)}`;
  $("#hud-tc").textContent = tcAt(j.live_frame || 0);
  $("#hud-rate").textContent = j.done ? `${j.fps.toFixed(1)} FPS · ETA ${fmtSec(j.eta)}` : "PREPARING";
  $("#live-bar").style.width = `${j.total ? (j.done / j.total) * 100 : 0}%`;
}
function fitCanvas() {
  const v = $("#viewer"), c = $("#canvas");
  if (!c || !S.target) return;
  const live = S.job && S.job.kind === "render" && S.job.live > 0;
  const pad = 44, top = S.preview && !live ? 56 : 0;
  const W = v.clientWidth - pad, H = v.clientHeight - pad - top;
  const ar = S.target.width / S.target.height;
  const w = Math.min(W, H * ar);
  c.style.width = `${w}px`;
  c.style.height = `${w / ar}px`;
  c.style.marginTop = `${top}px`;
}
window.addEventListener("resize", () => { fitCanvas(); renderTimeline(); });

// ---------------------------------------------------------------------------- render: timeline
function renderTimeline() {
  const tl = $("#timeline");
  const t = S.target;
  if (!t || t.kind !== "video") { tl.classList.add("hidden"); return; }
  tl.classList.remove("hidden");
  const n = Math.max(6, Math.min(18, Math.floor((tl.clientWidth || 900) / 110)));
  const thumbs = Array.from({ length: n }, (_, i) => Math.round((i + 0.5) * (t.frames / n)));
  const pct = (f) => (t.frames > 1 ? (f / (t.frames - 1)) * 100 : 0);
  const inF = S.out.in, outF = S.out.outp;
  tl.innerHTML = `
    <div class="tl-controls">
      <button class="btn sm icon-btn" id="tl-prev" title="1 フレーム戻る (←)">${I.prev}</button>
      <span class="tc" id="tl-tc">${tcAt(S.frame)}</span>
      <button class="btn sm icon-btn" id="tl-next" title="1 フレーム進む (→)">${I.next}</button>
      <span class="frameno num" id="tl-fn">${S.frame + 1} / ${t.frames}</span>
      <span class="spacer"></span>
      <span class="hint">範囲指定</span>
      <button class="btn sm" id="tl-in" title="ここから書き出す (I)">IN <span class="kbd">I</span></button>
      <button class="btn sm" id="tl-out" title="ここまで書き出す (O)">OUT <span class="kbd">O</span></button>
      ${inF != null || outF != null ? `<button class="btn sm ghost" id="tl-clear">範囲をクリア</button>` : ""}
    </div>
    <div class="track">
      <div class="strip"><div class="thumbs">${thumbs.map((f) => `<img src="${frameUrl(f, 200)}" alt="" loading="lazy">`).join("")}</div></div>
      ${t.audio?.length ? `<div class="wave"><img src="${withToken(`/api/targets/${t.id}/waveform`)}" alt="" onerror="this.parentElement.remove()"></div>` : ""}
      ${inF != null || outF != null ? `<div class="range" style="left:${pct(inF ?? 0)}%;right:${100 - pct(outF ?? t.frames - 1)}%"></div>` : ""}
      <div class="playhead" id="tl-head" style="left:${pct(S.frame)}%"></div>
      <input type="range" id="tl-range" min="0" max="${t.frames - 1}" value="${S.frame}" aria-label="フレーム">
    </div>`;
  $("#tl-range").oninput = (e) => setFrame(+e.target.value, true);
  $("#tl-range").onchange = (e) => setFrame(+e.target.value);
  $("#tl-prev").onclick = () => setFrame(S.frame - 1);
  $("#tl-next").onclick = () => setFrame(S.frame + 1);
  $("#tl-in").onclick = () => setIn();
  $("#tl-out").onclick = () => setOut();
  const clr = $("#tl-clear");
  if (clr) clr.onclick = () => { S.out.in = S.out.outp = null; renderTimeline(); renderInspector(); };
}
function updatePlayhead() {
  const t = S.target;
  if (!t) return;
  const pct = t.frames > 1 ? (S.frame / (t.frames - 1)) * 100 : 0;
  const h = $("#tl-head"); if (h) h.style.left = pct + "%";
  const tc = $("#tl-tc"); if (tc) tc.textContent = tcAt(S.frame);
  const fn = $("#tl-fn"); if (fn) fn.textContent = `${S.frame + 1} / ${t.frames}`;
}
const setIn = () => { S.out.in = S.frame; if (S.out.outp != null && S.out.outp < S.frame) S.out.outp = null; renderTimeline(); renderInspector(); };
const setOut = () => { S.out.outp = S.frame; if (S.out.in != null && S.out.in > S.frame) S.out.in = null; renderTimeline(); renderInspector(); };

const afterFrameChange = debounce(() => {
  const id = STEPS[S.step].id;
  if (id === "who") findFaces();
  if (id === "look") runPreview();
}, 350);
function setFrame(n, scrubbing = false) {
  if (!S.target) return;
  S.frame = Math.max(0, Math.min(S.target.frames - 1, n));
  updatePlayhead();
  if (S.view !== "frame" && S.preview) S.view = "frame";
  const img = $("#canvas img.base");
  if (img && S.view === "frame") img.src = frameUrl(S.frame);
  if (scrubbing) return;
  if (STEPS[S.step].id === "who") { S.faces = []; renderViewer(); }
  afterFrameChange();
}

function openKeys() {
  modal(`<div class="modal-head"><div><div class="eyebrow">SHORTCUTS</div><h2>キーボード操作</h2></div>
    <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
    <div class="modal-body"><div class="keys">
      <span><span class="kbd">←</span> <span class="kbd">→</span></span><span>1 フレーム移動</span>
      <span><span class="kbd">Shift</span> + <span class="kbd">←</span> <span class="kbd">→</span></span><span>10 フレーム移動</span>
      <span><span class="kbd">I</span> / <span class="kbd">O</span></span><span>書き出す範囲の開始 / 終了</span>
      <span><span class="kbd">1</span> 〜 <span class="kbd">5</span></span><span>ステップを切り替え</span>
      <span><span class="kbd">C</span></span><span>元の映像と比較を切り替え（プレビュー後）</span>
      <span><span class="kbd">?</span></span><span>この一覧</span>
    </div></div>`);
}
document.addEventListener("keydown", (e) => {
  if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName) || $("#modal").open) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "?") openKeys();
  if (/^[1-5]$/.test(e.key)) goStep(+e.key - 1);
  if ((e.key === "c" || e.key === "C") && S.preview) { S.view = S.view === "compare" ? "frame" : "compare"; renderViewer(); }
  if (e.key === "ArrowLeft") { setFrame(S.frame - (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  if (e.key === "ArrowRight") { setFrame(S.frame + (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  if (e.key === "i" || e.key === "I") setIn();
  if (e.key === "o" || e.key === "O") setOut();
});

// ---------------------------------------------------------------------------- render: inspector
function renderInspector() {
  const st = STEPS[S.step];
  const body = { face: inspFace, media: inspMedia, who: inspWho, look: inspLook, export: inspExport }[st.id]();
  const nextLabel = ["次へ：素材", "次へ：置き換える人", "次へ：仕上がり", "次へ：書き出し", null][S.step];
  $("#inspector").classList.toggle("enter", S.enter);
  S.enter = false;
  $("#inspector").innerHTML = `
    <div class="insp-head"><div class="eyebrow">STEP ${String(S.step + 1).padStart(2, "0")} — 05</div><h1>${st.title}</h1><p>${st.desc}</p></div>
    <div class="insp-body">${body.html}</div>
    <div class="insp-foot">
      ${S.step > 0 ? `<button class="btn ghost" id="back">${I.prev} 戻る</button>` : ""}
      <span style="flex:1"></span>
      ${nextLabel ? `<button class="btn primary" id="next">${nextLabel} ${I.next}</button>`
        : S.job ? `<button class="btn danger" id="cancel-job">${I.x} 書き出しを中止</button>`
        : `<button class="btn primary" id="render">${I.export} 書き出す</button>`}
    </div>`;
  const back = $("#back"); if (back) back.onclick = () => goStep(S.step - 1);
  const next = $("#next");
  if (next) next.onclick = () => {
    const need = [
      [!S.identity, "使う顔を選ぶか、新しく登録してください"],
      [!S.target, "素材を読み込んでください"],
      [S.who === "pick" && !S.picked.length, "置き換える人の顔をクリックして選んでください"],
      [false, ""],
    ][S.step];
    if (need && need[0]) return toast(need[1], "err");
    goStep(S.step + 1);
  };
  body.bind?.();
  const render = $("#render"); if (render) render.onclick = startRender;
  const cj = $("#cancel-job"); if (cj) cj.onclick = () => attempt(() => api(`/api/jobs/${S.job.id}/cancel`, { method: "POST" }));
}

function inspFace() {
  const cards = S.identities.map((i) => `
    <button class="face-card ${i.id === S.identity ? "on" : ""}" data-id="${esc(i.id)}">
      <img src="${withToken(i.thumb)}" alt="">
      <div class="meta"><b>${esc(i.label)}</b><small>${esc(i.person)}</small>
        <div style="margin-top:6px"><span class="tag">${esc(TYPE_LABEL[i.type] || i.type)}</span></div></div>
    </button>`).join("");
  return {
    html: `<div class="section"><div class="label">顔ライブラリ</div>
      <div class="lib">${cards}
        <button class="face-card add" id="add-face"><div class="plus">${I.plus}</div><b>新しい顔を登録</b></button>
      </div></div>
      ${S.identities.length ? "" : `<div class="note">${I.info}<div>まだ顔が登録されていません。「新しい顔を登録」から始めましょう。</div></div>`}`,
    bind() {
      $$(".face-card[data-id]").forEach((b) => b.onclick = () => { S.identity = b.dataset.id; S.preview = null; renderAll(); });
      $("#add-face").onclick = openRegister;
    },
  };
}

function inspMedia() {
  const t = S.target;
  const info = t ? `
    <div class="card"><dl class="kv">
      <dt>ファイル</dt><dd>${esc(t.name)}</dd>
      <dt>種類</dt><dd>${t.kind === "video" ? "動画" : "写真"}</dd>
      <dt>解像度</dt><dd>${t.width} × ${t.height}</dd>
      ${t.kind === "video" ? `
      <dt>フレームレート</dt><dd>${t.fps.toFixed(3)} fps <span class="muted">(${esc(t.fps_exact)})</span></dd>
      <dt>長さ</dt><dd>${t.duration.toFixed(2)} 秒・${t.frames} フレーム</dd>
      <dt>タイムコード</dt><dd class="mono">${esc(t.timecode || "なし")}</dd>
      <dt>コーデック</dt><dd>${esc(t.codec)} · ${esc(t.pix_fmt)}</dd>
      <dt>音声</dt><dd>${t.audio.length ? esc(t.audio.join(", ")) : "なし"}</dd>` : ""}
    </dl></div>
    ${t.vfr ? `<div class="note warn">${I.alert}<div>フレームレートが一定ではない動画です（スマホ撮影に多い）。音ズレする場合は、編集ソフトで一定のフレームレートに変換してから使ってください。</div></div>` : ""}
    ${t.interlaced ? `<div class="note warn">${I.alert}<div>インターレースの素材です。先にインターレース解除をするときれいに仕上がります。</div></div>` : ""}` : "";
  return {
    html: `
      <div class="section"><div class="label">ファイル</div>
        <button class="btn block" id="choose">${I.folder} ${t ? "別のファイルを選ぶ" : "ファイルを選ぶ"}</button>
        <div class="hint">プレビューの枠にドラッグ＆ドロップしても読み込めます。</div></div>
      <div class="section"><div class="label">大きなファイル（数 GB）はパスで指定</div>
        <div style="display:flex;gap:8px"><input class="input" id="path" placeholder="/Users/you/Movies/clip.mov">
        <button class="btn" id="load-path">読み込む</button></div>
        <div class="hint">Mac：Finder でファイルを右クリック →「option」を押しながら「パス名をコピー」</div></div>
      ${info}`,
    bind() {
      $("#choose").onclick = () => $("#file-target").click();
      const go = () => loadPath($("#path").value);
      $("#load-path").onclick = go;
      $("#path").onkeydown = (e) => { if (e.key === "Enter") go(); };
    },
  };
}

function inspWho() {
  const facesHtml = S.faces.length ? `<div class="people">${S.faces.map((f) => {
    const on = S.who === "all" || pickedIndex(f.emb) >= 0;
    return `<button class="person ${on ? "on" : ""}" data-face="${f.i}"><img src="${f.crop}" alt=""><span>${on ? "✓ " : ""}${f.i + 1} 番</span></button>`;
  }).join("")}</div>` : `<div class="hint">${S.busyKind === "scan" ? "顔を探しています…" : S.target ? "この場面では顔が見つかりませんでした。タイムラインで別の場面に移動してください。" : "先に素材を読み込んでください。"}</div>`;
  return {
    html: `
      <div class="seg" id="who">
        <button data-who="pick" class="${S.who === "pick" ? "on" : ""}">選んだ人だけ</button>
        <button data-who="all" class="${S.who === "all" ? "on" : ""}">映っている全員</button>
      </div>
      <div class="section"><div class="label">この場面の顔</div>${facesHtml}
        <div class="hint">プレビュー上の顔をクリックしても選べます。タイムラインで場面を動かすと、その場面の顔を探し直します。</div></div>
      ${S.who === "pick" ? `<div class="section"><div class="label">置き換える人（${S.picked.length} 人）</div>
        ${S.picked.length ? `<div class="people">${S.picked.map((p, i) => `<button class="person on" data-unpick="${i}" title="クリックで外す"><img src="${p.crop}" alt=""><span>外す</span></button>`).join("")}</div>`
          : `<div class="hint">まだ選ばれていません。</div>`}
        <div class="hint">選んだ人は、動画全体で自動的に追いかけます。</div></div>` : ""}`,
    bind() {
      $$("[data-who]").forEach((b) => b.onclick = () => { S.who = b.dataset.who; S.preview = null; renderAll(); });
      $$(".person[data-face]").forEach((b) => b.onclick = () => toggleFace(+b.dataset.face));
      $$("[data-unpick]").forEach((b) => b.onclick = () => { S.picked.splice(+b.dataset.unpick, 1); S.preview = null; renderAll(); });
    },
  };
}

function slider(key, label, min, max, step, left, right) {
  return `<div class="slider"><div class="top"><b>${label}</b><span class="num muted" id="v-${key}">${S.look[key]}</span></div>
    <input type="range" class="rng" data-look="${key}" min="${min}" max="${max}" step="${step}" value="${S.look[key]}">
    <div class="ends"><span>${left}</span><span>${right}</span></div></div>`;
}
function sw(key, title, sub, obj = "look") {
  const val = obj === "look" ? S.look[key] : S.out[key];
  return `<label class="switch"><span class="txt"><b>${title}</b><small>${sub}</small></span>
    <input type="checkbox" data-${obj}="${key}" ${val ? "checked" : ""}><span class="tog"></span></label>`;
}

function inspLook() {
  return {
    html: `
      <div class="field"><span>顔の解像度</span><div class="seg" id="detail">${DETAIL.map(([k, l]) => `<button data-detail="${k}" class="${S.look.detail === k ? "on" : ""}">${l}</button>`).join("")}</div>
        <small class="muted">${{ 128: "AI の素の解像度（128px）。顔が小さい映像ならこれで十分", 256: "おすすめ。顔のアップでも細部が残ります（約 4 倍の処理）", 512: "4K の顔アップ向け。とても遅くなります（約 16 倍の処理）" }[S.look.detail]}</small></div>
      ${sw("keep_front", "手や髪を顔の前に残す", "手・髪・小道具が顔に重なる場面向け（おすすめ）")}
      ${sw("sharpen", "顔をくっきり補正", "解像感を上げます。処理は遅くなります")}
      ${slider("blend", "境目のなじませ", 0.04, 0.3, 0.01, "くっきり", "なめらか")}
      ${slider("color", "肌の色を周りに合わせる", 0, 1, 0.05, "弱い", "強い")}
      <details class="more"><summary>詳細設定</summary><div class="inner">
        ${slider("strictness", "人物判定の厳しさ", 0.25, 0.7, 0.01, "ゆるい（選んだ人が置き換わらない時）", "厳しい（別人まで置き換わる時）")}
        ${slider("smoothing", "動きのなめらかさ", 0, 1, 0.05, "速い動きに追従", "揺れを抑える")}
        ${S.look.sharpen ? slider("enhance_blend", "補正の強さ", 0, 1, 0.05, "弱い", "強い") : ""}
      </div></details>
      <button class="btn block" id="refresh">${I.sparkle} この場面でプレビューを更新</button>
      <div class="hint">設定を変えると自動でプレビューを作り直します。タイムラインで場面を変えて確認できます。</div>`,
    bind() {
      bindLookControls();
      $("#refresh").onclick = runPreview;
    },
  };
}
const previewSoon = debounce(() => runPreview(), 650);
function bindLookControls() {
  $$("[data-detail]").forEach((el) => el.onclick = () => { S.look.detail = +el.dataset.detail; renderInspector(); previewSoon(); });
  $$("[data-look]").forEach((el) => {
    const key = el.dataset.look;
    if (el.type === "checkbox") el.onchange = () => { S.look[key] = el.checked; renderInspector(); previewSoon(); };
    else {
      el.oninput = () => { S.look[key] = +el.value; $(`#v-${key}`).textContent = el.value; };
      el.onchange = () => previewSoon();
    }
  });
}

function inspExport() {
  const f = currentFormat();
  const quick = [["h264", I.share, "確認・共有用", "H.264 (.mp4)。どこでも再生でき、ファイルが小さい"],
    ["prores422hq", I.edit, "編集用の高画質", "ProRes 422 HQ (.mov)。DaVinci Resolve / Premiere 向け"]];
  const t = S.target;
  const rangeText = t?.kind === "video" && (S.out.in != null || S.out.outp != null)
    ? `${tcAt(S.out.in ?? 0)} → ${tcAt(S.out.outp ?? t.frames - 1)}（${(S.out.outp ?? t.frames - 1) - (S.out.in ?? 0) + 1} フレーム）`
    : "全体";
  const r = S.result;
  const settingsHtml = `
      <div class="section"><div class="label">何に使いますか？</div>
        <div class="formats">${quick.map(([k, ic, b, s]) => `<button class="fmt ${S.out.codec === k ? "on" : ""}" data-codec="${k}">${ic}<b>${b}</b><small>${s}</small></button>`).join("")}</div></div>
      <details class="more" ${quick.some(([k]) => k === S.out.codec) ? "" : "open"}><summary>形式を細かく選ぶ</summary><div class="inner">
        <label class="field"><span>コーデック</span><select class="input" id="codec">
          ${S.env.formats.map((x) => `<option value="${x.name}" ${x.name === S.out.codec ? "selected" : ""}>${esc(x.label)}</option>`).join("")}</select></label>
        ${f.has_quality ? `<div class="field"><span>画質</span><div class="seg">${QUALITY.map(([k, l]) => `<button data-quality="${k}" class="${S.out.quality === k ? "on" : ""}">${l}</button>`).join("")}</div></div>` : ""}
        <div class="field"><span>エンコーダ</span><div class="seg">${ENCODER.map(([k, l]) => `<button data-encoder="${k}" class="${S.out.encoder === k ? "on" : ""}" ${k === "hardware" && !f.hardware.length ? "disabled title='この環境では使えません'" : ""}>${l}</button>`).join("")}</div>
          <div class="hint">使えるエンコーダ：${esc([...f.software, ...f.hardware.map((h) => h + "（HW）")].join(", ") || "なし")}</div></div>
        <div class="hint">解像度・フレームレート・タイムコード・音声は、どの形式でも元の素材と同じになります。</div>
      </div></details>
      <div class="section"><div class="label">範囲</div>
        <div class="card" style="display:flex;justify-content:space-between;align-items:center;gap:10px">
          <span class="mono" style="font-size:12.5px">${esc(rangeText)}</span>
          ${t?.kind === "video" ? `<span class="hint">タイムラインで <span class="kbd">I</span> <span class="kbd">O</span></span>` : ""}</div></div>
      ${sw("watermark", "「AI face-swapped」の文字を入れる", "画面の隅に小さく表示します", "out")}
      <details class="more"><summary>編集ソフトで仕上げる人向け</summary><div class="inner">
        <div class="field"><span>差し替え範囲のマスクも書き出す</span><div class="seg">${MATTE.map(([k, l]) => `<button data-matte="${k}" class="${S.out.matte === k ? "on" : ""}">${l}</button>`).join("")}</div>
          <div class="hint">本編と同じタイムコードで「〜_matte.mov」として書き出します。DaVinci Resolve などで本編に重ねて修正できます。</div></div>
        <label class="field"><span>保存先（空欄なら作品フォルダの renders）</span><input class="input" id="outpath" value="${esc(S.out.path)}" placeholder="自動"></label>
      </div></details>
      <div class="note">${I.info}<div>書き出したファイルには「AI face-swapped」という情報が自動で記録されます。</div></div>`;
  const resultHtml = r ? `<div class="result"><div class="head">${I.check} 書き出しが終わりました</div>
        <div class="name">${esc(r.output.split("/").pop())}</div>
        ${r.playable ? `<video src="${fileUrl(r.output)}" controls preload="metadata"></video>` : r.image ? `<img src="${fileUrl(r.output)}" alt="">` : ""}
        <div class="path">${esc(r.output)}</div>
        ${r.matte ? `<div class="path">マスク：${esc(r.matte)}</div>` : ""}
        ${r.frames > 1 ? `<div class="hint">${r.frames} フレーム · ${fmtSec(r.seconds)}</div>` : ""}
        ${(r.warnings || []).map((w) => `<div class="note warn">${I.alert}<div>${esc(w)}</div></div>`).join("")}
        <div style="display:flex;gap:8px"><button class="btn sm" id="reveal">${I.folder} Finder で表示</button></div></div>` : "";
  return {
    html: resultHtml + settingsHtml,
    bind() {
      $$("[data-codec]").forEach((b) => b.onclick = () => { S.out.codec = b.dataset.codec; renderInspector(); });
      $("#codec").onchange = (e) => { S.out.codec = e.target.value; if (S.out.encoder === "hardware" && !currentFormat().hardware.length) S.out.encoder = "auto"; renderInspector(); };
      $$("[data-quality]").forEach((b) => b.onclick = () => { S.out.quality = b.dataset.quality; renderInspector(); });
      $$("[data-encoder]").forEach((b) => b.onclick = () => { S.out.encoder = b.dataset.encoder; renderInspector(); });
      $$("[data-matte]").forEach((b) => b.onclick = () => { S.out.matte = b.dataset.matte; renderInspector(); });
      $$("[data-out]").forEach((el) => el.onchange = () => { S.out[el.dataset.out] = el.checked; });
      $("#outpath").onchange = (e) => { S.out.path = e.target.value.trim(); };
      const rv = $("#reveal"); if (rv) rv.onclick = () => attempt(() => api("/api/reveal", { body: { path: r.output } }));
    },
  };
}

function renderAll() {
  renderRail();
  renderViewer();
  renderTimeline();
  renderInspector();
  renderJob();
}

// ---------------------------------------------------------------------------- actions
async function loadProjects(select) {
  const env = await api("/api/state");
  S.env = env;
  S.project = select || (env.projects.includes(S.project) ? S.project : env.projects[0]);
  $("#project").innerHTML = env.projects.map((p) => `<option ${p === S.project ? "selected" : ""}>${esc(p)}</option>`).join("");
  const prov = env.providers[0] || "CPU";
  $("#device-pill").innerHTML = `${I.chipIcon}${esc(prov.replace("ExecutionProvider", ""))}`;
  await loadIdentities();
}
async function loadIdentities() {
  S.identities = (await attempt(() => api(`/api/projects/${encodeURIComponent(S.project)}/identities`))) || [];
  if (!S.identities.some((i) => i.id === S.identity)) S.identity = null;
}

async function loadTarget(promise) {
  S.busy = "素材を読み込んでいます…"; renderViewer();
  const t = await attempt(() => promise);
  S.busy = "";
  if (t) {
    Object.assign(S, { target: t, frame: 0, faces: [], facesFrame: -1, preview: null, result: null, view: "frame" });
    S.out.in = S.out.outp = null;
    toast(`「${t.name}」を読み込みました`, "ok", 3000);
  }
  renderAll();
}
const loadPath = (p) => p && p.trim() && loadTarget(api("/api/targets/path", { body: { path: p.trim() } }));
function loadFile(file) {
  const fd = new FormData();
  fd.append("project", S.project);
  fd.append("file", file);
  loadTarget(api("/api/targets/upload", { body: fd }));
}
$("#file-target").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) loadFile(f); };
function bindDrop(el) {
  el.ondragover = (e) => { e.preventDefault(); el.classList.add("over"); };
  el.ondragleave = () => el.classList.remove("over");
  el.ondrop = (e) => { e.preventDefault(); el.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) loadFile(f); };
}
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0] && !$("#modal").open) loadFile(e.dataTransfer.files[0]); });

let facesSeq = 0;
async function findFaces() {
  if (!S.target) return;
  const seq = ++facesSeq, frame = S.frame;
  S.busy = "顔を探しています…"; S.busyKind = "scan"; renderViewer();
  const r = await attempt(() => api(`/api/targets/${S.target.id}/faces`, { body: { frame, project: S.project } }));
  if (seq !== facesSeq) return;
  S.busy = ""; S.busyKind = "";
  if (r) {
    S.faces = r.faces; S.facesFrame = r.frame; S.facesFresh = true;
    if (S.faces.length === 1 && !S.picked.length && S.who === "pick") { S.picked.push({ emb: S.faces[0].emb, crop: S.faces[0].crop }); S.preview = null; }
  }
  renderAll();
}
function toggleFace(i) {
  const f = S.faces.find((x) => x.i === i);
  if (!f) return;
  if (S.who === "all") S.who = "pick";
  const k = pickedIndex(f.emb);
  if (k >= 0) S.picked.splice(k, 1); else { S.picked.push({ emb: f.emb, crop: f.crop }); S.justToggled = i; }
  S.preview = null;
  renderAll();
}

let previewSeq = 0;
async function runPreview() {
  if (!S.target || !S.identity || (S.who === "pick" && !S.picked.length)) return;
  const seq = ++previewSeq;
  S.busy = "プレビューを作成しています…"; S.busyKind = S.target ? "shimmer" : ""; renderViewer();
  const r = await attempt(() => api("/api/preview", { body: payload() }));
  if (seq !== previewSeq) return;
  S.busy = ""; S.busyKind = "";
  if (r) {
    S.preview = r; S.view = "compare";
    if (!r.faces) toast("この場面には置き換える人が映っていません。タイムラインで別の場面を選んでください。", "info");
  }
  renderAll();
}

async function startRender() {
  const r = await attempt(() => api("/api/render", { body: payload() }));
  if (!r) return;
  S.job = r; S.result = null;
  renderAll();
  pollJob(r.id, (job) => {
    if (job.status === "done") { S.result = job.result; toast("書き出しが終わりました", "ok"); }
    else if (job.status === "error") toast(job.message, "err", 10000);
    else if (job.status === "cancelled") toast("書き出しを中止しました", "info");
  });
}
function pollJob(id, onEnd) {
  const tick = async () => {
    const j = await attempt(() => api(`/api/jobs/${id}`));
    if (!j) { S.job = null; renderAll(); return; }
    S.job = j;
    if (["done", "error", "cancelled"].includes(j.status)) { S.job = null; onEnd(j); renderAll(); return; }
    renderJob();
    if (j.kind === "render" && j.live > 0) updateLive(j);
    setTimeout(tick, 600);
  };
  tick();
}
function renderJob() {
  const bar = $("#jobbar");
  const j = S.job;
  // the live render view shows progress on the canvas itself; the floating bar is only for other jobs
  if (!j || (j.kind === "render" && j.live > 0)) { bar.classList.add("hidden"); return; }
  bar.classList.remove("hidden");
  const pct = j.total ? Math.min(100, (j.done / j.total) * 100) : 0;
  const stats = j.total && j.done ? `${j.done} / ${j.total} フレーム · ${j.fps} fps · 残り ${fmtSec(j.eta)}` : j.message || "準備中…";
  bar.innerHTML = `
    <div class="title"><div class="spinner"></div>${esc(j.kind === "download" ? j.title : j.kind === "bench" ? "速度チェック中" : j.kind === "batch" ? "まとめて処理中：" + j.title : "書き出し中：" + j.title)}</div>
    <button class="btn sm danger" id="job-cancel">中止</button>
    <div class="bar ${j.done ? "" : "indet"}"><i style="width:${pct}%"></i></div>
    <div class="stats num">${esc(stats)}</div>`;
  $("#job-cancel").onclick = () => attempt(() => api(`/api/jobs/${j.id}/cancel`, { method: "POST" }));
}

async function downloadModels() {
  const r = await attempt(() => api("/api/models/download", { method: "POST" }));
  if (!r) return;
  S.job = r; renderJob();
  pollJob(r.id, async (job) => {
    if (job.status === "done") { toast("準備ができました", "ok"); await loadProjects(); renderAll(); }
    else if (job.status === "error") toast(job.message, "err", 10000);
  });
}

// ---------------------------------------------------------------------------- modals
function modal(html, bind) {
  const m = $("#modal");
  m.innerHTML = `<div class="modal">${html}</div>`;
  m.showModal();
  $$("[data-close]", m).forEach((b) => b.onclick = () => m.close());
  bind?.(m);
}

function openRegister() {
  let files = [];
  modal(`
    <div class="modal-head"><div><div class="eyebrow">新しい顔</div><h2>顔を登録する</h2>
      <p>差し替えに使う顔の写真と、使ってよいことの確認を記録します。記録はこの作品のフォルダにだけ保存されます。</p></div>
      <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
    <div class="modal-body">
      <div class="field"><span>顔の写真（1〜5 枚）</span>
        <div class="photo-drop" id="photos"><div class="empty">ここに写真をドロップ、またはクリックして選択<br><small>本人だけが写った、明るく正面に近い写真がおすすめ。角度の違う写真を数枚入れると安定します。</small></div></div>
        <input type="file" id="photo-input" accept="image/*" multiple class="hidden"></div>
      <div class="grid2">
        <label class="field"><span>呼び名</span><input class="input" id="r-label" placeholder="例：スタント A"></label>
        <label class="field"><span>本人の氏名</span><input class="input" id="r-person" placeholder="例：山田 太郎（AI の顔なら作成者名）"></label>
      </div>
      <div class="grid2">
        <div class="field"><span>誰の顔ですか？</span><div class="seg" id="r-type">
          ${Object.entries(TYPE_SHORT).map(([k, l], i) => `<button data-type="${k}" class="${i === 0 ? "on" : ""}">${l}</button>`).join("")}</div></div>
        <label class="field"><span>同意した日</span><input type="date" class="input" id="r-date" value="${S.env.today}" max="${S.env.today}"></label>
      </div>
      <label class="check"><input type="checkbox" id="r-attest"><span><b>この顔を使う権利があることを確認しました</b>
        <small>本人の同意を得ている／自分自身の顔／AI で作った架空の顔</small></span></label>
      <details class="more"><summary>同意書を添付する（任意）</summary><div class="inner">
        <input type="file" id="r-doc" class="input" style="padding-top:7px"></div></details>
    </div>
    <div class="modal-foot"><button class="btn ghost" data-close>キャンセル</button><button class="btn primary" id="r-save">${I.check} 登録する</button></div>`,
  (m) => {
    const drop = $("#photos", m), input = $("#photo-input", m);
    const show = () => {
      drop.innerHTML = files.length ? files.map((f) => `<img class="ph" src="${URL.createObjectURL(f)}" alt="">`).join("") + `<span class="hint">クリックで追加</span>`
        : drop.innerHTML;
    };
    const add = (list) => { files = [...files, ...[...list].filter((f) => f.type.startsWith("image/"))].slice(0, 5); show(); };
    drop.onclick = () => input.click();
    input.onchange = () => add(input.files);
    drop.ondragover = (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.add("over"); };
    drop.ondragleave = () => drop.classList.remove("over");
    drop.ondrop = (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.remove("over"); add(e.dataTransfer.files); };
    let type = "self";
    $$("[data-type]", m).forEach((b) => b.onclick = () => { type = b.dataset.type; $$("[data-type]", m).forEach((x) => x.classList.toggle("on", x === b)); });
    $("#r-save", m).onclick = async () => {
      if (!files.length) return toast("顔の写真を 1 枚以上追加してください", "err");
      const fd = new FormData();
      files.forEach((f) => fd.append("images", f));
      fd.append("label", $("#r-label", m).value);
      fd.append("person", $("#r-person", m).value);
      fd.append("consent_date", $("#r-date", m).value);
      fd.append("source_type", type);
      fd.append("attested", $("#r-attest", m).checked ? "true" : "false");
      const doc = $("#r-doc", m).files[0];
      if (doc) fd.append("consent_doc", doc);
      const btn = $("#r-save", m);
      btn.disabled = true; btn.innerHTML = `<div class="spinner"></div> 登録中…`;
      const r = await attempt(() => api(`/api/projects/${encodeURIComponent(S.project)}/identities`, { body: fd }));
      btn.disabled = false; btn.innerHTML = `${I.check} 登録する`;
      if (!r) return;
      m.close();
      await loadIdentities();
      S.identity = r.id; S.preview = null;
      toast("顔を登録しました", "ok");
      if (r.warning) toast(r.warning + "。同じ人の写真か確認してください。", "info", 9000);
      renderAll();
    };
  });
}

function openNewProject() {
  modal(`
    <div class="modal-head"><div><div class="eyebrow">作品</div><h2>新しい作品</h2><p>作品ごとに、登録した顔・記録・書き出したファイルを分けて保存します。</p></div>
      <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
    <div class="modal-body"><label class="field"><span>作品名</span><input class="input" id="p-name" placeholder="例：短編映画_スタントシーン" autofocus></label></div>
    <div class="modal-foot"><button class="btn ghost" data-close>キャンセル</button><button class="btn primary" id="p-save">作成</button></div>`,
  (m) => {
    const save = async () => {
      const r = await attempt(() => api("/api/projects", { body: { name: $("#p-name", m).value } }));
      if (!r) return;
      m.close();
      resetForProject();
      await loadProjects(r.project);
      renderAll();
    };
    $("#p-save", m).onclick = save;
    $("#p-name", m).onkeydown = (e) => { if (e.key === "Enter") save(); };
  });
}

function openBatch() {
  modal(`
    <div class="modal-head"><div><div class="eyebrow">まとめて処理</div><h2>フォルダ内の素材をまとめて処理</h2>
      <p>今の<b>使う顔・置き換える人・仕上がり・書き出し形式</b>の設定で、フォルダ内の動画と写真をすべて処理します。</p></div>
      <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
    <div class="modal-body">
      <label class="field"><span>フォルダの場所</span><input class="input" id="b-dir" placeholder="/Users/you/Movies/shoot_day1"></label>
      <div class="note">${I.info}<div>読めないファイルは飛ばして続け、最後に結果をまとめて表示します。</div></div>
      <div id="b-result"></div>
    </div>
    <div class="modal-foot"><button class="btn ghost" data-close>閉じる</button><button class="btn primary" id="b-run">${I.layers} 開始</button></div>`,
  (m) => {
    $("#b-run", m).onclick = async () => {
      const r = await attempt(() => api("/api/batch", { body: payload({ folder: $("#b-dir", m).value }) }));
      if (!r) return;
      m.close();
      S.job = r; renderJob();
      pollJob(r.id, (job) => {
        if (job.status === "done") {
          const items = job.result.items || [];
          const ok = items.filter((x) => x.ok).length;
          toast(`まとめて処理：${ok} / ${items.length} 件が完了しました`, ok === items.length ? "ok" : "info", 9000);
          modal(`<div class="modal-head"><div><div class="eyebrow">まとめて処理</div><h2>${ok} / ${items.length} 件が完了</h2><p class="mono">${esc(job.result.folder)}</p></div>
            <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
            <div class="modal-body"><table class="log"><tr><th>ファイル</th><th>結果</th><th>メモ</th></tr>
            ${items.map((x) => `<tr><td>${esc(x.source)}</td><td class="${x.ok ? "status-ok" : "status-err"}">${x.ok ? "完了" : "失敗"}</td><td>${esc(x.error || "")}</td></tr>`).join("")}</table></div>`);
        } else if (job.status === "error") toast(job.message, "err", 10000);
      });
    };
  });
}

async function openLog() {
  const r = await attempt(() => api(`/api/projects/${encodeURIComponent(S.project)}/log`));
  if (!r) return;
  const names = { project_created: "作品を作成", identity_registered: "顔を登録", render: "書き出し" };
  modal(`
    <div class="modal-head"><div><div class="eyebrow">同意の記録</div><h2>${esc(S.project)}</h2>
      <p>${r.problems.length ? `<span class="status-err">記録が書き換えられています：${esc(r.problems.join("; "))}</span>` : `<span class="status-ok">✓ 記録は改ざんされていません</span>`}</p></div>
      <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
    <div class="modal-body"><table class="log"><tr><th>日時（UTC）</th><th>内容</th><th>詳細</th></tr>
      ${r.entries.map((e) => {
        const d = e.data;
        const detail = e.event === "render" ? `${d.person_name || ""} → ${(d.output || "").split("/").pop()}` : d.person_name || d.name || "";
        return `<tr><td class="mono">${esc(e.timestamp.replace("T", " ").slice(0, 19))}</td><td>${esc(names[e.event] || e.event)}</td><td>${esc(detail)}</td></tr>`;
      }).join("")}</table></div>`);
}

// ---------------------------------------------------------------------------- performance check
const perf = { frames: 48, keep_front: null, sharpen: null, last: null };
function openPerf() {
  const t = S.target;
  if (perf.keep_front === null) { perf.keep_front = S.look.keep_front; perf.sharpen = S.look.sharpen; }
  const ready = t && t.kind === "video";
  const providers = (S.env?.providers || []).map((p) => p.replace("ExecutionProvider", ""));
  modal(`
    <div class="modal-head"><div><div class="eyebrow">PERFORMANCE</div><h2>速度チェック</h2>
      <p>このパソコンでどれくらいの速さで処理できるかを測ります。読み込んだ動画の一部を実際に処理し、ファイルは残しません。</p></div>
      <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
    <div class="modal-body">
      <div class="card"><dl class="kv">
        <dt>使える実行環境</dt><dd>${esc(providers.join(" → ") || "–")}</dd>
        <dt>測る素材</dt><dd>${ready ? esc(t.name) + `（${t.width}×${t.height}）` : "未選択"}</dd>
      </dl></div>
      ${ready ? `
      <div class="field"><span>測るフレーム数</span><div class="seg" id="pf-frames">
        ${[24, 48, 96].map((n) => `<button data-n="${n}" class="${perf.frames === n ? "on" : ""}">${n} フレーム</button>`).join("")}</div>
        <div class="hint">多いほど正確ですが、時間がかかります。</div></div>
      <label class="switch"><span class="txt"><b>手や髪を顔の前に残す</b><small>仕上がり設定と同じ条件で測ります</small></span>
        <input type="checkbox" id="pf-front" ${perf.keep_front ? "checked" : ""}><span class="tog"></span></label>
      <label class="switch"><span class="txt"><b>顔をくっきり補正</b><small>オンにすると遅くなります</small></span>
        <input type="checkbox" id="pf-sharp" ${perf.sharpen ? "checked" : ""}><span class="tog"></span></label>`
      : `<div class="note warn">${I.alert}<div>先に「2. 素材」で、顔が映っている動画を読み込んでください。</div></div>`}
      ${perf.last ? `<button class="btn ghost sm" id="pf-last" style="align-self:flex-start">前回の結果を見る</button>` : ""}
    </div>
    <div class="modal-foot"><button class="btn ghost" data-close>閉じる</button>
      ${ready ? `<button class="btn primary" id="pf-run">${I.gauge} 測定を開始</button>` : `<button class="btn primary" id="pf-media">素材を選ぶ</button>`}</div>`,
  (m) => {
    $$("[data-n]", m).forEach((b) => b.onclick = () => { perf.frames = +b.dataset.n; $$("[data-n]", m).forEach((x) => x.classList.toggle("on", x === b)); });
    const fr = $("#pf-front", m); if (fr) fr.onchange = () => { perf.keep_front = fr.checked; };
    const sh = $("#pf-sharp", m); if (sh) sh.onchange = () => { perf.sharpen = sh.checked; };
    const last = $("#pf-last", m); if (last) last.onclick = () => showPerf(perf.last);
    const media = $("#pf-media", m); if (media) media.onclick = () => { m.close(); goStep(1); };
    const run = $("#pf-run", m);
    if (run) run.onclick = async () => {
      if (S.job) return toast("ほかの処理が終わってから測定してください", "err");
      const r = await attempt(() => api("/api/bench", { body: { target: S.target.id, frames: perf.frames, keep_front: perf.keep_front, sharpen: perf.sharpen, detail: S.look.detail } }));
      if (!r) return;
      m.close();
      S.job = r; renderAll();
      pollJob(r.id, (job) => {
        if (job.status === "done") { perf.last = job.result; showPerf(job.result); }
        else if (job.status === "error") toast(job.message, "err", 10000);
      });
    };
  });
}

function showPerf(r) {
  const stages = Object.entries(r.timings).filter(([k]) => k !== "swap_total" && r.timings[k].ms_per_frame >= 0.5);
  const max = Math.max(...stages.map(([, v]) => v.ms_per_frame), 1);
  const providerClass = (p) => (/CPU/.test(p) ? "warn" : "ok");
  const swapOnCpu = r.runs_on.swapper && /CPU/.test(r.runs_on.swapper) && r.providers.some((p) => !/CPU/.test(p));
  modal(`
    <div class="modal-head"><div><div class="eyebrow">PERFORMANCE</div><h2>測定結果</h2><p>${esc(r.machine)}</p></div>
      <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
    <div class="modal-body">
      <div class="perf-hero">
        <div><div class="perf-num num">${r.fps}<small> fps</small></div><div class="hint">1 秒あたりに処理できるフレーム数</div></div>
        <div><div class="perf-num num">${r.minutes_per_minute ?? "–"}<small> 分</small></div><div class="hint">1 分の素材（${r.source_fps} fps）にかかる目安</div></div>
      </div>
      <div class="section"><div class="label">AI が動いている場所</div>
        <div class="chips">${Object.entries(r.runs_on).map(([k, v]) => `<span class="chip-p ${providerClass(v)}">${esc(k)} · ${esc(v)}</span>`).join("")}</div>
        ${swapOnCpu ? `<div class="note warn">${I.alert}<div>顔の差し替え AI が CPU で動いています。この結果をコピーして送ってもらえれば、原因を調べられます。</div></div>` : ""}</div>
      <div class="section"><div class="label">時間がかかっている処理（1 フレームあたり）</div>
        <div class="bars">${stages.map(([k, v]) => `
          <div class="bar-row"><span>${esc(r.labels[k] || k)}</span>
            <div class="bar-track"><i style="width:${(v.ms_per_frame / max) * 100}%"></i></div>
            <span class="num">${v.ms_per_frame} ms</span></div>`).join("")}</div></div>
      <div class="hint">${r.resolution} · ${r.frames} フレームを ${r.seconds} 秒で処理 · モデル読み込み ${r.load_s} 秒</div>
    </div>
    <div class="modal-foot"><button class="btn ghost" id="pf-again">もう一度測る</button>
      <button class="btn primary" id="pf-copy">${I.copy} 結果をコピー</button></div>`,
  (m) => {
    $("#pf-again", m).onclick = () => openPerf();
    $("#pf-copy", m).onclick = async () => {
      try { await navigator.clipboard.writeText(r.text); toast("結果をコピーしました。チャットに貼り付けて送ってください", "ok"); }
      catch { const ta = document.createElement("textarea"); ta.value = r.text; m.append(ta); ta.select(); document.execCommand("copy"); ta.remove(); toast("結果をコピーしました", "ok"); }
    };
  });
}

function resetForProject() {
  Object.assign(S, { identity: null, preview: null, result: null });
}

// ---------------------------------------------------------------------------- boot
$("#project").onchange = async (e) => { resetForProject(); S.project = e.target.value; await loadIdentities(); renderAll(); };
$("#new-project").onclick = openNewProject;
$("#keys-btn").onclick = openKeys;
$("#device-pill").onclick = () => openPerf();
$("#quit-btn").onclick = () => {
  modal(`<div class="modal-head"><div><div class="eyebrow">QUIT</div><h2>faceswapjp を終了しますか？</h2>
      <p>${S.job ? "<b>処理中のものは中止されます。</b>" : ""}もう一度使うときは、アプリ（または起動用ファイル）をダブルクリックしてください。</p></div>
      <button class="btn ghost icon-btn" data-close>${I.x}</button></div>
    <div class="modal-foot"><button class="btn ghost" data-close>キャンセル</button><button class="btn primary" id="quit-yes">終了する</button></div>`,
  (m) => {
    $("#quit-yes", m).onclick = async () => {
      await attempt(() => api("/api/shutdown", { method: "POST" }));
      m.close();
      document.body.innerHTML = `<div style="height:100vh;display:grid;place-items:center;text-align:center;color:#b4b4c2;font:15px/1.7 var(--font)">
        <div><div class="brand-mark" style="width:48px;height:48px;margin:0 auto 18px;border-radius:14px"></div>
        <div style="font-size:20px;color:#ededf2;font-weight:650">faceswapjp を終了しました</div>このタブは閉じてかまいません。</div></div>`;
    };
  });
};
(async () => {
  await attempt(() => loadProjects());
  if (!S.env) return;
  renderAll();
})();
