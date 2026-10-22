// Manufactured hook/stream/promise fixtures only: no DOM, camera or network.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
// @ts-expect-error Node fs exists in the test runtime; this browser project omits Node declarations.
import { readFileSync } from "node:fs";
const hooks = vi.hoisted(() => ({ states: [] as any[], refs: [] as any[], effects: [] as any[],
  pending: [] as any[], s: 0, r: 0, e: 0 }));
vi.mock("react", () => ({
  useState(initial: any) {
    const i = hooks.s++;
    if (!(i in hooks.states)) hooks.states[i] = initial;
    return [hooks.states[i], (value: any) => { hooks.states[i] = typeof value === "function" ? value(hooks.states[i]) : value; }];
  },
  useRef(initial: any) { const i = hooks.r++; return hooks.refs[i] ||= { current: initial }; },
  useEffect(fn: any, deps: any[]) {
    const i = hooks.e++, old = hooks.effects[i];
    if (!old || deps.some((x, n) => !Object.is(x, old.deps[n]))) { hooks.pending = hooks.pending.filter(x => x.i !== i); hooks.pending.push({ i, fn, deps }); }
  },
}));
vi.mock("recharts", () => ({ LineChart: "chart", Line: "line", XAxis: "x", YAxis: "y", Tooltip: "tooltip", ResponsiveContainer: "container" }));
import App from "./App";
import { DEMO_FIXTURES, DEMO_LABEL } from "./demoFixtures";
const deferred = <T,>() => {
  let resolve!: (x: T) => void, reject!: (x: any) => void;
  const promise = new Promise<T>((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
};
const response = (extra: any = {}) => ({ selected_model: "fixture-model", ethics_notice: "fixture ethics",
  face: { face_found: true, box: null, aus: [], quality: 1, expression: { probs: { happiness: 1, sadness: 0, anger: 0, surprise: 0, fear: 0, disgust: 0, neutral: 0, unknown: 0 }, confidence: 1,
    uncertainty: 0, source: "fixture-source" }, insufficient_evidence: { insufficient: false, reasons: [] }, ...extra } });
const ok = (j = response()) => ({ ok: true, json: vi.fn().mockResolvedValue(j) });
const stream = () => { const track = { stop: vi.fn() }; return { track, getTracks: () => [track] }; };
let storage: any, idb: any, cacheStore: any;
let tree: any, video: any, canvas: any, media: any, fetcher: any, timers: Map<number, () => void>, seq: number;
function render(commit = true) {
  hooks.s = hooks.r = hooks.e = 0; tree = App();
  hooks.refs[0].current = video; hooks.refs[1].current = canvas;
  if (commit) for (const { i, fn, deps } of hooks.pending.splice(0)) {
    hooks.effects[i]?.cleanup?.(); hooks.effects[i] = { deps, cleanup: fn() };
  }
}
function nodes(n: any): any[] {
  if (!n || typeof n !== "object") return [];
  if (Array.isArray(n)) return n.flatMap(x => nodes(x));
  return [n, ...nodes(n.props?.children)];
}
function text(n: any): string {
  if (n == null || typeof n === "boolean") return "";
  if (typeof n !== "object") return String(n);
  return Array.isArray(n) ? n.map(text).join("") : text(n.props?.children);
}
function select(i: number, value: string, commit = true) {
  nodes(tree).filter(n => n.type === "select")[i].props.onChange({ target: { value } }); render(commit);
}
function toggle(commit = true) { nodes(tree).find(n => n.type === "button").props.onClick(); render(commit); }
async function settle() { render(); for (let i = 0; i < 16; i++) await Promise.resolve(); render(); }
async function next() { const callbacks = [...timers.values()]; timers.clear(); callbacks.forEach(fn => fn()); await settle(); }
const chart = () => nodes(tree).find(n => n.type === "chart").props.data;
beforeEach(() => {
  hooks.states = []; hooks.refs = []; hooks.effects = []; hooks.pending = [];
  timers = new Map(); seq = 0;
  video = { videoWidth: 320, srcObject: null };
  canvas = { getContext: vi.fn(() => ({ drawImage: vi.fn() })), toBlob: (cb: any) => cb(new Blob(["manufactured pixels"])) };
  media = vi.fn().mockRejectedValue(new Error("unconfigured synthetic camera"));
  fetcher = vi.fn().mockRejectedValue(new Error("unmocked network denied"));
  const forbiddenStorage = () => { throw new Error("unmocked storage forbidden"); };
  storage = Object.fromEntries(["getItem", "setItem", "removeItem", "clear", "key"].map(key => [key, vi.fn(forbiddenStorage)]));
  idb = { open: vi.fn(forbiddenStorage), deleteDatabase: vi.fn(forbiddenStorage) };
  cacheStore = { open: vi.fn(forbiddenStorage), match: vi.fn(forbiddenStorage) };
  vi.stubGlobal("localStorage", storage); vi.stubGlobal("sessionStorage", storage);
  vi.stubGlobal("indexedDB", idb); vi.stubGlobal("caches", cacheStore);
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: media } });
  vi.stubGlobal("window", { localStorage: storage, sessionStorage: storage, indexedDB: idb, caches: cacheStore, setTimeout: vi.fn((fn: any) => { timers.set(++seq, fn); return seq; }), clearTimeout: vi.fn((id: number) => timers.delete(id)) });
  vi.stubGlobal("fetch", fetcher); render();
});
afterEach(() => { hooks.effects.forEach(e => e.cleanup?.()); vi.unstubAllGlobals(); vi.clearAllMocks(); });
describe("current dashboard run", () => {
  it("A1 clears old probabilities/history on failure and recovers on current success", async () => {
    media.mockResolvedValue(stream()); fetcher.mockResolvedValueOnce(ok()).mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({ detail: "fixture unavailable" }) }).mockResolvedValueOnce(ok());
    toggle(); await settle(); expect(text(tree)).toContain("fixture-source"); expect(chart()).toHaveLength(1);
    await next(); expect(text(tree)).toContain("Expression (no current result)"); expect(chart()).toEqual([]);
    expect(nodes(tree).find(n => n.props?.role === "alert").props.children).toContain("Inference unavailable");
    expect(text(tree)).not.toContain("fixture-source"); await next(); expect(text(tree)).toContain("fixture-source");
    expect(nodes(tree).some(n => n.props?.role === "alert")).toBe(false);
  });
  it("A2 ignores deferred old JSON after switch and binds accepted labels to captured profile", async () => {
    const old = deferred<any>(); media.mockResolvedValue(stream()); fetcher.mockResolvedValueOnce({ ok: true, json: () => old.promise }).mockResolvedValueOnce(ok()).mockResolvedValueOnce(ok());
    toggle(); await settle(); select(1, "legacy", false);
    expect(text(tree)).toContain("no current result"); expect(chart()).toEqual([]);
    old.resolve(response()); await settle(); expect(text(tree)).toContain("fixture-model/legacy"); expect(chart()).toHaveLength(1);
    select(0, "vit"); await settle(); expect(text(tree)).toContain("fixture-model/n/a");
  });
  it("A2 Stop aborts a pending response and ignores a late result", async () => {
    const pending = deferred<any>(), s = stream(); media.mockResolvedValue(s); fetcher.mockReturnValue(pending.promise);
    toggle(); await settle(); const signal = fetcher.mock.calls[0][1].signal;
    toggle(false); expect(signal.aborted).toBe(true); expect(s.track.stop).toHaveBeenCalledTimes(1);
    pending.resolve(ok()); await settle(); expect(text(tree)).toContain("Stopped"); expect(text(tree)).not.toContain("fixture-source");
    expect(chart()).toEqual([]); expect(timers.size).toBe(0);
  });
  it("A3 stops every late-acquired track without attachment, upload or timer", async () => {
    const pending = deferred<any>(), s = stream(); media.mockReturnValue(pending.promise);
    toggle(); toggle(); pending.resolve(s); await settle();
    expect(s.track.stop).toHaveBeenCalledTimes(1); expect(video.srcObject).toBeNull(); expect(fetcher).not.toHaveBeenCalled(); expect(timers.size).toBe(0); expect(text(tree)).toContain("Stopped");
  });
  it("A4 old acquisition cannot detach or stop the new stream", async () => {
    const a = deferred<any>(), b = deferred<any>(), old = stream(), current = stream();
    media.mockReturnValueOnce(a.promise).mockReturnValueOnce(b.promise); fetcher.mockResolvedValue(ok());
    toggle(); select(1, "legacy"); b.resolve(current); await settle(); a.resolve(old); await settle();
    expect(old.track.stop).toHaveBeenCalledTimes(1); expect(current.track.stop).not.toHaveBeenCalled(); expect(video.srcObject).toBe(current);
    toggle(); expect(current.track.stop).toHaveBeenCalledTimes(1); expect(video.srcObject).toBeNull();
  });
  it("A4 ignores an obsolete acquisition rejection after restart", async () => {
    const a = deferred<any>(), current = stream(); media.mockReturnValueOnce(a.promise).mockResolvedValueOnce(current); fetcher.mockResolvedValue(ok());
    toggle(); toggle(); toggle(); await settle(); a.reject(new Error("old failure")); await settle();
    expect(video.srcObject).toBe(current); expect(text(tree)).toContain("Stop webcam"); expect(text(tree)).not.toContain("old failure");
  });
  it("A4 deferred blob is never uploaded after Stop", async () => {
    let deliver!: (b: Blob) => void; canvas.toBlob = (cb: any) => { deliver = cb; }; media.mockResolvedValue(stream());
    toggle(); await settle(); toggle(); deliver(new Blob(["fixture"])); await settle();
    expect(fetcher).not.toHaveBeenCalled(); expect(timers.size).toBe(0); expect(text(tree)).toContain("Stopped");
  });
  it("direct cleanup is idempotent and ignores obsolete fetch rejection", async () => {
    const pending = deferred<any>(), s = stream(); media.mockResolvedValue(s); fetcher.mockReturnValue(pending.promise);
    toggle(); await settle(); hooks.effects[0].cleanup(); hooks.effects[0].cleanup(); pending.reject(new Error("late error")); await settle();
    expect(s.track.stop).toHaveBeenCalledTimes(1); expect(text(tree)).not.toContain("late error"); expect(timers.size).toBe(0);
  });
  it("current camera denial shows an error and returns to Start", async () => {
    media.mockRejectedValueOnce(new Error("SYNTHETIC permission denied"));
    toggle(); await settle();
    expect(nodes(tree).find(n => n.type === "button").props.children).toEqual(["Start", " webcam"]);
    expect(nodes(tree).find(n => n.props?.role === "status").props.children).toBe("Camera unavailable");
    expect(nodes(tree).find(n => n.props?.role === "alert").props.children).toContain("Camera unavailable");
    expect(text(tree)).toContain("SYNTHETIC permission denied");
    expect(video.srcObject).toBeNull(); expect(fetcher).not.toHaveBeenCalled(); expect(timers.size).toBe(0); expect(chart()).toEqual([]);
  });
  it("missing video element reports camera unavailable and stops acquired tracks", async () => {
    const s = stream(); media.mockResolvedValueOnce(s); video = null; render();
    toggle(); await settle();
    expect(text(tree)).toContain("Start webcam");
    expect(nodes(tree).find(n => n.props?.role === "status").props.children).toBe("Camera unavailable");
    expect(nodes(tree).find(n => n.props?.role === "alert").props.children).toContain("Camera unavailable");
    expect(text(tree)).toContain("video element unavailable");
    expect(s.track.stop).toHaveBeenCalledTimes(1); expect(fetcher).not.toHaveBeenCalled(); expect(timers.size).toBe(0); expect(chart()).toEqual([]);
  });
  it("null blob clears the current display and reports frame capture unavailable", async () => {
    const s = stream(); media.mockResolvedValue(s); fetcher.mockResolvedValue(ok());
    toggle(); await settle(); expect(text(tree)).toContain("fixture-source"); expect(chart()).toHaveLength(1);
    canvas.toBlob = (cb: any) => cb(null); await next();
    expect(nodes(tree).find(n => n.props?.role === "alert").props.children).toContain("frame capture unavailable");
    expect(text(tree)).toContain("no current result"); expect(text(tree)).not.toContain("fixture-source"); expect(chart()).toEqual([]);
    expect(fetcher).toHaveBeenCalledTimes(1); expect(timers.size).toBe(1);
  });
  it("Stop clears an already scheduled timer and any queued callback stays inert", async () => {
    const s = stream(); media.mockResolvedValue(s); fetcher.mockResolvedValue(ok());
    toggle(); await settle(); expect(timers.size).toBe(1);
    const [id, queued] = [...timers.entries()][0];
    toggle(false);
    expect(window.clearTimeout).toHaveBeenCalledWith(id); expect(timers.size).toBe(0);
    expect(s.track.stop).toHaveBeenCalledTimes(1); expect(video.srcObject).toBeNull();
    queued(); await settle();
    expect(fetcher).toHaveBeenCalledTimes(1); expect(timers.size).toBe(0); expect(chart()).toEqual([]);
    expect(nodes(tree).find(n => n.props?.role === "status").props.children).toBe("Stopped");
  });
  it("A5 preserves defaults, query rules and inspection-only evidence", async () => {
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok(response({ face_found: false, expression: { probs: { happiness: 0, sadness: 0, anger: 0, surprise: 0, fear: 0, disgust: 0, neutral: 0, unknown: 1 }, confidence: .15, uncertainty: .85, source: "fixture-source" }, abstained: true, abstain_threshold: .7, insufficient_evidence: { insufficient: true, reasons: ["fixture no face"] } })));
    expect(media).not.toHaveBeenCalled(); toggle(); await settle();
    expect(fetcher.mock.calls[0][0]).toBe("http://127.0.0.1:8000/api/infer?model=mini-xception&profile=corrected");
    expect(text(tree)).toContain("inspection only"); expect(text(tree)).toContain("not an emotion reading"); expect(text(tree)).toContain("fixture no face"); expect(text(tree)).toContain("fixture ethics"); expect(text(tree)).toContain("fixture-source"); expect(text(tree)).toContain("Model score unavailable"); expect(chart()).toEqual([]);
    nodes(tree).find(n => n.type === "input").props.onChange({ target: { checked: true } }); render(); await settle();
    expect(fetcher.mock.calls[fetcher.mock.calls.length - 1][0]).toContain("&calibrated=true");
    select(1, "legacy"); await settle(); expect(fetcher.mock.calls[fetcher.mock.calls.length - 1][0]).not.toContain("calibrated=true");
    select(2, "0.7"); await settle(); expect(fetcher.mock.calls[fetcher.mock.calls.length - 1][0]).toContain("&abstain=0.7");
    expect(nodes(tree).find(n => n.props?.role === "status").props["aria-live"]).toBe("polite");
  });
});


describe("U1 evidence-state rendering (hook fixtures, not DOM proof)", () => {
  it("shows requesting/waiting categories and keeps live status text stable on numeric updates", async () => {
    const acquisition = deferred<any>(), pending = deferred<any>(); media.mockReturnValue(acquisition.promise); fetcher.mockReturnValue(pending.promise);
    toggle(); expect(text(nodes(tree).find(n => n.props?.role === "status"))).toBe("Requesting camera");
    acquisition.resolve(stream()); await settle(); expect(text(nodes(tree).find(n => n.props?.role === "status"))).toBe("Waiting for current result");
    pending.resolve(ok()); await settle(); const category = text(nodes(tree).find(n => n.props?.role === "status"));
    const changed = response(); changed.face.expression.confidence = .8; changed.face.expression.uncertainty = .2;
    fetcher.mockResolvedValue(ok(changed)); await next();
    expect(text(nodes(tree).find(n => n.props?.role === "status"))).toBe(category);
    expect(chart()).toHaveLength(2);
  });
  it("invalid > no-face > abstained > insufficient; no placeholder or invalid score enters history", async () => {
    media.mockResolvedValue(stream());
    const noface = response({ face_found: false, abstained: true, insufficient_evidence: { insufficient: true, reasons: ["no-face-detected"] } });
    const invalid = response({ ...noface.face, expression: { ...noface.face.expression, probs: { happiness: null } } });
    fetcher.mockResolvedValueOnce(ok()).mockResolvedValueOnce(ok(invalid)).mockResolvedValueOnce(ok(noface));
    toggle(); await settle(); expect(chart()).toHaveLength(1);
    await next(); expect(text(tree)).toContain("Invalid response"); expect(chart()).toEqual([]);
    await next(); expect(text(nodes(tree).find(n => n.props?.role === "status"))).toBe("No face detected");
    expect(text(tree)).toContain("no expression inference ran"); expect(chart()).toEqual([]);
    fetcher.mockResolvedValueOnce(ok(response({ abstained: true, abstain_threshold: .7, insufficient_evidence: { insufficient: true, reasons: ["below-abstention-threshold"] } })));
    await next(); expect(text(nodes(tree).find(n => n.props?.role === "status"))).toBe("Abstained"); expect(chart()).toEqual([]);
    fetcher.mockResolvedValueOnce(ok(response({ insufficient_evidence: { insufficient: true, reasons: ["new-rule"] } })));
    await next(); expect(text(nodes(tree).find(n => n.props?.role === "status"))).toBe("Insufficient evidence");
    expect(text(tree)).toContain("Unknown evidence status: new-rule"); expect(chart()).toEqual([]);
  });
  it("missing confidence is unavailable and raw values are confined to labelled inspection details", async () => {
    media.mockResolvedValue(stream()); const missing: any = response(); delete missing.face.expression.confidence;
    fetcher.mockResolvedValue(ok(missing)); toggle(); await settle();
    expect(text(tree)).toContain("Model score unavailable"); expect(chart()).toEqual([]);
    const details = nodes(tree).find(n => n.type === "details");
    expect(text(details)).toContain("Inspection details"); expect(text(details)).toContain("Raw response");
    expect(text(details)).toContain("fixture-source");
  });
  it("recognises a model-not-run sentinel as unavailable before insufficient evidence", async () => {
    media.mockResolvedValue(stream());
    fetcher.mockResolvedValue(ok(response({ expression: { probs: { happiness: 0, sadness: 0, anger: 0, surprise: 0, fear: 0, disgust: 0, neutral: 0, unknown: 1 }, confidence: .15, uncertainty: .85, source: "hdf5 missing" }, insufficient_evidence: { insufficient: true, reasons: ["high-uncertainty"] } })));
    toggle(); await settle(); expect(text(nodes(tree).find(n => n.props?.role === "status"))).toBe("Inference unavailable");
    expect(chart()).toEqual([]); expect(text(tree)).toContain("not-run placeholder");
  });
  it("explicit class support displays unavailable, while unreported support retains legitimate zeros", async () => {
    media.mockResolvedValue(stream()); const explicit: any = response(); explicit.face.expression.supported_classes = ["happy"];
    fetcher.mockResolvedValueOnce(ok(explicit)).mockResolvedValueOnce(ok()); toggle(); await settle();
    expect(text(tree)).toContain("unavailable — unsupported class"); expect(chart()).toHaveLength(1);
    await next(); expect(text(tree)).toContain("support not reported");
    expect(text(tree)).not.toContain("unavailable — unsupported class");
    expect(nodes(tree).find(n => n.type === "details").props.open).toBeUndefined();
  });
});


const mode = (value: "live" | "demo", commit = true) => select(3, value, commit);
const statusText = () => text(nodes(tree).find(n => n.props?.role === "status"));
const noStorageCalls = () => {
  for (const spy of [...Object.values(storage), ...Object.values(idb), ...Object.values(cacheStore)]) expect(spy).not.toHaveBeenCalled();
};
describe("U2 explicit synthetic demo (hook fixtures only)", () => {
  it("defaults to stopped live; selecting Demo alone never starts camera or fixtures", async () => {
    expect(nodes(tree).filter(n => n.type === "select")[3].props.value).toBe("live");
    expect(statusText()).toBe("Stopped"); expect(text(tree)).not.toContain(DEMO_LABEL);
    mode("demo"); await settle();
    expect(statusText()).toBe("Stopped"); expect(text(tree)).toContain(DEMO_LABEL);
    expect(text(tree)).toContain("Demo stopped"); expect(timers.size).toBe(0); expect(chart()).toEqual([]);
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); noStorageCalls();
    expect(nodes(tree).filter(n => n.type === "select").slice(0, 3).every(n => n.props.disabled)).toBe(true);
  });
  it("steps all seven scenarios through U1 without camera/fetch/storage or measured timing", async () => {
    mode("demo"); toggle(); await settle();
    const expected = ["Current apparent-expression response", "No face detected", "Insufficient evidence", "Abstained", "Inference unavailable", "Current apparent-expression response", "Invalid response"];
    for (let i = 0; i < DEMO_FIXTURES.length; i++) {
      if (i) await next();
      expect(statusText()).toContain(expected[i]); expect(text(tree)).toContain(DEMO_LABEL);
      expect(text(tree)).toContain(DEMO_FIXTURES[i].label); expect(text(tree)).toContain("Demo timing: not measured");
      if ([1, 2, 3].includes(i)) expect(chart()).toHaveLength(1);
      if ([4, 6].includes(i)) expect(chart()).toEqual([]);
      if (i === 5) expect(text(tree)).toContain("unsupported class");
      expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); noStorageCalls();
    }
    expect(canvas.getContext).not.toHaveBeenCalled(); expect(video.srcObject).toBeNull(); expect(timers.size).toBe(1);
    await next(); expect(statusText()).toBe(expected[0]); // repeat manufactured stream, still no IO
    toggle(false); expect(timers.size).toBe(0); await settle();
    expect(statusText()).toBe("Stopped"); expect(text(tree)).toContain(DEMO_LABEL); expect(chart()).toEqual([]);
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); noStorageCalls();
  });
  it("mode switch aborts synchronously; late live JSON cannot replace a running synthetic fixture", async () => {
    const late = deferred<any>(), s = stream(); media.mockResolvedValue(s);
    fetcher.mockResolvedValue({ ok: true, json: () => late.promise });
    toggle(); await settle(); const signal = fetcher.mock.calls[0][1].signal;
    mode("demo", false);
    expect(signal.aborted).toBe(true); expect(s.track.stop).toHaveBeenCalledTimes(1); expect(video.srcObject).toBeNull();
    expect(chart()).toEqual([]); expect(text(tree)).toContain("no current result");
    toggle(); await settle(); expect(text(tree)).toContain("SYNTHETIC-demo/synthetic fixture");
    late.resolve(response()); await settle();
    expect(text(tree)).not.toContain("fixture-source"); expect(text(tree)).toContain(DEMO_LABEL);
    expect(chart()).toHaveLength(1); expect(media).toHaveBeenCalledTimes(1); expect(fetcher).toHaveBeenCalledTimes(1); noStorageCalls();
  });
  it("stops a late-acquired live stream after switching to running Demo without attachment/upload", async () => {
    const late = deferred<any>(), s = stream(); media.mockReturnValue(late.promise);
    toggle(); mode("demo"); toggle(); await settle(); late.resolve(s); await settle();
    expect(s.track.stop).toHaveBeenCalledTimes(1); expect(video.srcObject).toBeNull();
    expect(text(tree)).toContain("SYNTHETIC-demo/synthetic fixture"); expect(chart()).toHaveLength(1);
    expect(media).toHaveBeenCalledTimes(1); expect(fetcher).not.toHaveBeenCalled(); noStorageCalls();
  });
  it("a late live blob and queued live callbacks stay inert after mode cancellation", async () => {
    let deliver!: (blob: Blob) => void; const s = stream();
    canvas.toBlob = (cb: any) => { deliver = cb; }; media.mockResolvedValue(s);
    toggle(); await settle(); mode("demo", false); expect(s.track.stop).toHaveBeenCalledTimes(1);
    toggle(); await settle(); deliver(new Blob(["manufactured late blob"])); await settle();
    expect(fetcher).not.toHaveBeenCalled(); expect(text(tree)).toContain("SYNTHETIC-demo/synthetic fixture");
    expect(timers.size).toBe(1); expect(chart()).toHaveLength(1); noStorageCalls();
  });
  it("switching back is stopped, clears synthetic history, and cannot run a queued demo callback", async () => {
    select(0, "vit"); select(1, "legacy"); mode("demo"); toggle(); await settle();
    expect(chart()).toHaveLength(1); const queued = [...timers.values()][0];
    mode("live", false); expect(timers.size).toBe(0); expect(chart()).toEqual([]);
    queued(); await settle();
    expect(statusText()).toBe("Stopped"); expect(text(tree)).not.toContain(DEMO_LABEL);
    expect(text(tree)).not.toContain("SYNTHETIC-demo"); expect(video.srcObject).toBeNull();
    expect(nodes(tree).find(n => n.type === "button").props.children).toEqual(["Start", " webcam"]);
    const selectors = nodes(tree).filter(n => n.type === "select");
    expect(selectors[0].props.value).toBe("vit"); expect(selectors[1].props.value).toBe("legacy"); expect(selectors[2].props.value).toBe("off");
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); noStorageCalls();
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok()); toggle(); await settle();
    expect(media).toHaveBeenCalledTimes(1);
    expect(fetcher.mock.calls[0][0]).toBe("http://127.0.0.1:8000/api/infer?model=vit&profile=legacy");
  });
  it("an old scheduled live callback cannot resume uploads once Demo owns the timer", async () => {
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok()); toggle(); await settle();
    const queued = [...timers.values()][0]; mode("demo", false); expect(timers.size).toBe(0);
    toggle(); await settle(); queued(); await settle();
    expect(fetcher).toHaveBeenCalledTimes(1); expect(media).toHaveBeenCalledTimes(1);
    expect(text(tree)).toContain(DEMO_LABEL); expect(text(tree)).not.toContain("fixture-source");
    expect(timers.size).toBe(1); noStorageCalls();
  });
});


describe("H1 history omission", () => {
  it("retains prior points for no-face/abstained/insufficient, then appends recovery and clears on transitions", async () => {
    media.mockResolvedValue(stream());
    fetcher.mockResolvedValueOnce(ok())
      .mockResolvedValueOnce(ok(response({ face_found: false })))
      .mockResolvedValueOnce(ok(response({ abstained: true, abstain_threshold: .7 })))
      .mockResolvedValueOnce(ok(response({ insufficient_evidence: { insufficient: true, reasons: ["poor-detection-quality"] } })))
      .mockResolvedValue(ok());
    toggle(); await settle(); const first = chart().slice();
    expect(first).toHaveLength(1);
    for (let i = 0; i < 3; i++) { await next(); expect(chart()).toEqual(first); }
    await next(); expect(chart()).toHaveLength(2);
    select(1, "legacy", false); expect(chart()).toEqual([]); await settle();
    await settle(); expect(chart()).toHaveLength(1);
    select(0, "vit", false); expect(chart()).toEqual([]); await settle();
    toggle(false); expect(chart()).toEqual([]); await settle();
  });
});


describe("U3 selector, error and privacy clarity (mocked only)", () => {
  it("shows the narrow static notice before Start and requests no services or storage", () => {
    expect(text(tree)).toContain("configured local backend at http://127.0.0.1:8000");
    expect(text(tree)).toContain("Stop ends capture"); expect(text(tree)).toContain("does not undo work");
    expect(text(tree)).toContain("Server logging and retention are not verified");
    expect(nodes(tree).findIndex(n => n.type === "p" && text(n).startsWith("Live webcam:")))
      .toBeLessThan(nodes(tree).findIndex(n => n.type === "button"));
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); noStorageCalls();
  });
  it("retains calibration/profile selections for ViT/legacy without changing query semantics", async () => {
    const checkbox = () => nodes(tree).find(n => n.type === "input");
    checkbox().props.onChange({ target: { checked: true } }); render();
    select(1, "legacy"); expect(checkbox().props.disabled).toBe(true);
    select(0, "vit"); expect(text(tree)).toContain("n/a for ViT"); expect(text(tree)).toContain("Calibration unavailable for ViT");
    expect(nodes(tree).filter(n => n.type === "select")[1].props).toMatchObject({ value: "legacy", disabled: true, hidden: true });
    expect(checkbox().props.checked).toBe(true);
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok()); toggle(); await settle();
    expect(fetcher.mock.calls.at(-1)[0]).toBe("http://127.0.0.1:8000/api/infer?model=vit&profile=legacy");
    select(0, "mini-xception"); select(1, "corrected"); await settle();
    expect(checkbox().props.checked).toBe(true); expect(checkbox().props.disabled).toBe(false);
    expect(fetcher.mock.calls.at(-1)[0]).toContain("profile=corrected&calibrated=true");
  });
  it("adds only server-attested 0.95, retaining abstain-off default and query rule", async () => {
    const abstain = nodes(tree).filter(n => n.type === "select")[2];
    expect(abstain.props.value).toBe("off"); expect(nodes(abstain).filter(n => n.type === "option").map(n => n.props.value)).toEqual(["off", "0.5", "0.7", "0.9", "0.95"]);
    select(2, "0.95"); media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok()); toggle(); await settle();
    expect(fetcher.mock.calls.at(-1)[0]).toContain("&abstain=0.95");
  });
  it.each(["NotAllowedError", "NotFoundError"])("camera %s is readable with permission help and a sanitised diagnostic", async name => {
    const error = new Error("SYNTHETIC failure at C:\\private\\camera.py\n    at rawStack (/home/private/file.js:20)"); error.name = name;
    media.mockRejectedValue(error); toggle(); await settle();
    expect(text(tree)).toContain(name === "NotAllowedError" ? "Camera permission denied" : "Camera unavailable");
    expect(text(tree)).not.toContain("private"); expect(text(tree)).not.toContain("rawStack");
    const diag = nodes(tree).find(n => n.props?.['aria-label'] === "Sanitised error diagnostic");
    expect(diag.props.open).toBeUndefined(); expect(text(diag)).toContain("[path removed]");
    expect(fetcher).not.toHaveBeenCalled(); expect(timers.size).toBe(0);
  });
  it.each([
    [422, { detail: [{ loc: ["query", "abstain"], msg: "SYNTHETIC invalid threshold", input: "DO-NOT-EXPOSE", ctx: { path: "DO-NOT-EXPOSE" } }] }, "request rejected", "query > abstain: SYNTHETIC invalid threshold"],
    [422, { detail: { loc: ["body", "image"], message: "SYNTHETIC invalid frame", stack: "DO-NOT-EXPOSE" } }, "request rejected", "body > image: SYNTHETIC invalid frame"],
    [422, { detail: "SYNTHETIC rejected" }, "request rejected", "SYNTHETIC rejected"],
    [501, { detail: "SYNTHETIC integration unavailable" }, "model not integrated", "SYNTHETIC integration unavailable"],
    [503, { detail: "SYNTHETIC model unavailable /home/private/model.py" }, "requested model unavailable", "[path removed]"],
    [500, { detail: { stack: "DO-NOT-EXPOSE" } }, "HTTP 500", "No readable diagnostic reported"],
  ])("HTTP %s detail is bounded readable text, never object/stack coercion", async (status, payload, expected, detail) => {
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue({ ok: false, status, json: async () => payload });
    toggle(); await settle(); expect(text(nodes(tree).find(n => n.props?.role === "alert"))).toContain(expected);
    const diag = nodes(tree).find(n => n.props?.['aria-label'] === "Sanitised error diagnostic");
    expect(text(diag)).toContain(detail); expect(diag.props.open).toBeUndefined();
    expect(text(tree)).not.toContain("[object Object]"); expect(text(tree)).not.toContain("DO-NOT-EXPOSE");
    expect(text(tree)).not.toContain("/home/private"); expect(chart()).toEqual([]); noStorageCalls();
  });
  it.each([200, 500, 503])("non-JSON HTTP %s keeps the category and omits the raw body", async status => {
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue({ ok: status === 200, status, json: async () => { throw new SyntaxError("RAW-STACK C:/private/file.py"); } });
    toggle(); await settle(); expect(text(tree)).toContain(status === 200 ? "unparseable JSON" : `HTTP ${status}`);
    expect(text(tree)).toContain("raw body omitted"); expect(text(tree)).not.toContain("RAW-STACK"); expect(chart()).toEqual([]);
  });
  it("unknown transport errors are generic outside collapsed sanitised details and recover", async () => {
    media.mockResolvedValue(stream()); fetcher.mockRejectedValueOnce(new Error("SYNTHETIC transport C:/private/file.py\n at rawStack" )).mockResolvedValue(ok());
    toggle(); await settle(); const alert = text(nodes(tree).find(n => n.props?.role === "alert"));
    expect(alert).toBe("Inference unavailable. Check the local backend and try again.");
    expect(text(tree)).not.toContain("private"); expect(text(tree)).not.toContain("rawStack");
    await next(); expect(statusText()).toBe("Current apparent-expression response");
    expect(nodes(tree).find(n => n.props?.['aria-label'] === "Sanitised error diagnostic")).toBeUndefined();
  });
  it("draw failures are frame failures, with no upload and no raw stack path", async () => {
    media.mockResolvedValue(stream()); canvas.getContext.mockImplementation(() => { throw new Error("SYNTHETIC frame /private/source.py"); });
    toggle(); await settle(); expect(text(tree)).toContain("frame capture unavailable");
    expect(text(tree)).not.toContain("/private/source.py"); expect(fetcher).not.toHaveBeenCalled();
  });
});


const provenancePanel = () => nodes(tree).find(n => n.type === "details" && n.props?.['aria-label'] === "Response provenance and uncertainty");
const provenanceValues = () => {
  const entries = nodes(provenancePanel()).filter(n => n.type === "dt" || n.type === "dd");
  return Object.fromEntries(entries.filter((_, i) => i % 2 === 0).map((n, i) => [text(n), text(entries[i * 2 + 1])]));
};
describe("U4 self-reported provenance and uncertainty", () => {
  it("uses native collapsed details/summary and explicit not-reported defaults without requests", () => {
    const panel = provenancePanel(); expect(panel.props.open).toBeUndefined();
    expect(nodes(panel).find(n => n.type === "summary")).toBeDefined();
    expect(provenanceValues()).toMatchObject({ "Response model": "not reported", "Captured request profile": "not reported", Source: "not reported",
      "Raw/calibrated flag": "not reported", Temperature: "not reported", Revision: "not reported", Warning: "not reported",
      "Model score": "unavailable", "Source-reported uncertainty": "unavailable" });
    expect(text(panel)).toContain("not calibrated probabilities of feelings");
    expect(text(panel)).toContain("different models do not share comparable calibration");
    expect(text(panel)).toContain("runtime identity and asset provenance are not verified");
    expect(fetcher).not.toHaveBeenCalled(); expect(media).not.toHaveBeenCalled(); noStorageCalls();
  });
  it("renders the real calibrated shape with temperature/probs_raw without inventing a revision", async () => {
    const calibrated = response(); calibrated.selected_model = "mini-xception";
    Object.assign(calibrated.face.expression, { probs: { happiness: .8, sadness: 0, anger: 0, surprise: 0, fear: 0, disgust: 0, neutral: .1, unknown: .1 },
      confidence: .8, uncertainty: .2, source: "trained-minixception-fer2013", profile: "corrected", calibrated: true,
      probs_raw: { happiness: 1, sadness: 0, anger: 0, surprise: 0, fear: 0, disgust: 0, neutral: 0, unknown: 0 }, temperature: 1.045 });
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok(calibrated)); toggle(); await settle();
    expect(provenanceValues()).toMatchObject({ "Response model": "mini-xception", "Captured request profile": "corrected", "Response profile": "corrected",
      "Raw/calibrated flag": "calibrated (source-reported)", Temperature: "1.045", Revision: "not reported", "Model score": "0.80", "Source-reported uncertainty": "0.20" });
    expect(text(provenancePanel())).toContain("1 - max(class score)");
    expect(text(provenancePanel())).toContain("not independent uncertainty evidence");
  });
  it("renders the real ViT shape with n/a captured profile, null heuristics and an always-visible experimental warning", async () => {
    const vit = response(); vit.selected_model = "vit";
    Object.assign(vit.face.expression, { source: "hf-vit-dima806", calibrated: false, experimental: true,
      revision: "SYNTHETIC-revision", warning: "SYNTHETIC apparent-expression scores; possible training overlap",
      valence: null, arousal: null, expressiveness: null });
    expect(vit.face.expression).not.toHaveProperty("profile");
    select(0, "vit"); media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok(vit)); toggle(); await settle();
    expect(provenanceValues()).toMatchObject({ "Response model": "vit", "Captured request profile": "n/a", "Response profile": "not reported",
      "Raw/calibrated flag": "raw / uncalibrated (source-reported)", Revision: "SYNTHETIC-revision", Temperature: "not reported" });
    const warning = nodes(tree).find(n => n.props?.['aria-label'] === "Visible source warning");
    expect(text(warning)).toContain("Experimental response"); expect(text(warning)).toContain("SYNTHETIC apparent-expression");
    expect(nodes(tree).filter(n => n.type === "details").every(d => !nodes(d).includes(warning))).toBe(true);
    expect(text(provenancePanel())).toContain("1 - max(class score)");
  });
  it("keeps the experimental warning visible even if source metadata is not reported", async () => {
    const vit = response(); vit.selected_model = "vit"; select(0, "vit");
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok(vit)); toggle(); await settle();
    expect(text(nodes(tree).find(n => n.props?.['aria-label'] === "Visible source warning"))).toContain("Source warning: not reported");
    expect(text(provenancePanel())).toContain("Definition not reported for this response");
  });
  it("does not attest the complement for an unknown source or inconsistent reported numbers", async () => {
    media.mockResolvedValue(stream()); fetcher.mockResolvedValueOnce(ok()); toggle(); await settle();
    expect(text(provenancePanel())).toContain("Definition not reported");
    const mismatch = response(); mismatch.selected_model = "mini-xception";
    Object.assign(mismatch.face.expression, { source: "trained-minixception-fer2013", confidence: .8, uncertainty: .2 });
    fetcher.mockResolvedValue(ok(mismatch)); await next();
    expect(text(provenancePanel())).toContain("Definition not reported");
  });
  it("keeps captured provenance bound to the current run and clears it before accepting switched responses", async () => {
    const stale = deferred<any>(); media.mockResolvedValue(stream()); fetcher.mockResolvedValue({ ok: true, json: () => stale.promise });
    toggle(); await settle(); select(1, "legacy", false);
    expect(provenanceValues()["Captured request profile"]).toBe("not reported");
    const current = response(); current.selected_model = "SYNTHETIC-current";
    fetcher.mockResolvedValue(ok(current)); await settle();
    expect(provenanceValues()).toMatchObject({ "Captured request profile": "legacy", "Response model": "SYNTHETIC-current" });
    const old = response(); old.selected_model = "SYNTHETIC-stale"; stale.resolve(old); await settle();
    expect(text(provenancePanel())).not.toContain("SYNTHETIC-stale"); expect(provenanceValues()["Captured request profile"]).toBe("legacy");
    toggle(false); expect(provenanceValues()["Response model"]).toBe("not reported");
  });
  it("leaves no-face scores unavailable and sanitises metadata without mutating the response", async () => {
    const noface = response({ face_found: false, expression: { probs: { happiness: 0, sadness: 0, anger: 0, surprise: 0, fear: 0, disgust: 0, neutral: 0, unknown: 1 },
      confidence: .15, uncertainty: .85, source: "SYNTHETIC C:/private/model.py", warning: "SYNTHETIC /private/warning.py" } });
    const before = JSON.stringify(noface); media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok(noface)); toggle(); await settle();
    expect(provenanceValues()).toMatchObject({ "Model score": "unavailable", "Source-reported uncertainty": "unavailable", "Raw/calibrated flag": "not reported" });
    expect(text(tree)).not.toContain("/private/"); expect(text(provenancePanel())).toContain("[path removed]");
    expect(JSON.stringify(noface)).toBe(before);
  });
  it("labels Demo metadata synthetic and retains the persistent disclosure with zero IO", async () => {
    mode("demo"); toggle(); await settle();
    expect(provenanceValues()).toMatchObject({ "Response model": "SYNTHETIC-demo", "Captured request profile": "synthetic fixture", "Raw/calibrated flag": "not reported" });
    expect(text(tree)).toContain(DEMO_LABEL); expect(text(provenancePanel())).toContain("Definition not reported");
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); noStorageCalls();
  });
});


const historyTable = () => nodes(tree).find(n => n.type === "table" && n.props.className === "history-table");
const historyTableRows = () => nodes(nodes(historyTable()).find(n => n.type === "tbody"))
  .filter(n => n.type === "tr").map(row => nodes(row).filter(n => n.type === "th" || n.type === "td").map(text));
const expectedHistoryRows = () => chart().map((point: any, i: number) => [String(i + 1), point.conf.toFixed(2), point.top]);
describe("U5 text history alternative (hook evidence, not DOM/layout)", () => {
  it("keeps table points aligned with chart, omits non-usable frames, and clears both on Stop", async () => {
    expect(text(historyTable())).toContain("No usable score points");
    media.mockResolvedValue(stream()); const changed = response();
    changed.face.expression.confidence = .8; changed.face.expression.uncertainty = .2;
    fetcher.mockResolvedValueOnce(ok()).mockResolvedValueOnce(ok(changed)).mockResolvedValueOnce(ok(response({ face_found: false })));
    toggle(); await settle(); await next(); expect(chart()).toHaveLength(2);
    expect(historyTableRows()).toEqual(expectedHistoryRows());
    await next(); expect(statusText()).toBe("No face detected");
    expect(historyTableRows()).toEqual(expectedHistoryRows()); expect(chart()).toHaveLength(2);
    expect(nodes(historyTable()).filter(n => n.type === "th" && n.props.scope === "col").map(text))
      .toEqual(["Sample", "Model score", "Top class (score label)"]);
    toggle(false); expect(chart()).toEqual([]); expect(text(historyTable())).toContain("No usable score points");
  });
  it("uses the same latest 60 run-owned points without adding separate history state", async () => {
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok()); toggle(); await settle();
    for (let i = 0; i < 64; i++) await next();
    expect(chart()).toHaveLength(60); expect(historyTableRows()).toEqual(expectedHistoryRows());
    select(1, "legacy", false); expect(chart()).toEqual([]); expect(text(historyTable())).toContain("No usable score points");
  });
  it("keeps the Demo alternative fabricated, clears it on failure/mode switch, and performs zero IO", async () => {
    mode("demo"); toggle(); await settle();
    expect(text(nodes(historyTable()).find(n => n.type === "caption"))).toBe("SYNTHETIC DEMO - fabricated score points");
    expect(historyTableRows()).toEqual(expectedHistoryRows());
    for (let i = 0; i < 4; i++) await next();
    expect(statusText()).toBe("Inference unavailable"); expect(text(historyTable())).toContain("No usable score points");
    expect(text(tree)).toContain(DEMO_LABEL); mode("live", false);
    expect(text(historyTable())).toContain("No usable score points"); await settle();
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); noStorageCalls();
  });
});


describe("UI contrast source values and native diagnostic structure", () => {
  it("uses an error class whose actual opaque CSS colours meet 4.5:1 on their background", async () => {
    const error = new Error("SYNTHETIC camera diagnostic"); error.name = "NotAllowedError";
    media.mockRejectedValue(error); toggle(); await settle();
    const alert = nodes(tree).find(n => n.props?.role === "alert");
    expect(text(alert)).toContain("Camera permission denied"); expect(alert.props.style).toBeUndefined();
    const css: string = readFileSync(new URL("./App.css", import.meta.url), "utf8");
    const rule = css.match(new RegExp("\\." + alert.props.className + "\\s*\\{([^}]*)\\}"));
    expect(rule).not.toBeNull();
    const property = (name: string) => {
      const found = rule![1].match(new RegExp("(?:^|;)\\s*" + name + "\\s*:\\s*([^;]+)"));
      expect(found).not.toBeNull(); return found![1].trim();
    };
    const luminance = (hex: string) => {
      expect(hex).toMatch(/^#(?:[a-f\d]{3}|[a-f\d]{6})$/i);
      const expanded = hex.length === 4 ? hex.slice(1).split("").map(c => c + c).join("") : hex.slice(1);
      const linear = [0, 2, 4].map(i => parseInt(expanded.slice(i, i + 2), 16) / 255)
        .map(channel => channel <= .04045 ? channel / 12.92 : ((channel + .055) / 1.055) ** 2.4);
      return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
    };
    const foreground = luminance(property("color")), background = luminance(property("background-color"));
    const contrast = (Math.max(foreground, background) + .05) / (Math.min(foreground, background) + .05);
    expect(contrast).toBeGreaterThanOrEqual(4.5);
  });
  it("keeps the conditional native diagnostic summary first, unhidden and before inspection", async () => {
    media.mockResolvedValue(stream()); fetcher.mockRejectedValue(new Error("SYNTHETIC transport diagnostic"));
    toggle(); await settle();
    const diag = nodes(tree).find(n => n.props?.['aria-label'] === "Sanitised error diagnostic");
    const summary = nodes(diag).find(n => n.type === "summary");
    expect(diag.type).toBe("details"); expect(diag.props.children[0]).toBe(summary);
    for (const node of [diag, summary]) {
      expect(node.props.hidden).toBeUndefined(); expect(node.props.inert).toBeUndefined();
      expect(node.props.tabIndex).toBeUndefined();
    }
    expect(diag.props.open).toBeUndefined();
    const inspection = nodes(tree).find(n => n.type === "summary" && text(n).startsWith("Inspection details"));
    expect(nodes(tree).indexOf(summary)).toBeLessThan(nodes(tree).indexOf(inspection));
  });
});


describe("S39 dashboard movement session", () => {
  it("sends one cryptographic memory-only id on repeated requests and control restarts", async () => {
    const uuid = vi.fn().mockReturnValue("a9876543-1234-4567-89ab-123456789abc");
    vi.stubGlobal("crypto", { randomUUID: uuid });
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok());
    toggle(); await settle(); await next(); select(1, "legacy"); await settle();
    const posts = fetcher.mock.calls.filter((call: any[]) => call[1].method === "POST");
    expect(posts.length).toBeGreaterThanOrEqual(2);
    const ids = posts.map((call: any[]) => new Headers(call[1].headers).get("X-Socialcue-Session"));
    expect(ids.every((id: string | null) => id === "a9876543-1234-4567-89ab-123456789abc")).toBe(true);
    expect(uuid).toHaveBeenCalledTimes(1);
    expect(ids[0]!.length).toBeGreaterThanOrEqual(22);
    expect(storage.setItem).not.toHaveBeenCalled(); expect(idb.open).not.toHaveBeenCalled(); expect(cacheStore.open).not.toHaveBeenCalled();
  });
  it("uses a fresh id for a new page instance and explains bounded numeric retention", async () => {
    const uuid = vi.fn().mockReturnValueOnce("a9876543-1234-4567-89ab-123456789abc").mockReturnValueOnce("b9876543-1234-4567-89ab-123456789abc");
    vi.stubGlobal("crypto", { randomUUID: uuid });
    media.mockResolvedValue(stream()); fetcher.mockResolvedValue(ok());
    toggle(); await settle();
    const first = new Headers(fetcher.mock.calls[0][1].headers).get("X-Socialcue-Session");
    hooks.effects.forEach(effect => effect.cleanup?.());
    hooks.states = []; hooks.refs = []; hooks.effects = []; hooks.pending = []; timers.clear(); render();
    expect(text(tree)).toContain("numeric body movement");
    toggle(); await settle();
    const posts = fetcher.mock.calls.filter((call: any[]) => call[1].method === "POST");
    const second = new Headers(posts[posts.length - 1][1].headers).get("X-Socialcue-Session");
    expect(first).not.toBe(second); expect(uuid).toHaveBeenCalledTimes(2);
    expect(storage.getItem).not.toHaveBeenCalled(); expect(storage.setItem).not.toHaveBeenCalled();
  });
});


it("U6 G5 explains dashboard settings in plain language", () => {
  for (const label of ["Analysis model", "Input processing profile", "Use calibrated model scores (T=1.045; corrected profile only)", "Minimum top model score (abstain below threshold)"]) expect(text(tree)).toContain(label);
});
