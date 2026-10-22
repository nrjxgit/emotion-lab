// Manufactured fakes only: no DOM, camera, network, storage or real browser.
import { afterEach, describe, expect, it, vi } from "vitest";
// @ts-expect-error Node fs exists in the test runtime; this browser project omits Node declarations.
import { readFileSync, mkdtempSync, writeFileSync } from "node:fs";
// @ts-expect-error Node built-ins are provided by the test runtime.
import { tmpdir } from "node:os";
// @ts-expect-error Node built-ins are provided by the test runtime.
import { join } from "node:path";
import {
  ANNOUNCE_MIN_GAP_MS, CONFIDENCE_CAP, CONSENT_LABEL, CUE_PERIOD_MS, CueController, DEFAULT_BANNER, SESSION_HEADER,
  STATE_NAMES, bodyEvidenceText, cuesEnabled, newSessionId, parseCueBlock, presentCueBlock, reasonText, shouldAnnounce,
  type CueBlock, type CueDeps,
} from "./socialCueLogic";
import { CUE_FIXTURES, fixtureBlock } from "./socialCueFixtures";

const BLOCK = { states: [{ name: "nervous-like", confidence: 0.6, evidence: ["p_neg_arousal=0.700"] }],
  insufficient: { flag: false, reasons: [] }, banner: "server banner", heuristic: "HB-1 hypothesis, unvalidated" };
const deferred = <T,>() => { let resolve!: (x: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return { promise, resolve }; };
const last = <T,>(a: T[]): T | undefined => a[a.length - 1];
const pump = async () => { for (let i = 0; i < 25; i++) await Promise.resolve(); };

function harness(over: Partial<CueDeps> = {}, response: any = { socialcue: BLOCK }) {
  const timers = new Map<number, { fn: () => void; ms: number }>();
  let seq = 0, ids = 0, clock = 1000, active = 0, maxActive = 0;
  const aborts: any[] = [];
  const blocks: (CueBlock | null)[] = [], statuses: [string, string][] = [];
  const camera = { open: vi.fn(async () => {}), frame: vi.fn(async () => new Blob(["manufactured"])), close: vi.fn() };
  const fetchFn = vi.fn(async (_url: string, init: any) => {
    if (init.method === "DELETE") return { ok: true, json: async () => ({}) };
    active++; maxActive = Math.max(maxActive, active); // in flight until the body is consumed
    return { ok: true, json: async () => { active--; return response; } };
  });
  const deps: CueDeps = {
    api: "http://api.test", fetchFn, camera, newId: () => `session-${++ids}-abcdefghijklmnopqrstuv`,
    setTimer: (fn, ms) => { timers.set(++seq, { fn, ms }); return seq; }, clearTimer: id => { timers.delete(id); },
    now: () => clock, makeAbort: () => { const a = { signal: { aborted: false }, abort: vi.fn() }; aborts.push(a); return a; },
    makeForm: blob => ({ image: blob }), onBlock: b => blocks.push(b), onStatus: (s, m) => statuses.push([s, m]), ...over,
  };
  const c = new CueController(deps);
  const fire = async () => { const [[id, t]] = [...timers.entries()]; timers.delete(id); t.fn(); await pump(); };
  return { c, deps, timers, fetchFn, camera, blocks, statuses, aborts, fire, advance: (ms: number) => { clock += ms; }, maxActive: () => maxActive };
}
const posts = (h: ReturnType<typeof harness>) => h.fetchFn.mock.calls.filter(([, i]) => i.method === "POST");
const deletes = (h: ReturnType<typeof harness>) => h.fetchFn.mock.calls.filter(([, i]) => i.method === "DELETE");

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("consent gate", () => {
  it("sends nothing and opens no camera before consent", async () => {
    const h = harness();
    expect(await h.c.start()).toBe(false);
    expect(h.camera.open).not.toHaveBeenCalled(); expect(h.fetchFn).not.toHaveBeenCalled(); expect(h.timers.size).toBe(0);
    expect(last(h.statuses)![1]).toContain("Consent is required");
    expect(CONSENT_LABEL).toContain("I am the only person in view and I agree to send camera frames to the configured local backend");
    expect(CONSENT_LABEL).toContain("Stop or unchecking this box ends capture"); expect(CONSENT_LABEL).toContain("retention limits");
  });
  it("starts after consent and the first frame is the first request", async () => {
    const h = harness(); h.c.setConsent(true);
    expect(await h.c.start()).toBe(true); expect(h.fetchFn).not.toHaveBeenCalled();
    await h.fire(); expect(posts(h)).toHaveLength(1); expect(last(h.blocks)?.states[0].name).toBe("nervous-like");
  });
  it("revoking consent stops the loop, closes the camera, clears the display and calls DELETE", async () => {
    const h = harness(); h.c.setConsent(true); await h.c.start(); await h.fire();
    const id = h.c.sessionId!;
    h.c.revoke();
    expect(h.c.consent).toBe(false); expect(h.c.running).toBe(false); expect(h.timers.size).toBe(0);
    expect(h.camera.close).toHaveBeenCalled(); expect(last(h.blocks)).toBeNull();
    expect(deletes(h)).toHaveLength(1);
    const [url, init] = deletes(h)[0];
    expect(url).toBe("http://api.test/api/socialcue/session"); expect(init.headers[SESSION_HEADER]).toBe(id);
    expect(await h.c.start()).toBe(false); // consent must be given again
  });
  it("does not call DELETE when nothing was ever sent", async () => {
    const h = harness(); h.c.setConsent(true); await h.c.start(); h.c.stop();
    expect(deletes(h)).toHaveLength(0); expect(h.fetchFn).not.toHaveBeenCalled();
  });
  it("Stop ends the consent: the next run needs a fresh agreement", async () => {
    const seen: boolean[] = []; const h = harness({ onConsent: v => seen.push(v) });
    h.c.setConsent(true); await h.c.start(); await h.fire(); h.c.stop();
    expect(h.c.consent).toBe(false); expect(seen[seen.length - 1]).toBe(false);
    const before = h.fetchFn.mock.calls.length;
    expect(await h.c.start()).toBe(false); expect(h.camera.open).toHaveBeenCalledTimes(1); expect(h.fetchFn.mock.calls.length).toBe(before);
    h.c.setConsent(true); expect(await h.c.start()).toBe(true);
  });
  it("a failed start also ends the consent", async () => {
    const seen: boolean[] = [];
    const cam = harness({ camera: { open: async () => { throw new Error("denied"); }, frame: async () => null, close: vi.fn() }, onConsent: v => seen.push(v) });
    cam.c.setConsent(true); expect(await cam.c.start()).toBe(false);
    expect(cam.c.consent).toBe(false); expect(seen).toEqual([true, false]);
    const rnd = harness({ newId: () => newSessionId({} as any), onConsent: v => seen.push(v) });
    rnd.c.setConsent(true); expect(await rnd.c.start()).toBe(false); expect(rnd.c.consent).toBe(false);
  });
  it("a late camera grant after revoke is closed and never sends", async () => {
    const open = deferred<void>(); const h = harness({ camera: { open: () => open.promise, frame: async () => new Blob(["x"]), close: vi.fn() } });
    h.c.setConsent(true); const started = h.c.start(); h.c.revoke(); open.resolve(); expect(await started).toBe(false); await pump();
    expect(h.deps.camera.close).toHaveBeenCalled(); expect(h.fetchFn).not.toHaveBeenCalled(); expect(h.timers.size).toBe(0);
  });
});

describe("request shape and session ids", () => {
  it("posts one JPEG blob to ?socialcue=1 with the session header", async () => {
    const h = harness(); h.c.setConsent(true); await h.c.start(); await h.fire();
    const [url, init] = posts(h)[0];
    expect(url).toBe("http://api.test/api/infer?socialcue=1"); expect(init.method).toBe("POST");
    expect(init.headers[SESSION_HEADER]).toBe(h.c.sessionId); expect(h.c.sessionId!.length).toBeGreaterThanOrEqual(22);
    expect(init.body.image).toBeInstanceOf(Blob);
  });
  it("rotates the session id on every start", async () => {
    const h = harness(); h.c.setConsent(true);
    await h.c.start(); await h.fire(); const a = h.c.sessionId; h.c.stop();
    h.c.setConsent(true); await h.c.start(); await h.fire(); const b = h.c.sessionId;
    expect(a).not.toBe(b); expect(posts(h).map(([, i]) => i.headers[SESSION_HEADER])).toEqual([a, b]);
    expect(deletes(h).map(([, i]) => i.headers[SESSION_HEADER])).toEqual([a]);
  });
  it("newSessionId prefers randomUUID, falls back to getRandomValues and never uses Math.random", () => {
    vi.spyOn(Math, "random").mockImplementation(() => { throw new Error("Math.random must not be used"); });
    expect(newSessionId({ randomUUID: () => "123e4567-e89b-42d3-a456-426614174000" })).toBe("123e4567-e89b-42d3-a456-426614174000");
    const id = newSessionId({ getRandomValues: a => { a.forEach((_, i) => { a[i] = i * 7; }); return a; } });
    expect(id).toMatch(/^[0-9a-f]{32}$/);
    expect(() => newSessionId(undefined as any)).not.toThrow(); // real globalThis.crypto exists in the test runtime
    expect(() => newSessionId({} as any)).toThrow("secure random");
    expect(newSessionId()).toMatch(/^[A-Za-z0-9_-]{22,64}$/);
  });
  it("real ids are unique and match the server's id rule", () => {
    const ids = new Set(Array.from({ length: 50 }, () => newSessionId()));
    expect(ids.size).toBe(50); ids.forEach(id => expect(id).toMatch(/^[A-Za-z0-9_-]{22,64}$/));
  });
  it("start fails closed when no secure random source exists", async () => {
    const h = harness({ newId: () => newSessionId({} as any) }); h.c.setConsent(true);
    expect(await h.c.start()).toBe(false); expect(h.camera.open).not.toHaveBeenCalled(); expect(h.fetchFn).not.toHaveBeenCalled();
    expect(last(h.statuses)![0]).toBe("error");
  });
  it("a camera failure is reported and nothing is sent", async () => {
    const h = harness({ camera: { open: async () => { throw new Error("denied"); }, frame: async () => null, close: vi.fn() } });
    h.c.setConsent(true); expect(await h.c.start()).toBe(false);
    expect(last(h.statuses)![1]).toContain("Camera unavailable"); expect(h.fetchFn).not.toHaveBeenCalled(); expect(h.c.sessionId).toBeNull();
  });
});

describe("paced loop", () => {
  it("targets 8-10 fps", () => { expect(1000 / CUE_PERIOD_MS).toBeGreaterThanOrEqual(8); expect(1000 / CUE_PERIOD_MS).toBeLessThanOrEqual(10); });
  it("has exactly one request in flight and a busy tick is skipped", async () => {
    const gate = deferred<any>(); const h = harness();
    h.fetchFn.mockImplementation(async (_u: string, init: any) => init.method === "DELETE" ? { ok: true } : gate.promise);
    h.c.setConsent(true); await h.c.start(); await h.fire(); expect(posts(h)).toHaveLength(1);
    expect(h.c.inFlight).toBe(true);
    await h.c.tick(); await h.c.tick(); expect(posts(h)).toHaveLength(1); // direct ticks while busy do nothing
    expect(h.timers.size).toBe(1);
    gate.resolve({ ok: true, json: async () => ({ socialcue: BLOCK }) }); await pump();
    expect(h.c.inFlight).toBe(false); await h.fire(); expect(posts(h)).toHaveLength(2);
  });
  it("never overlaps requests across a long run", async () => {
    const h = harness(); h.c.setConsent(true); await h.c.start();
    for (let i = 0; i < 20; i++) await h.fire();
    expect(posts(h)).toHaveLength(20); expect(h.maxActive()).toBe(1);
  });
  it("schedules the next tick so the period is kept", async () => {
    const h = harness(); h.c.setConsent(true); await h.c.start();
    h.fetchFn.mockImplementation(async (_u: string, init: any) => { h.advance(50); return { ok: true, json: async () => ({ socialcue: BLOCK }) }; });
    await h.fire(); expect([...h.timers.values()][0].ms).toBe(CUE_PERIOD_MS - 50);
    h.fetchFn.mockImplementation(async () => { h.advance(500); return { ok: true, json: async () => ({ socialcue: BLOCK }) }; });
    await h.fire(); expect([...h.timers.values()][0].ms).toBe(0);
  });
  it("pauses on a hidden tab (no timer, in-flight aborted) and resumes when visible", async () => {
    const gate = deferred<any>(); const h = harness();
    h.fetchFn.mockImplementation(async (_u: string, init: any) => init.method === "DELETE" ? { ok: true } : gate.promise);
    h.c.setConsent(true); await h.c.start(); await h.fire();
    h.c.setHidden(true);
    expect(h.timers.size).toBe(0); expect(h.aborts[0].abort).toHaveBeenCalled(); expect(last(h.statuses)![0]).toBe("paused");
    gate.resolve({ ok: true, json: async () => ({ socialcue: BLOCK }) }); await pump();
    expect(h.timers.size).toBe(0); expect(posts(h)).toHaveLength(1); // nothing scheduled or sent while hidden
    h.fetchFn.mockImplementation(async () => ({ ok: true, json: async () => ({ socialcue: BLOCK }) }));
    h.c.setHidden(false); expect(last(h.statuses)![0]).toBe("running"); await h.fire(); expect(posts(h)).toHaveLength(2);
  });
  it("stop aborts the pending request and ignores its late result", async () => {
    const gate = deferred<any>(); const h = harness();
    h.fetchFn.mockImplementation(async (_u: string, init: any) => init.method === "DELETE" ? { ok: true } : gate.promise);
    h.c.setConsent(true); await h.c.start(); await h.fire(); const before = h.blocks.length;
    h.c.stop(); expect(h.aborts[0].abort).toHaveBeenCalled();
    gate.resolve({ ok: true, json: async () => ({ socialcue: BLOCK }) }); await pump();
    expect(h.blocks.slice(before)).toEqual([null]); expect(h.timers.size).toBe(0); expect(deletes(h)).toHaveLength(1);
  });
  it("keeps trying after an HTTP error or a malformed response, without showing server text", async () => {
    const h = harness(); h.c.setConsent(true); await h.c.start();
    h.fetchFn.mockImplementation(async () => ({ ok: false, status: 500, json: async () => ({ detail: "<img src=x onerror=alert(1)>" }) }));
    await h.fire(); expect(last(h.statuses)![0]).toBe("error"); expect(last(h.statuses)![1]).not.toContain("<img"); expect(h.timers.size).toBe(1);
    h.fetchFn.mockImplementation(async () => ({ ok: true, json: async () => ({ socialcue: { states: "no" } }) }));
    await h.fire(); expect(last(h.statuses)![0]).toBe("error"); expect(h.blocks.filter(Boolean)).toHaveLength(0);
    h.fetchFn.mockImplementation(async () => ({ ok: true, json: async () => ({ socialcue: BLOCK }) }));
    await h.fire(); expect(last(h.blocks)).not.toBeNull();
  });
  it("a frame-less tick (video not ready) sends nothing and keeps pacing", async () => {
    const h = harness({ camera: { open: async () => {}, frame: async () => null, close: vi.fn() } });
    h.c.setConsent(true); await h.c.start(); await h.fire(); await h.fire();
    expect(posts(h)).toHaveLength(0); expect(h.timers.size).toBe(1);
  });
});

describe("presentation", () => {
  const view = (b: any) => presentCueBlock(parseCueBlock(b));
  it("shows the multiple-people message and never 'no face' for that case", () => {
    const v = view({ ...BLOCK, states: [], insufficient: { flag: true, reasons: ["multiple_faces", "no_face"] } });
    expect(v.reasonText).toBe("More than one person in view: not analysed");
    for (const row of v.rows) { expect(row.text).toContain("More than one person in view: not analysed"); expect(row.text.toLowerCase()).not.toContain("no face"); }
    expect(v.summary.toLowerCase()).not.toContain("no face"); expect(reasonText(["no_face"])).toBe("No face in view");
  });
  it("defaults every row to 'No cue / insufficient evidence' with the reason", () => {
    const v = presentCueBlock(null);
    expect(v.rows.map(r => r.name)).toEqual([...STATE_NAMES]);
    v.rows.forEach(r => { expect(r.text).toContain("No cue / insufficient evidence"); expect(r.active).toBe(false); });
    expect(view({ ...BLOCK, states: [], insufficient: { flag: true, reasons: ["low_quality"] } }).rows[0].text).toContain("Detection quality too low");
    expect(view({ ...BLOCK, states: [], insufficient: { flag: true, reasons: ["window_filling"] } }).rows[1].text).toContain("Collecting frames");
    expect(view({ ...BLOCK, states: [], insufficient: { flag: true, reasons: ["something_new"] } }).reasonText).toBe("Insufficient evidence");
    expect(view({ ...BLOCK, states: [], insufficient: { flag: false, reasons: [] } }).rows.every(r => r.text === "No cue")).toBe(true);
  });
  it("explains a session_invalidated block instead of the generic fallback (S60, review S59 finding 9)", () => {
    const text = "The session was reset or expired; cues restart with a new window";
    expect(reasonText(["session_invalidated"])).toBe(text);
    const v = view({ ...BLOCK, states: [], insufficient: { flag: true, reasons: ["session_invalidated"] } });
    expect(v.reasonText).toBe(text); expect(v.reasonText).not.toBe("Insufficient evidence");
    v.rows.forEach(r => { expect(r.active).toBe(false); expect(r.text).toContain(text); });
  });
  it("uses the fixed hedged wording, caps confidence at 0.8 and never lets insufficient evidence show a state", () => {
    const v = view({ ...BLOCK, states: [{ name: "needs-help-like", confidence: 0.95, evidence: ["dwell_s=9.000"] }] });
    const row = v.rows[2];
    expect(row.active).toBe(true); expect(row.confidence).toBe(CONFIDENCE_CAP); expect(row.band).toBe("high");
    expect(row.text).toMatch(/^Signs consistent with .*, not a reading of feelings$/); expect(row.evidence).toEqual(["dwell_s=9.000"]);
    const hidden = view({ ...BLOCK, insufficient: { flag: true, reasons: ["low_quality"] } });
    expect(hidden.rows.every(r => !r.active)).toBe(true);
    expect(view({ ...BLOCK, states: [{ name: "unsure-like", confidence: 0.1, evidence: [] }] }).rows[1].band).toBe("low");
    expect(view({ ...BLOCK, states: [{ name: "unsure-like", confidence: 0.4, evidence: [] }] }).rows[1].band).toBe("medium");
  });
  it("rejects blocks it does not understand and bounds untrusted text", () => {
    for (const bad of [null, 5, "x", {}, { states: [], insufficient: {} }, { states: [{ name: "angry", confidence: 1 }], insufficient: { flag: false, reasons: [] } },
      { states: [{ name: "nervous-like", confidence: NaN }], insufficient: { flag: false, reasons: [] } },
      { states: [{ name: "nervous-like", confidence: "0.5" }], insufficient: { flag: false, reasons: [] } }]) expect(parseCueBlock(bad)).toBeNull();
    const long = parseCueBlock({ ...BLOCK, banner: "b".repeat(5000), states: [{ name: "nervous-like", confidence: 0.5, evidence: Array(50).fill("e".repeat(500)) }] })!;
    expect(long.banner.length).toBe(400); expect(long.states[0].evidence).toHaveLength(12); expect(long.states[0].evidence[0].length).toBe(120);
    expect(parseCueBlock({ ...BLOCK, banner: "" })!.banner).toBe(DEFAULT_BANNER);
  });
  it("throttles polite announcements", () => {
    expect(shouldAnnounce("a", "a", 0, 99999)).toBe(false);
    expect(shouldAnnounce("a", "b", 1000, 1000 + ANNOUNCE_MIN_GAP_MS - 1)).toBe(false);
    expect(shouldAnnounce("a", "b", 1000, 1000 + ANNOUNCE_MIN_GAP_MS)).toBe(true);
    expect(shouldAnnounce("", "No cue", -Infinity, 5)).toBe(true);
  });
  it("is hidden unless explicitly enabled by ?socialcues=1", () => {
    expect(cuesEnabled("?socialcues=1")).toBe(true); expect(cuesEnabled("?x=1&socialcues=1")).toBe(true);
    for (const q of ["", "?socialcues=0", "?socialcues=true", "?socialcues", "?socialcue=1", "?SOCIALCUES=1"]) expect(cuesEnabled(q)).toBe(false);
    vi.stubGlobal("window", {}); expect(cuesEnabled()).toBe(false);
    vi.stubGlobal("window", { location: { search: "?socialcues=1" } }); expect(cuesEnabled()).toBe(true);
    vi.stubGlobal("window", undefined); expect(cuesEnabled()).toBe(false);
  });
});

describe("synthetic demo fixtures", () => {
  it("has exactly the seven scenarios (incl. dropout), frozen, each passing the live validator", () => {
    expect(CUE_FIXTURES.map(f => f.id)).toEqual(["calm", "nervous-like", "unsure-like", "needs-help-like", "insufficient", "multi-face", "dropout"]);
    expect(Object.isFrozen(CUE_FIXTURES)).toBe(true);
    for (const f of CUE_FIXTURES) { expect(Object.isFrozen(f.block)).toBe(true); expect(fixtureBlock(f.id)).not.toBeNull(); }
    expect(fixtureBlock("nope")).toBeNull();
  });
  it("renders each scenario as intended", () => {
    const names = (id: string) => presentCueBlock(fixtureBlock(id)).rows.filter(r => r.active).map(r => r.name);
    expect(names("calm")).toEqual([]); expect(names("nervous-like")).toEqual(["nervous-like"]);
    expect(names("unsure-like")).toEqual(["unsure-like"]); expect(names("needs-help-like")).toEqual(["unsure-like", "needs-help-like"]);
    expect(presentCueBlock(fixtureBlock("insufficient")).reasonText).toBe("Detection quality too low");
    expect(presentCueBlock(fixtureBlock("multi-face")).reasonText).toBe("More than one person in view: not analysed");
    expect(presentCueBlock(fixtureBlock("dropout")).reasonText).toBe("Too few usable frames in the window");
    expect(names("dropout")).toEqual([]);
    CUE_FIXTURES.forEach(f => f.block.states.forEach(s => expect(s.confidence).toBeLessThanOrEqual(CONFIDENCE_CAP)));
  });
});

describe("static source checks", () => {
  const src = (name: string) => readFileSync(new URL(`./${name}`, import.meta.url), "utf8") as string;
  const files = ["socialCueLogic.ts", "socialCueFixtures.ts", "SocialCuePanel.tsx"];
  it("never injects HTML, evaluates code, uses Math.random or touches storage", () => {
    for (const f of files) {
      const text = src(f);
      for (const needle of ["innerHTML", "dangerouslySetInnerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
        "Math.random", "localStorage", "sessionStorage", "indexedDB", "navigator.sendBeacon", "XMLHttpRequest", "WebSocket"]) expect(text, `${f}: ${needle}`).not.toContain(needle);
    }
  });
  it("only calls the two documented endpoints", () => {
    const text = src("socialCueLogic.ts");
    expect([...text.matchAll(/\/api\/[a-z/?=0-9]+/g)].map(m => m[0]).sort()).toEqual(["/api/infer?socialcue=1", "/api/socialcue/session"]);
  });
  it("respects reduced motion and keeps the panel opt-in in App", () => {
    const css = src("App.css");
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\) \{ \.cue-bar \{ transition: none; \} \}/);
    const app = src("App.tsx");
    expect(app).toContain("{cuesEnabled() && <SocialCuePanel api={API} enabled />}");
    expect(app).toContain('import SocialCuePanel from "./SocialCuePanel";');
  });
});


describe("body evidence status (S69)", () => {
  it.each([
    ["no_pose_model", "no pose model installed - expected"], ["pose_not_visible", "pose not visible"],
    ["invalid_pose_payload", "invalid pose payload"], ["consent_off", "consent is off"],
    ["no_subject", "no subject"], ["multiple_people", "multiple people"],
    ["config_not_pinned", "configuration not pinned"], ["internal_error", "internal error"],
  ])("explains %s", (reason, wording) => {
    expect(parseCueBlock({ ...BLOCK, body: { status: "unavailable", reason } })?.bodyText)
      .toBe(`Body evidence: unavailable (${wording})`);
    expect(bodyEvidenceText({ status: "none", reason }, [])).toContain(wording);
  });
  it("handles absent, unknown, non-string and oversized fields defensively", () => {
    expect(parseCueBlock(BLOCK)?.bodyText).toBe("Body evidence: unavailable (reason not supplied)");
    for (const body of [null, [], 5, { status: {}, reason: {} }, { status: "__proto__", reason: "__proto__" }]) {
      expect(bodyEvidenceText(body, [])).toMatch(/^Body evidence: unavailable/);
    }
    expect(bodyEvidenceText({ reason: "x".repeat(10000) }, [])).toBe("Body evidence: unavailable (unknown reason: " + "x".repeat(120) + ")");
    expect(bodyEvidenceText({ status: "ok", cues: Array(65).fill({ cue: "x", present: true, status: "ok" }) }, []))
      .toBe("Body evidence (hypothesis, unvalidated): 0 cue(s) supporting none");
  });
  it("counts present cues and reads support only from actual state support items", () => {
    const body = { status: "ok", cues: [{ cue: "fidget", status: "ok", present: true },
      { cue: "stillness", status: "ok", present: false }, { cue: "bad", status: "none", present: true }] };
    const states = [{ ...BLOCK.states[0], body_support: { delta: 0.01, items: ["fidget"] } }];
    expect(parseCueBlock({ ...BLOCK, body, states })?.bodyText)
      .toBe("Body evidence (hypothesis, unvalidated): 1 cue(s) supporting nervous-like");
    expect(bodyEvidenceText(body, [{ name: "invented-state", body_support: { items: ["fidget"] } }]))
      .toBe("Body evidence (hypothesis, unvalidated): 1 cue(s) supporting none");
    expect(bodyEvidenceText({ status: "ok", cues: [] }, states))
      .toBe("Body evidence (hypothesis, unvalidated): 0 cue(s) supporting none");
    expect(parseCueBlock({ ...BLOCK, body, states, insufficient: { flag: true, reasons: [] } })?.bodyText)
      .toBe("Body evidence (hypothesis, unvalidated): 1 cue(s) supporting none");
    expect(bodyEvidenceText({ status: "ok", cues: [{ cue: {}, status: "ok", present: true },
      { cue: "x".repeat(121), status: "ok", present: true }] }, states)).toContain("0 cue(s)");
  });
});


describe("S71 backend body reasons", () => {
  it.each([
    ["window_filling", "collecting - needs about 2-3 s of frames"], ["invalid_input", "invalid input"],
    ["no_pose", "no pose available"], ["fps<5", "frame rate too low for body motion"],
    ["fps<8", "frame rate too low for body motion"], ["no_shoulders", "shoulders not visible"],
    ["identity_discontinuity", "person changed or jumped - reset"],
    ["body_face_mismatch", "body does not match the face in view"],
    ["occluded", "body partly hidden"], ["hands_not_visible", "hands not visible"], ["too_few_pairs", "too few frame pairs"],
  ])("explains %s", (reason, wording) => {
    expect(bodyEvidenceText({ status: "none", reason }, [])).toBe(`Body evidence: unavailable (${wording})`);
  });
  it("maps parameterized reasons with a clipped suffix and keeps unknown fallback", () => {
    expect(bodyEvidenceText({ reason: "occluded: coverage 0.25" }, [])).toContain("body partly hidden: coverage 0.25");
    expect(bodyEvidenceText({ reason: "fps<8: measured 6" }, [])).toContain("frame rate too low for body motion: measured 6");
    expect(bodyEvidenceText({ reason: "occluded: " + "x".repeat(500) }, []).length).toBeLessThan(200);
    expect(bodyEvidenceText({ reason: "fps<invented" }, [])).toContain("unknown reason:");
  });
  it("keeps the exact backend body-reason inventory mapped, including reformatted source", () => {
    const sources = readBodyReasonSources();
    assertBodyReasonInventory(sources);
    assertBodyReasonInventory(sources.map(s => s.replace(/"/g, "'").replace(/\r?\n/g, "\r\n")
      .replace(/_fail\(/g, "_fail( \r\n ").replace(/_unavailable\(/g, "_unavailable( \r\n ")));
  });
  it("rejects new reasons in temporary source copies without editing backend files", () => {
    const sources = readBodyReasonSources();
    const scratch = mkdtempSync(join(tmpdir(), "el006-s73-reason-"));
    const mutations: [number, string][] = [
      [0, '\n_fail(f"s73_new_reason {value}")\n'],
      [0, "\n_fail( F's73_new_reason {value}' )\n"],
      [0, '\n_unavailable(["fidget"], {"fidget": "speed"}, "s73_new_reason", 0, 0, cfg)\n'],
      [1, '\nPosePayload( reason = "s73_new_reason" )\n'],
      [2, '\nbody = {"status": "none", "reason": "s73_new_reason"}\n'],
    ];
    mutations.forEach(([index, addition], n) => {
      const path = join(scratch, `mutation-${n}.py`);
      writeFileSync(path, sources[index] + addition, "utf8");
      const mutated = [...sources]; mutated[index] = readFileSync(path, "utf8");
      expect(scanBodyReasons(mutated)).toContain("s73_new_reason");
      expect(() => assertBodyReasonInventory(mutated)).toThrow();
      expect(bodyEvidenceText({ reason: "s73_new_reason" }, [])).toContain("unknown reason:");
    });
    console.log("S73_MUTATION_COPIES=" + scratch);
  });
  it("reports unavailable cue statuses only when no present cues exist", () => {
    const cues = [{ status: "none", present: false, cue: "fidget" }, { status: "unavailable", cue: "self_touch" },
      { status: "ok", present: false, cue: "stillness" }, { status: 3 }, null];
    expect(bodyEvidenceText({ status: "ok", cues }, [])).toBe("Body evidence (hypothesis, unvalidated): 0 cue(s) supporting none; 2 cue(s) unavailable");
    expect(bodyEvidenceText({ status: "ok", cues: [...cues, { status: "ok", present: true, cue: "fidget" }] }, []))
      .toBe("Body evidence (hypothesis, unvalidated): 1 cue(s) supporting none");
  });
});


// Source-derived inventory: body.py _fail/cue reasons plus PosePayload and body
// attachment reasons. Updating a backend literal requires reviewing this list AND UI text.
const EXPECTED_BODY_REASONS = [
  "body_face_mismatch", "config_not_pinned", "consent_off", "fps<5", "fps<8",
  "hands_not_visible", "identity_discontinuity", "internal_error", "invalid_input",
  "invalid_pose_payload", "multiple_people", "no_pose", "no_pose_model", "no_shoulders",
  "no_subject", "occluded", "pose_not_visible", "too_few_pairs", "window_filling",
].sort();
function readBodyReasonSources(): string[] {
  return ["../../socialcue/body.py", "../../backend/app/body/pose_payload.py", "../../backend/app/socialcue_session.py"]
    .map(path => readFileSync(new URL(path, import.meta.url), "utf8"));
}

// Split positional arguments while respecting nested lists/dicts/calls and quoted strings.
// This is a literal inventory scanner, not a Python evaluator; variable-computed reasons
// still require review. F-string constant prefixes are deliberately retained up to {.
function positionalCallArgs(source: string, name: string): string[][] {
  const calls: string[][] = [];
  for (const match of source.matchAll(new RegExp("\\b" + name + "\\s*\\(", "g"))) {
    let depth = 1, quote = "", escaped = false, arg = ""; const args: string[] = [];
    for (let i = match.index! + match[0].length; i < source.length; i++) {
      const c = source[i];
      if (quote) {
        arg += c;
        if (escaped) escaped = false;
        else if (c === "\\") escaped = true;
        else if (c === quote) quote = "";
      } else if (c === '"' || c === "'") { quote = c; arg += c; }
      else if ("([{".includes(c)) { depth++; arg += c; }
      else if (")]}".includes(c)) {
        if (--depth === 0) { args.push(arg); calls.push(args); break; }
        arg += c;
      } else if (c === "," && depth === 1) { args.push(arg); arg = ""; }
      else arg += c;
    }
  }
  return calls;
}
function scanBodyReasons(sources: string[]): string[] {
  const [body, pose, session] = sources, reasons = new Set<string>();
  const add = (s: string) => reasons.add(s.split("{")[0].replace(/: coverage.*$/, "").trimEnd());
  const literals = (s: string) => { for (const m of s.matchAll(/[fF]?["']([^"']+)["']/g)) add(m[1]); };
  for (const m of body.matchAll(/_fail\s*\(\s*[fF]?["']([^"']+)["']/g)) add(m[1]);
  for (const args of positionalCallArgs(body, "_unavailable")) if (args[2]) literals(args[2]);
  const hand = body.match(/^def _hand_reason\([\s\S]*?(?=^def |$(?![\s\S]))/m)?.[0] ?? "";
  for (const m of hand.matchAll(/return\s+[fF]?["']([^"']+)["']/g)) add(m[1]);
  for (const line of body.split(/\r?\n/)) if (/^\s*\w*reason\s*=/.test(line)) literals(line);
  for (const source of [body, pose]) {
    for (const m of source.matchAll(/\breason(?:\s*:[^=\r\n]+)?\s*=\s*[fF]?["']([^"']+)["']/g)) add(m[1]);
  }
  for (const m of session.matchAll(/["']reason["']\s*:\s*[fF]?["']([^"']+)["']/g)) add(m[1]);
  return [...reasons].sort();
}
function assertBodyReasonInventory(sources: string[]) {
  const reasons = scanBodyReasons(sources);
  expect(reasons).toEqual(EXPECTED_BODY_REASONS);
  for (const reason of reasons) expect(bodyEvidenceText({ reason }, []), reason).not.toContain("unknown reason:");
}
