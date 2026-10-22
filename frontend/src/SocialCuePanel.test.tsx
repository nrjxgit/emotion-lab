// Manufactured hook/stream fixtures only: no DOM, real browser, camera or network.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
const hooks = vi.hoisted(() => ({ states: [] as any[], refs: [] as any[], effects: [] as any[], pending: [] as any[], s: 0, r: 0, e: 0 }));
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
import SocialCuePanel, { SocialCueInner } from "./SocialCuePanel";
import { CUE_FIXTURES } from "./socialCueFixtures";
import { SESSION_HEADER, SYNTHETIC_DEMO_LABEL, CONSENT_LABEL } from "./socialCueLogic";

const BLOCK = { states: [{ name: "nervous-like", confidence: 0.6, evidence: ["<b>p_neg_arousal=0.700</b>"] }],
  insufficient: { flag: false, reasons: [] }, banner: "server banner <script>x</script>", heuristic: "HB-1 hypothesis, unvalidated" };
let tree: any, video: any, canvas: any, media: any, fetcher: any, timers: Map<number, () => void>, seq: number, track: any;

function render() {
  hooks.s = hooks.r = hooks.e = 0; tree = SocialCueInner({ api: "http://api.test" });
  for (const { i, fn, deps } of hooks.pending.splice(0)) { hooks.effects[i]?.cleanup?.(); hooks.effects[i] = { deps, cleanup: fn() }; }
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
const byId = (id: string) => nodes(tree).find(n => n.props?.id === id);
async function settle() { for (let i = 0; i < 25; i++) await Promise.resolve(); render(); }
async function fire() { const cbs = [...timers.values()]; timers.clear(); cbs.forEach(fn => fn()); await settle(); }
async function consentAndStart() {
  byId("cue-consent").props.onChange({ target: { checked: true } }); render();
  byId("cue-start").props.onClick(); await settle();
}
const posts = () => fetcher.mock.calls.filter(([, i]: any) => i.method === "POST");
const deletes = () => fetcher.mock.calls.filter(([, i]: any) => i.method === "DELETE");

beforeEach(() => {
  hooks.states = []; hooks.refs = []; hooks.effects = []; hooks.pending = [];
  timers = new Map(); seq = 0; track = { stop: vi.fn() };
  video = { videoWidth: 320, srcObject: null };
  canvas = { getContext: vi.fn(() => ({ drawImage: vi.fn() })), toBlob: (cb: any) => cb(new Blob(["manufactured pixels"])) };
  media = vi.fn().mockResolvedValue({ getTracks: () => [track] });
  fetcher = vi.fn(async (_u: string, init: any) => init.method === "DELETE" ? { ok: true } : { ok: true, json: async () => ({ socialcue: BLOCK }) });
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: media } });
  vi.stubGlobal("window", { setTimeout: vi.fn((fn: any) => { timers.set(++seq, fn); return seq; }), clearTimeout: vi.fn((id: number) => timers.delete(id)) });
  vi.stubGlobal("fetch", fetcher);
  render(); hooks.refs[0].current = video; hooks.refs[1].current = canvas;
});
afterEach(() => { hooks.effects.forEach(e => e.cleanup?.()); vi.unstubAllGlobals(); vi.clearAllMocks(); });

describe("opt-in panel", () => {
  it("renders nothing unless enabled", () => {
    expect(SocialCuePanel({ api: "x", enabled: false })).toBeNull();
    const on: any = SocialCuePanel({ api: "x", enabled: true });
    expect(on.type).toBe(SocialCueInner);
  });
  it("starts consent-less with no request, camera or timer", () => {
    expect(text(tree)).toContain("Social cues (research prototype)");
    expect(text(byId("cue-ethics"))).toContain("Do not use it for decisions about people");
    expect(byId("cue-consent").props.checked).toBe(false);
    expect(text(nodes(tree).find(n => n.type === "label" && n.props.htmlFor === "cue-consent"))).toContain(CONSENT_LABEL);
    expect(byId("cue-start").props.disabled).toBe(true); expect(byId("cue-stop").props.disabled).toBe(true);
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); expect(timers.size).toBe(0);
    expect(text(tree)).toContain("No cue / insufficient evidence");
  });
});

describe("consent gate", () => {
  it("clicking Start without consent still sends nothing", async () => {
    byId("cue-start").props.onClick(); await settle();
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); expect(timers.size).toBe(0);
    expect(text(byId("cue-state-message"))).toContain("Consent is required");
  });
  it("consent enables Start; the first request carries the session header and the flag", async () => {
    await consentAndStart();
    expect(media).toHaveBeenCalledTimes(1); expect(video.srcObject).not.toBeNull(); expect(fetcher).not.toHaveBeenCalled();
    await fire();
    expect(posts()).toHaveLength(1);
    const [url, init] = posts()[0];
    expect(url).toBe("http://api.test/api/infer?socialcue=1"); expect(init.headers[SESSION_HEADER].length).toBeGreaterThanOrEqual(22);
    expect(text(tree)).toContain("Signs consistent with nervous-like behaviour, not a reading of feelings");
    expect(byId("cue-stop").props.disabled).toBe(false); expect(byId("cue-start").props.disabled).toBe(true);
  });
  it("revoking consent stops the loop, stops the camera, clears the display and calls DELETE", async () => {
    await consentAndStart(); await fire();
    const id = posts()[0][1].headers[SESSION_HEADER];
    byId("cue-consent").props.onChange({ target: { checked: false } }); await settle();
    expect(track.stop).toHaveBeenCalled(); expect(video.srcObject).toBeNull(); expect(timers.size).toBe(0);
    expect(deletes()).toHaveLength(1); expect(deletes()[0][0]).toBe("http://api.test/api/socialcue/session"); expect(deletes()[0][1].headers[SESSION_HEADER]).toBe(id);
    expect(text(tree)).not.toContain("Signs consistent with nervous-like"); expect(byId("cue-start").props.disabled).toBe(true);
    // the live region itself must not keep the old cue (a stale announcement would fail here)
    expect(text(byId("cue-status"))).toBe("Stopped. The camera is off."); expect(text(byId("cue-status"))).not.toContain("nervous-like");
    await fire(); expect(posts()).toHaveLength(1);
  });
  it("Stop also deletes, resets consent, and the next run needs a fresh agreement and a new session id", async () => {
    await consentAndStart(); await fire(); const a = posts()[0][1].headers[SESSION_HEADER];
    byId("cue-stop").props.onClick(); await settle(); expect(deletes()).toHaveLength(1);
    expect(byId("cue-consent").props.checked).toBe(false); expect(byId("cue-start").props.disabled).toBe(true);
    expect(text(byId("cue-status"))).toBe("Stopped. The camera is off.");
    byId("cue-start").props.onClick(); await settle(); expect(media).toHaveBeenCalledTimes(1); // a direct call is refused too
    await consentAndStart(); await fire();
    expect(posts()[1][1].headers[SESSION_HEADER]).not.toBe(a);
  });
  it("a camera failure ends the run, resets consent and shows the message in the live region", async () => {
    media.mockRejectedValueOnce(new Error("denied"));
    await consentAndStart();
    expect(byId("cue-consent").props.checked).toBe(false); expect(fetcher).not.toHaveBeenCalled();
    expect(text(byId("cue-status"))).toContain("Camera unavailable"); expect(nodes(tree).some(n => n.props?.role === "alert")).toBe(false);
  });
  it("states what is stored and when the camera stops in the consent label itself, linked to the capture notice", () => {
    const label = text(nodes(tree).find(n => n.type === "label" && n.props.htmlFor === "cue-consent"));
    expect(label).toContain("send camera frames to the configured local backend"); expect(label).toContain("retention limits");
    expect(byId("cue-consent").props["aria-describedby"]).toBe("cue-capture-notice cue-controls-help");
  });
  it("unmounting cleans up like Stop", async () => {
    await consentAndStart(); await fire();
    hooks.effects[0].cleanup(); await settle();
    expect(track.stop).toHaveBeenCalled(); expect(deletes()).toHaveLength(1);
  });
});

describe("synthetic demo", () => {
  it("renders every scenario with no camera, request, timer or session", async () => {
    const select = byId("cue-demo");
    expect(select.props.value).toBe("");
    for (const f of CUE_FIXTURES) {
      select.props.onChange({ target: { value: f.id } }); render();
      expect(text(byId("cue-banner"))).toBe(SYNTHETIC_DEMO_LABEL); expect(text(byId("cue-status"))).toBe(SYNTHETIC_DEMO_LABEL);
      for (const s of f.block.states) expect(text(tree)).toContain(s.name);
    }
    expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled(); expect(timers.size).toBe(0);
    expect(byId("cue-start").props.disabled).toBe(true); expect(byId("cue-consent").props.disabled).toBe(true);
  });
  it("shows the multiple-people message and never 'no face' for that scenario", () => {
    byId("cue-demo").props.onChange({ target: { value: "multi-face" } }); render();
    const t = text(tree);
    expect(t).toContain("More than one person in view: not analysed"); expect(t.toLowerCase()).not.toContain("no face");
    expect(t).not.toContain("Signs consistent with nervous-like");
  });
  it("shows evidence under a native Why disclosure with the 0-0.8 scale noted", () => {
    byId("cue-demo").props.onChange({ target: { value: "needs-help-like" } }); render();
    const details = nodes(tree).filter(n => n.type === "details");
    expect(details.length).toBe(2); expect(text(details[1])).toContain("dwell_s=13.875");
    expect(text(tree)).toContain("scale 0 to 0.8"); expect(text(tree)).toContain("confidence 0.80");
  });
  it("cannot be combined with a running live session", async () => {
    await consentAndStart();
    expect(byId("cue-demo").props.disabled).toBe(true);
  });
});

describe("live region (review U1)", () => {
  const controller = () => hooks.refs[2].current;
  it("shows the pause message, drops the stale cue, and resumes with the next change", async () => {
    await consentAndStart(); await fire();
    expect(text(byId("cue-status"))).toContain("nervous-like");
    controller().setHidden(true); render();
    expect(text(byId("cue-status"))).toBe("Paused while this tab is hidden. No frames are sent.");
    expect(text(byId("cue-status"))).not.toContain("nervous-like");
    controller().setHidden(false); render();
    expect(text(byId("cue-status"))).toContain("Running");
    await fire(); expect(text(byId("cue-status"))).toContain("nervous-like"); // the new block announces again after the reset
  });
  it("clearing the display is never throttled: Stop right after an announcement shows the stopped message", async () => {
    await consentAndStart(); await fire();
    expect(text(byId("cue-status"))).toContain("nervous-like"); // announced a moment ago, inside the 2 s gap
    byId("cue-stop").props.onClick(); await settle();
    expect(text(byId("cue-status"))).toBe("Stopped. The camera is off.");
  });
  it("shows request errors in the live region instead of a cue from before", async () => {
    await consentAndStart(); await fire();
    fetcher.mockImplementation(async (_u: string, i: any) => i.method === "DELETE" ? { ok: true } : { ok: false, status: 500, json: async () => ({}) });
    await fire();
    expect(text(byId("cue-status"))).toContain("request failed"); expect(text(byId("cue-status"))).not.toContain("Signs consistent");
  });
  it("offers a dropout demo that renders the insufficient reason without any request", () => {
    byId("cue-demo").props.onChange({ target: { value: "dropout" } }); render();
    expect(text(tree)).toContain("Too few usable frames in the window"); expect(text(tree)).not.toContain("Signs consistent with nervous-like");
    expect(text(byId("cue-status"))).toBe(SYNTHETIC_DEMO_LABEL); expect(media).not.toHaveBeenCalled(); expect(fetcher).not.toHaveBeenCalled();
    const options = nodes(byId("cue-demo")).filter(n => n.type === "option").map(n => n.props.value);
    expect(options).toEqual(["", ...CUE_FIXTURES.map(f => f.id)]); expect(options).toContain("dropout");
  });
});

describe("accessibility and safety of the markup", () => {
  it("has a polite status region, native keyboard-operable controls and hidden media elements", () => {
    const status = byId("cue-status");
    expect(status.props.role).toBe("status"); expect(status.props["aria-live"]).toBe("polite");
    const clickable = nodes(tree).filter(n => typeof n.props?.onClick === "function");
    expect(clickable.length).toBeGreaterThan(0); clickable.forEach(n => expect(n.type).toBe("button"));
    nodes(tree).filter(n => n.type === "button").forEach(b => expect(b.props.type).toBe("button"));
    expect(nodes(tree).find(n => n.type === "section").props["aria-labelledby"]).toBe("cues-heading");
    expect(byId("cue-start").props["aria-describedby"]).toBe("cue-capture-notice cue-controls-help");
    expect(nodes(tree).find(n => n.type === "video").props.hidden).toBe(true);
  });
  it("renders server strings only as React text children, never as HTML", async () => {
    await consentAndStart(); await fire();
    expect(nodes(tree).some(n => n.props && "dangerouslySetInnerHTML" in n.props)).toBe(false);
    const t = text(tree);
    expect(t).toContain("server banner"); // the banner is shown as plain text
    expect(typeof byId("cue-banner").props.children).toBe("string");
    expect(nodes(tree).filter(n => n.type === "li").every(n => !n.props.dangerouslySetInnerHTML)).toBe(true);
  });
  it("announces state changes politely and not more than once per throttle window", async () => {
    await consentAndStart(); await fire();
    const first = text(byId("cue-status"));
    expect(first).toContain("nervous-like"); fetcher.mockImplementation(async (_u: string, i: any) => i.method === "DELETE" ? { ok: true } :
      { ok: true, json: async () => ({ socialcue: { ...BLOCK, states: [], insufficient: { flag: true, reasons: ["low_quality"] } } }) });
    await fire(); expect(text(byId("cue-status"))).toBe(first); // changed within the gap: not re-announced
  });
});


describe("S21 review regressions", () => {
  it("keeps Stop and revoke available during request failure and refuses demo even through its handler", async () => {
    fetcher.mockRejectedValue(new Error("SYNTHETIC request failure"));
    await consentAndStart(); await fire();
    expect(text(byId("cue-status"))).toContain("request failed");
    expect(byId("cue-stop").props.disabled).toBe(false);
    expect(byId("cue-consent").props.disabled).toBe(false);
    expect(byId("cue-demo").props.disabled).toBe(true);
    byId("cue-demo").props.onChange({ target: { value: "nervous-like" } }); render();
    expect(text(byId("cue-banner"))).not.toBe(SYNTHETIC_DEMO_LABEL);
    await fire(); expect(posts()).toHaveLength(2);
    byId("cue-consent").props.onChange({ target: { checked: false } }); await settle();
    expect(track.stop).toHaveBeenCalled(); expect(video.srcObject).toBeNull();
    await fire(); expect(posts()).toHaveLength(2);
  });

  it.each([false, true])("rejects a pre-pause deferred JSON response (resume=%s)", async resume => {
    let resolve: (x: any) => void = () => {};
    fetcher.mockResolvedValue({ ok: true, json: () => new Promise(r => { resolve = r; }) });
    await consentAndStart(); await fire();
    const ctl = hooks.refs[2].current;
    ctl.setHidden(true); render();
    if (resume) { ctl.setHidden(false); render(); }
    resolve({ socialcue: BLOCK }); await settle();
    expect(text(byId("cue-status"))).not.toContain("nervous-like");
    expect(text(tree)).not.toContain("Signs consistent with nervous-like behaviour");
    if (!resume) expect(text(byId("cue-status"))).toContain("Paused");
  });
});


describe("S21 async boundary coverage", () => {
  it.each(["frame", "fetch"])("rejects a pre-hide %s result after hide-resume", async boundary => {
    let release: (x?: any) => void = () => {};
    const json = vi.fn(async () => ({ socialcue: BLOCK }));
    if (boundary === "frame") canvas.toBlob = (cb: any) => { release = () => cb(new Blob(["synthetic"])); };
    else fetcher.mockImplementation(() => new Promise(r => { release = () => r({ ok: true, json }); }));
    await consentAndStart(); await fire();
    const ctl = hooks.refs[2].current;
    ctl.setHidden(true); ctl.setHidden(false); render();
    release(); await settle();
    expect(json).not.toHaveBeenCalled();
    if (boundary === "frame") expect(posts()).toHaveLength(0);
    expect(text(tree)).not.toContain("Signs consistent with nervous-like behaviour");
  });
});


describe("S22 hidden during camera start", () => {
  it("starts paused after a hide while camera permission is pending, then resumes normally", async () => {
    let grant: (stream: any) => void = () => {};
    media.mockImplementation(() => new Promise(resolve => { grant = resolve; }));
    await consentAndStart();
    expect(text(byId("cue-status"))).toContain("Starting");
    const ctl = hooks.refs[2].current;
    ctl.setHidden(true); render();
    grant({ getTracks: () => [track] }); await settle();
    expect(ctl.running).toBe(true); expect(ctl.paused).toBe(true);
    expect(text(byId("cue-status"))).toContain("Paused");
    expect(byId("cue-stop").props.disabled).toBe(false);
    expect(byId("cue-demo").props.disabled).toBe(true);
    await fire(); expect(posts()).toHaveLength(0); expect(timers.size).toBe(0);
    ctl.setHidden(false); await fire();
    expect(ctl.paused).toBe(false); expect(posts()).toHaveLength(1);
    expect(text(byId("cue-status"))).toContain("nervous-like");
  });
});


describe("S22 pending camera visibility boundaries", () => {
  it("uses the latest visibility when hidden then visible before the grant", async () => {
    let grant: (stream: any) => void = () => {};
    media.mockImplementation(() => new Promise(resolve => { grant = resolve; }));
    await consentAndStart();
    const ctl = hooks.refs[2].current;
    ctl.setHidden(true); ctl.setHidden(false);
    grant({ getTracks: () => [track] }); await settle(); await fire();
    expect(ctl.running).toBe(true); expect(ctl.paused).toBe(false); expect(posts()).toHaveLength(1);
  });
  it("reports camera failure and ends consent even if hidden during start", async () => {
    let fail: (reason: any) => void = () => {};
    media.mockImplementation(() => new Promise((_resolve, reject) => { fail = reject; }));
    await consentAndStart();
    const ctl = hooks.refs[2].current;
    ctl.setHidden(true); fail(new Error("SYNTHETIC permission denied")); await settle();
    expect(text(byId("cue-status"))).toContain("Camera unavailable");
    expect(ctl.running).toBe(false); expect(byId("cue-consent").props.checked).toBe(false);
    await fire(); expect(posts()).toHaveLength(0); expect(timers.size).toBe(0);
  });
});


describe("S23 ownership and evidence regressions", () => {
  it.each(["old-first", "new-first"])("releases every track across Stop/restart in %s grant order", async order => {
    const tracksA = [{ stop: vi.fn() }, { stop: vi.fn() }], tracksB = [{ stop: vi.fn() }, { stop: vi.fn() }];
    const a = { getTracks: () => tracksA }, b = { getTracks: () => tracksB };
    const publications: any[] = []; let currentStream = video.srcObject;
    Object.defineProperty(video, "srcObject", { get: () => currentStream, set: value => { publications.push(value); currentStream = value; } });
    const grants: ((stream: any) => void)[] = [];
    media.mockImplementation(() => new Promise(resolve => grants.push(resolve)));
    await consentAndStart();
    byId("cue-stop").props.onClick(); await settle();
    await consentAndStart();
    if (order === "new-first") { grants[1](b); await settle(); expect(video.srcObject).toBe(b); }
    grants[0](a); await settle();
    tracksA.forEach(t => expect(t.stop).toHaveBeenCalled());
    expect(publications).not.toContain(a); // stale grant is never published, even transiently
    if (order === "old-first") { grants[1](b); await settle(); }
    expect(video.srcObject).toBe(b); tracksB.forEach(t => expect(t.stop).not.toHaveBeenCalled());
    byId("cue-stop").props.onClick(); await settle();
    [...tracksA, ...tracksB].forEach(t => expect(t.stop).toHaveBeenCalled());
    expect(video.srcObject).toBeNull(); await fire(); expect(posts()).toHaveLength(0);
  });
  it.each(["stop", "revoke", "unmount"])("stops all tracks granted after pending %s", async action => {
    const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }];
    let grant: (stream: any) => void = () => {};
    media.mockImplementation(() => new Promise(resolve => { grant = resolve; }));
    await consentAndStart();
    if (action === "stop") byId("cue-stop").props.onClick();
    else if (action === "revoke") byId("cue-consent").props.onChange({ target: { checked: false } });
    else hooks.effects[0].cleanup();
    grant({ getTracks: () => tracks }); await settle();
    tracks.forEach(t => expect(t.stop).toHaveBeenCalled()); expect(video.srcObject).toBeNull();
    await fire(); expect(posts()).toHaveLength(0);
  });
  it.each(["pause", "error"])("clears active rows and evidence on %s after a successful cue", async event => {
    await consentAndStart(); await fire();
    expect(text(tree)).toContain("Signs consistent with nervous-like behaviour");
    if (event === "pause") { hooks.refs[2].current.setHidden(true); render(); }
    else { fetcher.mockRejectedValue(new Error("SYNTHETIC outage")); await fire(); }
    const rows = nodes(tree).filter(n => n.type === "li" && n.props.className?.includes("cue-row"));
    expect(rows).toHaveLength(3);
    rows.forEach(row => { expect(row.props.className).not.toContain("cue-active"); expect(text(row)).not.toContain("confidence"); });
    expect(text(tree)).not.toContain("Signs consistent with nervous-like behaviour");
    expect(nodes(tree).filter(n => n.type === "details")).toHaveLength(0);
  });
  it("initialises visibility on an already-hidden mount without an event", async () => {
    hooks.effects.forEach(e => e.cleanup?.()); hooks.effects = [];
    const doc = { visibilityState: "hidden", addEventListener: vi.fn(), removeEventListener: vi.fn() };
    vi.stubGlobal("document", doc); render();
    await consentAndStart(); await fire();
    expect(hooks.refs[2].current.paused).toBe(true); expect(posts()).toHaveLength(0);
    expect(text(byId("cue-status"))).toContain("Paused");
  });
});


describe("body status line (S69)", () => {
  it("renders hostile reason strings as bounded React text, outside the live region", async () => {
    const hostile = "<script>alert(1)</script>";
    fetcher.mockResolvedValue({ ok: true, json: async () => ({ socialcue: { ...BLOCK,
      body: { status: "unavailable", reason: hostile + "x".repeat(1000) } } }) });
    await consentAndStart(); await fire();
    const line = byId("cue-body-status");
    expect(typeof line.props.children).toBe("string"); expect(text(line)).toContain(hostile);
    expect(text(line).length).toBeLessThan(200);
    expect(line.props.dangerouslySetInnerHTML).toBeUndefined();
    expect(line.props["aria-live"]).toBe("off"); expect(line.props.role).toBe("status");
    expect(nodes(tree).some(n => n.type === "script")).toBe(false);
    expect(text(byId("cue-status"))).not.toContain("Body evidence");
  });
  it.each(["pause", "error", "stop"])("clears previous body status on %s", async event => {
    fetcher.mockResolvedValue({ ok: true, json: async () => ({ socialcue: { ...BLOCK,
      body: { status: "unavailable", reason: "no_pose_model" } } }) });
    await consentAndStart(); await fire();
    expect(text(byId("cue-body-status"))).toContain("no pose model installed - expected");
    if (event === "pause") hooks.refs[2].current.setHidden(true);
    else if (event === "stop") byId("cue-stop").props.onClick();
    else { fetcher.mockRejectedValueOnce(new Error("manufactured failure")); await fire(); }
    await settle();
    expect(text(byId("cue-body-status"))).toBe("Body evidence: unavailable (no data yet)");
  });
  it("shows an explicit demo-only line without changing scenario fixtures", () => {
    for (const f of CUE_FIXTURES) {
      byId("cue-demo").props.onChange({ target: { value: f.id } }); render();
      expect(text(byId("cue-body-status"))).toBe("Body evidence: not part of the synthetic demo");
    }
    expect(fetcher).not.toHaveBeenCalled(); expect(media).not.toHaveBeenCalled();
  });
});


describe("U6 static accessibility gaps", () => {
  const descriptions = (id: string) => byId(id).props["aria-describedby"].split(" ").map((part: string) => {
    expect(byId(part)).toBeDefined(); return text(byId(part));
  }).join(" ");
  it("G1 associates reasons for stopped, ready, running and demo controls", async () => {
    for (const id of ["cue-consent", "cue-start", "cue-stop", "cue-demo"]) expect(descriptions(id)).toContain("no camera run is active");
    byId("cue-consent").props.onChange({ target: { checked: true } }); render();
    expect(descriptions("cue-start")).toContain("Ready to start");
    byId("cue-start").props.onClick(); await settle();
    expect(descriptions("cue-demo")).toContain("unavailable until then");
    byId("cue-stop").props.onClick(); await settle();
    byId("cue-demo").props.onChange({ target: { value: "needs-help-like" } }); render();
    expect(descriptions("cue-consent")).toContain("Select off to use the camera");
  });
  it("G2 restores a focused Stop to consent", async () => {
    await consentAndStart();
    const focus = vi.fn(); byId("cue-consent").ref.current = { focus };
    const stop = {}; byId("cue-stop").ref.current = stop;
    vi.stubGlobal("document", { activeElement: stop, removeEventListener: vi.fn() });
    byId("cue-stop").props.onClick(); await settle(); expect(focus).toHaveBeenCalledOnce();
  });
  it("G2 restores a focused failed Start to consent without stealing other focus", async () => {
    const focus = vi.fn(); byId("cue-consent").ref.current = { focus };
    const start = {}; byId("cue-start").ref.current = start;
    vi.stubGlobal("document", { activeElement: start, removeEventListener: vi.fn() });
    media.mockRejectedValueOnce(new Error("denied")); await consentAndStart();
    expect(focus).toHaveBeenCalledOnce();
    focus.mockClear(); (document as any).activeElement = {};
    media.mockRejectedValueOnce(new Error("denied")); await consentAndStart(); expect(focus).not.toHaveBeenCalled();
  });
  it("G3 associates errors with controls and keeps only the existing polite live region", async () => {
    media.mockRejectedValueOnce(new Error("denied")); await consentAndStart();
    for (const id of ["cue-consent", "cue-start", "cue-stop"]) expect(descriptions(id)).toContain("Camera unavailable");
    expect(byId("cue-error").props.role).toBeUndefined();
    expect(nodes(tree).filter(n => n.props?.["aria-live"] === "polite").map(n => n.props.id)).toEqual(["cue-status"]);
    expect(byId("cue-consent").props["aria-invalid"]).toBeUndefined();
  });
  it("G4 defines technical evidence without replacing raw measurements or adding announcements", () => {
    byId("cue-demo").props.onChange({ target: { value: "needs-help-like" } }); render();
    const why = nodes(tree).filter(n => n.type === "details");
    expect(text(why)).toContain("unvalidated research rule");
    expect(text(why)).toContain("p_neg_arousal: weighted expression-model score");
    expect(text(why)).toContain("dwell_s: seconds");
    expect(text(why)).toContain("HB-1"); expect(text(why)).toContain("dwell_s=13.875");
    expect(why.every(n => !n.props["aria-live"] && !n.props.role)).toBe(true);
  });
});
