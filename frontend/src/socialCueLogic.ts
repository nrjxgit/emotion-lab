/** Social cues (research prototype) - pure logic for the panel. No DOM, no React, no storage.
 * The controller owns consent, the session id, the paced loop (exactly one request in flight),
 * pause on hidden tab, and the DELETE on stop. All side effects come in through `CueDeps`. */

export const SESSION_HEADER = "X-Socialcue-Session";
export const CUE_PERIOD_MS = 120; // ~8.3 fps target (design: 8-10 fps)
export const CONFIDENCE_CAP = 0.8;
export const ANNOUNCE_MIN_GAP_MS = 2000;
export const MAX_EVIDENCE = 12;
export const STATE_NAMES = ["nervous-like", "unsure-like", "needs-help-like"] as const;
export type CueName = (typeof STATE_NAMES)[number];

export const CONSENT_LABEL =
  "I am the only person in view and I agree to send camera frames to the configured local backend for this research prototype. Stop or unchecking this box ends capture. Read the capture notice for retention limits.";
export const WORDING = "Signs consistent with ..., not a reading of feelings";
export const ETHICS_NOTE =
  "Research prototype using an unvalidated heuristic. It looks at observable head and expression patterns over a few seconds, " +
  "not at feelings. Do not use it for decisions about people, and never point it at staff, students or job candidates.";
export const DEFAULT_BANNER =
  "Social cues (research prototype): signs consistent with observable behaviour, not a reading of feelings. Hypothesis heuristic (HB-1); not validated.";
export const SYNTHETIC_DEMO_LABEL = "SYNTHETIC DEMO - fabricated values, no camera and no request";

const ROW_LABEL: Record<CueName, string> = {
  "nervous-like": "nervous-like behaviour",
  "unsure-like": "unsure-like behaviour",
  "needs-help-like": "needs-help-like behaviour (sustained over several seconds)",
};
const REASON_TEXT: Record<string, string> = {
  multiple_faces: "More than one person in view: not analysed",
  no_face: "No face in view",
  window_filling: "Collecting frames (the window is still filling)",
  low_coverage: "Too few usable frames in the window",
  low_fps: "Frame rate too low for this window",
  low_quality: "Detection quality too low",
  no_session_id: "No session id was sent",
  invalid_session_id: "The session id was not accepted",
  internal_error: "The social-cue service hit an internal error",
  session_invalidated: "The session was reset or expired; cues restart with a new window",
};

export type CueState = { name: CueName; confidence: number; evidence: string[] };
export type CueBlock = { states: CueState[]; insufficient: { flag: boolean; reasons: string[] }; banner: string; heuristic: string; bodyText?: string };

const isObj = (x: unknown): x is Record<string, unknown> => !!x && typeof x === "object" && !Array.isArray(x);
const clip = (s: string, n: number) => (s.length > n ? s.slice(0, n) : s);

const BODY_REASON_TEXT: Record<string, string> = {
  no_pose_model: "no pose model installed - expected",
  pose_not_visible: "pose not visible", invalid_pose_payload: "invalid pose payload",
  consent_off: "consent is off", no_subject: "no subject", multiple_people: "multiple people",
  config_not_pinned: "configuration not pinned", internal_error: "internal error",
  window_filling: "collecting - needs about 2-3 s of frames", invalid_input: "invalid input",
  no_pose: "no pose available", "fps<5": "frame rate too low for body motion", "fps<8": "frame rate too low for body motion",
  no_shoulders: "shoulders not visible", identity_discontinuity: "person changed or jumped - reset",
  body_face_mismatch: "body does not match the face in view", occluded: "body partly hidden",
  hands_not_visible: "hands not visible", too_few_pairs: "too few frame pairs",
};

/** Read only existing body.cues and states[*].body_support.items; never infer a new state. */
export function bodyEvidenceText(body: unknown, states: unknown): string {
  if (!isObj(body) || body.status !== "ok") {
    const reason = isObj(body) && typeof body.reason === "string" ? clip(body.reason, 120) : "";
    let known = Object.prototype.hasOwnProperty.call(BODY_REASON_TEXT, reason) ? BODY_REASON_TEXT[reason] : null;
    // Backend currently parameterizes occluded coverage; retain bounded diagnostic suffixes.
    if (known === null) for (const prefix of ["occluded", "fps<5", "fps<8"]) {
      if (reason.startsWith(prefix + ":")) { known = BODY_REASON_TEXT[prefix] + clip(reason.slice(prefix.length), 100); break; }
    }
    return `Body evidence: unavailable (${known ?? (reason ? `unknown reason: ${reason}` : "reason not supplied")})`;
  }
  // Bound both iteration and text; malformed/oversized collections cannot manufacture support.
  const cues = Array.isArray(body.cues) && body.cues.length <= 64 ? body.cues : [];
  const present = new Set(cues.filter(c => isObj(c) && c.status === "ok" && c.present === true
    && typeof c.cue === "string" && c.cue.length > 0 && c.cue.length <= 120).map(c => c.cue as string));
  const support = new Set<string>();
  if (Array.isArray(states) && states.length <= 64) for (const state of states) {
    if (!isObj(state) || !(STATE_NAMES as readonly unknown[]).includes(state.name) || !isObj(state.body_support)) continue;
    const items = state.body_support.items;
    if (Array.isArray(items) && items.length <= 64 && items.some(i => typeof i === "string" && present.has(i))) {
      support.add(state.name as string);
    }
  }
  const unavailable = cues.filter(c => isObj(c) && (c.status === "none" || c.status === "unavailable")).length;
  return `Body evidence (hypothesis, unvalidated): ${present.size} cue(s) supporting ${STATE_NAMES.filter(n => support.has(n)).join(", ") || "none"}`
    + (present.size === 0 && unavailable > 0 ? `; ${unavailable} cue(s) unavailable` : "");
}

/** Validate and normalise the server's socialcue block; null when it is not understood. */
export function parseCueBlock(x: unknown): CueBlock | null {
  if (!isObj(x) || !Array.isArray(x.states) || !isObj(x.insufficient)) return null;
  const ins = x.insufficient;
  if (typeof ins.flag !== "boolean" || !Array.isArray(ins.reasons)) return null;
  const states: CueState[] = [];
  for (const s of x.states) {
    if (!isObj(s) || !(STATE_NAMES as readonly string[]).includes(s.name as string)) return null;
    const c = s.confidence;
    if (typeof c !== "number" || !Number.isFinite(c)) return null;
    const ev = Array.isArray(s.evidence) ? s.evidence.filter((e): e is string => typeof e === "string") : [];
    states.push({ name: s.name as CueName, confidence: Math.min(CONFIDENCE_CAP, Math.max(0, c)),
      evidence: ev.slice(0, MAX_EVIDENCE).map(e => clip(e, 120)) });
  }
  return {
    states,
    bodyText: bodyEvidenceText(x.body, x.insufficient.flag ? [] : x.states),
    insufficient: { flag: ins.flag, reasons: ins.reasons.filter((r): r is string => typeof r === "string").slice(0, 8) },
    banner: typeof x.banner === "string" && x.banner ? clip(x.banner, 400) : DEFAULT_BANNER,
    heuristic: typeof x.heuristic === "string" ? clip(x.heuristic, 80) : "HB-1 hypothesis, unvalidated",
  };
}

export type CueRow = {
  name: CueName; label: string; active: boolean; confidence: number | null; band: "low" | "medium" | "high" | null;
  text: string; evidence: string[];
};
export type CueView = { rows: CueRow[]; insufficient: boolean; reasonText: string; banner: string; summary: string };

export function reasonText(reasons: string[]): string {
  // More than one person in view is never reported as "no face", even if the server also lists no_face.
  if (reasons.includes("multiple_faces")) return REASON_TEXT.multiple_faces;
  const known = reasons.map(r => REASON_TEXT[r]).filter(Boolean);
  return known.length ? known.join("; ") : "Insufficient evidence";
}
const band = (c: number): "low" | "medium" | "high" => (c < 0.3 ? "low" : c < 0.55 ? "medium" : "high");

export function presentCueBlock(block: CueBlock | null): CueView {
  const insufficient = block === null ? true : block.insufficient.flag;
  const reason = block === null ? "No data yet" : insufficient ? reasonText(block.insufficient.reasons) : "";
  const rows = STATE_NAMES.map((name): CueRow => {
    const hit = block && !insufficient ? block.states.find(s => s.name === name) : undefined;
    if (hit) {
      return { name, label: name, active: true, confidence: hit.confidence, band: band(hit.confidence),
        text: `Signs consistent with ${ROW_LABEL[name]}, not a reading of feelings`, evidence: hit.evidence };
    }
    return { name, label: name, active: false, confidence: null, band: null,
      text: insufficient ? `No cue / insufficient evidence: ${reason}` : "No cue", evidence: [] };
  });
  const active = rows.filter(r => r.active).map(r => `${r.name} (${r.band})`);
  const summary = insufficient ? `No cue / insufficient evidence: ${reason}`
    : active.length ? `Signs consistent with: ${active.join(", ")}; not a reading of feelings` : "No cue";
  return { rows, insufficient, reasonText: reason, banner: block ? block.banner : DEFAULT_BANNER, summary };
}

/** Throttle polite announcements: only when the summary changed and the gap has passed. */
export function shouldAnnounce(prev: string, next: string, lastAt: number, now: number, gap = ANNOUNCE_MIN_GAP_MS): boolean {
  return next !== prev && now - lastAt >= gap;
}

export function cuesEnabled(search?: string): boolean {
  try {
    const q = search ?? (typeof window !== "undefined" ? window.location?.search ?? "" : "");
    return new URLSearchParams(q).get("socialcues") === "1";
  } catch { return false; }
}

/** crypto.randomUUID, else crypto.getRandomValues; never a non-cryptographic generator. Throws when neither exists. */
export function newSessionId(c: { randomUUID?: () => string; getRandomValues?: (a: Uint8Array) => Uint8Array } | undefined =
  (globalThis as { crypto?: any }).crypto): string {
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  if (c && typeof c.getRandomValues === "function") {
    const bytes = c.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
  }
  throw new Error("secure random source unavailable");
}

export type CueStatus = "idle" | "starting" | "running" | "paused" | "error";
/** Camera owns acquisition epochs: close invalidates pending grants; open discards stale grants. */
export type Camera = { open(): Promise<void>; frame(): Promise<Blob | null>; close(): void };
export type CueDeps = {
  api: string;
  fetchFn: (url: string, init: any) => Promise<any>;
  camera: Camera;
  newId?: () => string;
  setTimer: (fn: () => void, ms: number) => number;
  clearTimer: (id: number) => void;
  now?: () => number;
  makeAbort?: () => { signal: unknown; abort: () => void };
  makeForm?: (blob: Blob) => unknown;
  onBlock: (block: CueBlock | null) => void;
  onStatus: (status: CueStatus, message: string) => void;
  onConsent?: (consent: boolean) => void;
};

export class CueController {
  consent = false;
  sessionId: string | null = null;
  sent = false;
  running = false;
  paused = false;
  inFlight = false;
  private gen = 0;
  private pauseEpoch = 0;
  private hidden = false;
  private timer: number | undefined;
  private abort: { abort: () => void } | null = null;
  constructor(private d: CueDeps) {}

  private now() { return this.d.now ? this.d.now() : Date.now(); }
  setConsent(value: boolean) { this.consent = value; this.d.onConsent?.(value); if (!value) this.stop(); }
  /** A finished or failed run ends the consent: the next run needs a fresh agreement. */
  private endConsent() { this.consent = false; this.d.onConsent?.(false); }
  revoke() { this.setConsent(false); }

  async start(): Promise<boolean> {
    if (!this.consent) { this.d.onStatus("idle", "Consent is required before the camera or any request starts."); return false; }
    if (this.running) return true;
    const g = ++this.gen;
    this.d.onStatus("starting", "Starting the camera...");
    try { this.sessionId = (this.d.newId ?? newSessionId)(); }
    catch { this.sessionId = null; this.endConsent(); this.d.onStatus("error", "A secure random session id is not available in this browser."); return false; }
    this.sent = false;
    try { await this.d.camera.open(); }
    catch { if (g === this.gen) { this.sessionId = null; this.endConsent(); this.d.onStatus("error", "Camera unavailable. Check the camera permission, then agree again to retry."); } return false; }
    // stop/revoke already invalidated that acquisition; never close a newer camera here.
    if (g !== this.gen || !this.consent) return false;
    this.running = true; this.paused = this.hidden;
    if (this.paused) {
      this.d.onStatus("paused", "Paused while this tab is hidden. No frames are sent.");
    } else {
      this.d.onStatus("running", "Running: about 8 frames per second are sent to the local backend.");
      this.schedule(g, 0);
    }
    return true;
  }

  private schedule(g: number, ms: number) {
    if (this.timer !== undefined) this.d.clearTimer(this.timer);
    this.timer = this.d.setTimer(() => { this.timer = undefined; void this.tick(g); }, ms);
  }

  /** One paced tick. Never starts a request while one is in flight. */
  async tick(g: number = this.gen): Promise<void> {
    if (g !== this.gen || !this.running || this.paused) return;
    if (this.inFlight) { this.schedule(g, CUE_PERIOD_MS); return; }
    this.inFlight = true;
    const t0 = this.now(), epoch = this.pauseEpoch;
    const current = () => g === this.gen && this.running && !this.paused && epoch === this.pauseEpoch;
    try {
      const blob = await this.d.camera.frame();
      if (!current()) return;
      if (blob) {
        const ctl = this.d.makeAbort ? this.d.makeAbort() : new AbortController();
        this.abort = ctl;
        const form: any = this.d.makeForm ? this.d.makeForm(blob) : new FormData();
        if (!this.d.makeForm) form.append("image", blob, "f.jpg");
        this.sent = true; // from here the server may hold a session for this id
        const res = await this.d.fetchFn(`${this.d.api}/api/infer?socialcue=1`, {
          method: "POST", body: form, headers: { [SESSION_HEADER]: this.sessionId as string }, signal: ctl.signal });
        if (!current()) return;
        if (!res.ok) throw new Error("http");
        const block = parseCueBlock((await res.json())?.socialcue);
        if (!current()) return;
        if (!block) throw new Error("shape");
        this.d.onBlock(block);
        this.d.onStatus("running", "Running: about 8 frames per second are sent to the local backend.");
      }
    } catch {
      if (current()) this.d.onStatus("error", "Social-cue request failed or the response was not understood. Still trying.");
    } finally {
      this.inFlight = false;
      this.abort = null;
    }
    if (g === this.gen && this.running && !this.paused) this.schedule(g, Math.max(0, CUE_PERIOD_MS - (this.now() - t0)));
  }

  /** Pause (tab hidden): no timer, in-flight request aborted. Resume restarts the paced loop. */
  setHidden(hidden: boolean) {
    this.hidden = hidden; // Remember visibility even while camera.open() is pending.
    if (!this.running) return;
    if (hidden && !this.paused) {
      this.paused = true;
      this.pauseEpoch++;
      if (this.timer !== undefined) { this.d.clearTimer(this.timer); this.timer = undefined; }
      this.abort?.abort();
      this.d.onStatus("paused", "Paused while this tab is hidden. No frames are sent.");
    } else if (!hidden && this.paused) {
      this.paused = false;
      this.d.onStatus("running", "Running: about 8 frames per second are sent to the local backend.");
      this.schedule(this.gen, 0);
    }
  }

  /** Stop: end the loop, abort, close the camera, clear the display and ask the backend to drop the session. */
  stop() {
    this.gen++;
    const id = this.sessionId, wasSent = this.sent;
    this.running = false; this.paused = false;
    if (this.timer !== undefined) { this.d.clearTimer(this.timer); this.timer = undefined; }
    this.abort?.abort(); this.abort = null;
    this.d.camera.close();
    this.sessionId = null; this.sent = false;
    this.d.onBlock(null);
    this.endConsent();
    this.d.onStatus("idle", "Stopped. The camera is off.");
    if (id && wasSent) {
      try {
        void Promise.resolve(this.d.fetchFn(`${this.d.api}/api/socialcue/session`, { method: "DELETE", headers: { [SESSION_HEADER]: id } })).catch(() => undefined);
      } catch { /* best effort: the backend also expires idle sessions */ }
    }
  }
}
