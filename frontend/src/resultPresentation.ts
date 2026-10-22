/** Pure U1 display boundary. No requests, inference, renormalisation or mutation.
 * Current API: legacy eight, including an application-residual `unknown`.
 * FER7 is accepted explicitly for fixtures/future clients, never conflated by index.
 */
export const FER7 = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"] as const;
export const LEGACY8 = ["happiness", "sadness", "anger", "surprise", "fear", "disgust", "neutral", "unknown"] as const;
export type Phase = "stopped" | "requesting-camera" | "waiting";
type ResponseKind = "usable" | "no-face" | "insufficient" | "abstained";
export type InferResponse = {
  selected_model: string; ethics_notice: string; supported_classes?: string[];
  face: {
    face_found: boolean; box: { x: number; y: number; w: number; h: number } | null;
    aus: { code: string; intensity: number }[]; quality: number;
    insufficient_evidence: { insufficient: boolean; reasons: string[] };
    abstained?: boolean; abstain_threshold?: number | null;
    expression: {
      probs: Record<string, number>; confidence: number; uncertainty: number; source: string;
      supported_classes?: string[]; valence?: number | null; arousal?: number | null;
      expressiveness?: number | null; profile?: string; calibrated?: boolean;
      temperature?: number; revision?: string; warning?: string; note?: string; experimental?: boolean;
    };
  };
};
export type Score = { label: string; value: number | null; support: "supported" | "unsupported" | "not-reported" | "residual" };
type Base = {
  statusText: string; message: string; reasons: string[]; raw: unknown; diagnostic?: string;
  scores: Score[]; classSpace: "legacy-eight" | "FER7" | null; supportText: string;
  confidence: number | null; uncertainty: number | null;
  historyPoint: { conf: number; top: string } | null;
};
type State<K extends string, R> = Base & { kind: K; response: R };
export type Presentation = State<"stopped", null> | State<"requesting-camera", null> | State<"waiting", null>
  | State<"unavailable", null> | State<"invalid", null>
  | { [K in ResponseKind]: State<K, InferResponse> }[ResponseKind];

const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const unit = (value: unknown): value is number => finite(value) && value >= 0 && value <= 1;
const text = (value: unknown): value is string => typeof value === "string" && value.length > 0 && value.length <= 8192;
const canonical = (name: string) => ({ anger: "angry", happiness: "happy", sadness: "sad" } as Record<string, string>)[name] || name;
const statuses = {
  stopped: "Stopped", "requesting-camera": "Requesting camera", waiting: "Waiting for current result",
  usable: "Current apparent-expression response", "no-face": "No face detected",
  insufficient: "Insufficient evidence", abstained: "Abstained",
  unavailable: "Inference unavailable", invalid: "Invalid response — inference unavailable",
};
const reasons: Record<string, string> = {
  "no-face-detected": "No face detected",
  "poor-detection-quality": "Low face-detection confidence",
  "high-uncertainty": "Top score below the server's evidence threshold",
  "possible-occlusion": "Possible occlusion reported",
  "below-abstention-threshold": "Score below the selected abstention threshold",
};
const readableReason = (reason: string) => Object.prototype.hasOwnProperty.call(reasons, reason) ? reasons[reason] : `Unknown evidence status: ${reason.replace(/[\u0000-\u001f\u007f]/g, " ").slice(0, 160)}`;

function empty(kind: Phase | "invalid" | "unavailable", raw: unknown, message: string, statusText = statuses[kind]): Presentation {
  return { kind, raw, message, statusText, response: null, reasons: [], scores: [], classSpace: null,
    supportText: "Class support not reported", confidence: null, uncertainty: null, historyPoint: null };
}

function validate(raw: unknown): { response: InferResponse; space: "legacy-eight" | "FER7"; support: Set<string> | null } | string {
  if (!record(raw) || !text(raw.selected_model) || !text(raw.ethics_notice) || !record(raw.face)) return "Response structure could not be read.";
  const f = raw.face;
  if (typeof f.face_found !== "boolean" || !record(f.expression) || !unit(f.quality) || !Array.isArray(f.aus) || f.aus.length > 128) return "Face response could not be read.";
  if (f.box !== null && (!record(f.box) || ![f.box.x, f.box.y, f.box.w, f.box.h].every(finite) || (f.box.w as number) <= 0 || (f.box.h as number) <= 0)) return "Face box could not be read.";
  if (f.aus.some(a => !record(a) || !text(a.code) || !unit(a.intensity))) return "Observable action units could not be read.";
  const evidence = f.insufficient_evidence;
  if (!record(evidence) || typeof evidence.insufficient !== "boolean" || !Array.isArray(evidence.reasons) || evidence.reasons.length > 32 || evidence.reasons.some(x => !text(x))) return "Evidence status could not be read.";
  if (f.abstained !== undefined && typeof f.abstained !== "boolean") return "Abstention status could not be read.";
  if (f.abstain_threshold !== undefined && f.abstain_threshold !== null && !unit(f.abstain_threshold)) return "Abstention threshold could not be read.";
  const ex = f.expression;
  if (!record(ex.probs)) return "Class scores unavailable: response could not be read.";
  const probabilities = ex.probs;
  const keys = Object.keys(probabilities);
  const matches = (names: readonly string[]) => keys.length === names.length && names.every(name => Object.prototype.hasOwnProperty.call(probabilities, name));
  const space = matches(LEGACY8) ? "legacy-eight" : matches(FER7) ? "FER7" : null;
  if (!space || !Object.values(ex.probs).every(unit)) return "Class scores unavailable: invalid class space or probability.";
  // Fixed numerical tolerance only. Never change or renormalise any supplied score.
  if (Math.abs((Object.values(ex.probs) as number[]).reduce((a, b) => a + b, 0) - 1) > 1e-5) return "Class scores unavailable: probability mass is invalid.";
  if (!unit(ex.confidence)) return "Model score unavailable: confidence was not reported as a finite value.";
  if (!unit(ex.uncertainty)) return "Uncertainty unavailable: no finite value was reported.";
  if (!text(ex.source)) return "Response source unavailable.";
  for (const key of ["profile", "revision", "warning", "note"]) if (ex[key] !== undefined && typeof ex[key] !== "string") return "Response metadata could not be read.";
  if (ex.experimental !== undefined && typeof ex.experimental !== "boolean") return "Experimental metadata could not be read.";
  if (ex.calibrated !== undefined && typeof ex.calibrated !== "boolean") return "Calibration metadata could not be read.";
  if (ex.temperature !== undefined && (!finite(ex.temperature) || ex.temperature <= 0)) return "Temperature metadata could not be read.";
  for (const key of ["valence", "arousal", "expressiveness"]) if (ex[key] !== undefined && ex[key] !== null && (!finite(ex[key]) || (ex[key] as number) < (key === "valence" ? -1 : 0) || (ex[key] as number) > 1)) return "Inspection quantity could not be read.";
  const declared = [ex.supported_classes, raw.supported_classes].filter(x => x !== undefined);
  let support: Set<string> | null = null;
  for (const value of declared) {
    if (!Array.isArray(value) || !value.length || value.some(x => typeof x !== "string" || ![...FER7, ...LEGACY8].includes(x as never))) return "Class support could not be read.";
    const normalized = new Set((value as string[]).map(canonical));
    if (normalized.size !== value.length || (space === "FER7" && normalized.has("unknown"))) return "Class support could not be read.";
    if (support && (support.size !== normalized.size || [...support].some(x => !normalized.has(x)))) return "Class support declarations disagree.";
    support = normalized;
  }
  if (support && Object.entries(ex.probs).some(([name, value]) => name !== "unknown" && !support!.has(canonical(name)) && value !== 0)) return "Class scores contradict declared support.";
  return { response: raw as unknown as InferResponse, space, support };
}

export function presentResult(input: { phase: Phase; response?: unknown; error?: string; unavailableStatus?: "Camera unavailable" | "Inference unavailable" }): Presentation {
  if (input.error) return empty("unavailable", input.response, input.error, input.unavailableStatus || statuses.unavailable);
  if (input.phase !== "waiting" || input.response === undefined) return empty(input.phase, undefined, input.phase === "stopped" ? "No current response." : "Waiting for a current response.");
  const checked = validate(input.response);
  if (typeof checked === "string") return empty("invalid", input.response, checked);
  const { response, space, support } = checked;
  const f = response.face, ex = f.expression;
  // Documented legacy not-run sentinel, not a source whitelist or readiness proof.
  if (f.face_found && space === "legacy-eight" && ex.probs.unknown === 1 && Object.entries(ex.probs).every(([name, value]) => name === "unknown" || value === 0) && Math.abs(ex.confidence - .15) < 1e-6) {
    return empty("unavailable", input.response, "Model unavailable: the response contains a legacy not-run placeholder, not expression scores.");
  }
  const readable = f.insufficient_evidence.reasons.map(readableReason);
  const kind: ResponseKind = !f.face_found ? "no-face" : f.abstained ? "abstained" : f.insufficient_evidence.insufficient || readable.length ? "insufficient" : "usable";
  const scores: Score[] = (space === "legacy-eight" ? LEGACY8 : FER7).map(label => {
    const state = label === "unknown" ? "residual" : support === null ? "not-reported" : support.has(canonical(label)) ? "supported" : "unsupported";
    return { label, support: state, value: state === "unsupported" ? null : ex.probs[label] };
  });
  const top = scores.filter(score => score.value !== null).sort((a, b) => b.value! - a.value!)[0];
  const message = kind === "no-face" ? "No face detected — no expression inference ran. Placeholder values are inspection only, not an emotion reading."
    : kind === "abstained" ? `Abstained at threshold ${f.abstain_threshold ?? "not reported"} — values are inspection only, not an emotion reading.`
    : kind === "insufficient" ? "Insufficient evidence — values are inspection only, not an emotion reading."
    : "Apparent-expression model scores are not proof of internal feeling.";
  return { kind, response, raw: input.response, statusText: statuses[kind], message, reasons: readable,
    scores: kind === "no-face" ? [] : scores, classSpace: space,
    supportText: support === null ? "Class support not reported; a zero is not evidence of non-support." : "Class support explicitly reported; unsupported classes are unavailable.",
    confidence: kind === "no-face" ? null : ex.confidence, uncertainty: kind === "no-face" ? null : ex.uncertainty,
    historyPoint: kind === "usable" && top && top.support !== "residual" ? { conf: ex.confidence, top: top.label } : null };
}

export function inspectionText(raw: unknown): string {
  try { return JSON.stringify(raw, (_key, value) => typeof value === "string" ? sanitiseDiagnostic(value) : typeof value === "number" && !Number.isFinite(value) ? String(value) : value, 2) ?? "No raw response."; }
  catch { return "Raw response could not be displayed."; }
}


/** Bounded display-only diagnostics: never stringify arbitrary server objects. */
export function sanitiseDiagnostic(value: unknown): string {
  if (typeof value !== "string") return "";
  return value.slice(0, 8192).split(/\r?\n/).filter(line =>
    !/^\s*(at\s|File\s|Traceback\b)/i.test(line)).join(" ")
    .replace(/[a-zA-Z]:[\\/][^\s<>"']*/g, "[path removed]")
    .replace(/(^|[\s(=])(?:\\\\|\/)[^\s<>"']*/g, "$1[path removed]")
    .replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g, " ")
    .replace(/\s+/g, " ").trim().slice(0, 500);
}
export function errorDiagnostic(error: unknown): string {
  return error instanceof Error ? sanitiseDiagnostic(error.message) : "No diagnostic reported.";
}
export class RequestFailure extends Error {
  constructor(message: string, public diagnostic = "") { super(message); }
}
export function httpFailure(status: number, payload: unknown, unparseable = false): RequestFailure {
  const code = Number.isInteger(status) && status >= 100 && status <= 599 ? status : 0;
  const message = code === 422 ? "Inference unavailable: request rejected (HTTP 422). Check the selected options or captured frame."
    : code === 501 ? "Inference unavailable: model not integrated (HTTP 501)."
    : code === 503 ? "Inference unavailable: requested model unavailable (HTTP 503)."
    : code === 413 ? "Inference unavailable: captured frame too large (HTTP 413)."
    : unparseable && code >= 200 && code < 300 ? "Inference unavailable: response could not be read (unparseable JSON)."
    : `Inference unavailable${code ? ` (HTTP ${code})` : ""}. Check the local backend and try again.`;
  const detail = record(payload) ? payload.detail : undefined;
  const item = (value: unknown) => {
    if (typeof value === "string") return sanitiseDiagnostic(value);
    if (!record(value)) return "";
    const msg = sanitiseDiagnostic(typeof value.msg === "string" ? value.msg : value.message);
    // Locations are bounded scalar labels only; input/context/stack fields are ignored.
    const loc = Array.isArray(value.loc) ? value.loc.slice(0, 8).filter(x => typeof x === "string" || typeof x === "number")
      .map(x => sanitiseDiagnostic(String(x))).join(" > ") : "";
    return msg ? `${loc ? loc + ": " : ""}${msg}` : "";
  };
  const diagnostic = (Array.isArray(detail) ? detail.slice(0, 8).map(item).filter(Boolean).join("; ") : item(detail)).slice(0, 500);
  return new RequestFailure(message, diagnostic || (unparseable ? "Response body was not readable JSON; raw body omitted." : "No readable diagnostic reported."));
}
export function cameraFailure(error: unknown): RequestFailure {
  const denied = error instanceof Error && ["NotAllowedError", "PermissionDeniedError", "SecurityError"].includes(error.name);
  return new RequestFailure(denied ? "Camera permission denied. Review this site's camera permission in your browser, then try Start again."
    : "Camera unavailable. Check camera access and whether another application is using it, then try Start again.", errorDiagnostic(error));
}


/** Presentation of self-reported metadata, never asset/runtime authentication. */
export function responseProvenance(response: InferResponse | null, capturedProfile?: string) {
  const ex = response?.face.expression;
  const reported = (value: unknown) => sanitiseDiagnostic(value) || "not reported";
  const source = reported(ex?.source);
  const knownRoute = response?.face.face_found && (
    (response.selected_model === "mini-xception" && ex?.source === "trained-minixception-fer2013") ||
    (response.selected_model === "vit" && ex?.source === "hf-vit-dima806"));
  // A documented source name is self-reported, not proof that a model ran.
  const max = ex ? Math.max(...Object.values(ex.probs)) : NaN;
  const complement = knownRoute && ex && Math.abs(ex.confidence - max) <= 1e-5 && Math.abs(ex.uncertainty - (1 - max)) <= 1e-5;
  const warning = reported(ex?.warning);
  const experimental = response?.selected_model === "vit" || ex?.experimental === true;
  return {
    model: reported(response?.selected_model), capturedProfile: reported(capturedProfile), source,
    responseProfile: reported(ex?.profile),
    calibration: ex?.calibrated === true ? "calibrated (source-reported)" : ex?.calibrated === false ? "raw / uncalibrated (source-reported)" : "not reported",
    temperature: ex?.temperature === undefined ? "not reported" : String(ex.temperature),
    revision: reported(ex?.revision), warning,
    visibleWarning: experimental ? `Experimental response; possible training overlap. Source warning: ${warning}` : warning !== "not reported" ? `Source warning: ${warning}` : "",
    uncertaintyDefinition: complement
      ? "For this reported source, the documented API defines uncertainty as 1 - max(class score). The supplied numbers agree within 1e-5; this is not independent uncertainty evidence."
      : "Definition not reported for this response. Treat uncertainty as a source-reported quantity only.",
  };
}
