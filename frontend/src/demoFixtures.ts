/** Static manufactured data only. No media, camera, transport or persistence. */
import { LEGACY8, presentResult, type Presentation } from "./resultPresentation";

export const DEMO_LABEL = "SYNTHETIC DEMO - fabricated scores, not accuracy or a camera reading";
export type DemoFixture = {
  id: string; label: string; response: unknown; error?: string; expectedKind: Presentation["kind"];
};
const vector = (top: string) => Object.fromEntries(LEGACY8.map(name => [name, Number(name === top)]));
const response = (face: Record<string, unknown> = {}, expression: Record<string, unknown> = {}) => ({
  synthetic_only: true, selected_model: "SYNTHETIC-demo", ethics_notice: DEMO_LABEL,
  face: { face_found: true, box: null, aus: [], quality: .9,
    insufficient_evidence: { insufficient: false, reasons: [] }, ...face,
    expression: { probs: vector("happiness"), confidence: 1, uncertainty: 0,
      source: "SYNTHETIC manufactured values - no model ran", ...expression } },
});
function freeze<T>(value: T): T {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze); Object.freeze(value);
  }
  return value;
}

export const DEMO_FIXTURES: readonly DemoFixture[] = freeze<DemoFixture[]>([
  { id: "usable", label: "Usable fabricated response", expectedKind: "usable", response: response() },
  { id: "no-face", label: "Simulated no-face response", expectedKind: "no-face",
    response: response({ face_found: false, quality: 0, insufficient_evidence: { insufficient: true, reasons: ["no-face-detected"] } },
      { probs: vector("unknown"), confidence: .15, uncertainty: .85, note: "Fabricated no-face placeholders; no camera or inference ran" }) },
  { id: "low-quality", label: "Simulated low detection confidence", expectedKind: "insufficient",
    response: response({ quality: .2, insufficient_evidence: { insufficient: true, reasons: ["poor-detection-quality"] } }) },
  { id: "abstained", label: "Simulated abstention", expectedKind: "abstained",
    response: response({ abstained: true, abstain_threshold: .7,
      insufficient_evidence: { insufficient: true, reasons: ["below-abstention-threshold"] } },
      { probs: { ...vector("happiness"), happiness: .6, neutral: .4 }, confidence: .6, uncertainty: .4 }) },
  { id: "unavailable", label: "Simulated unavailable service (no request made)", expectedKind: "unavailable", response: null,
    error: "Synthetic demo: simulated service unavailable; no backend request was made." },
  { id: "unsupported", label: "Fabricated response with explicit unsupported classes", expectedKind: "usable",
    response: response({}, { supported_classes: ["happy", "neutral"] }) },
  { id: "malformed", label: "Simulated malformed response", expectedKind: "invalid",
    response: response({}, { probs: { ...vector("happiness"), happiness: null } }) },
]);

export function demoPresentation(fixture: DemoFixture): Presentation {
  return presentResult({ phase: "waiting", response: fixture.response, error: fixture.error });
}
