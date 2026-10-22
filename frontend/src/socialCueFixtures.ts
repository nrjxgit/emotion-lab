/** Static manufactured social-cue blocks for the no-camera demo. No media, camera, transport or storage. */
import { DEFAULT_BANNER, parseCueBlock, type CueBlock } from "./socialCueLogic";

export type CueFixture = { id: string; label: string; block: CueBlock };

const block = (states: CueBlock["states"], insufficient: CueBlock["insufficient"]): CueBlock =>
  ({ states, insufficient, banner: DEFAULT_BANNER, heuristic: "HB-1 hypothesis, unvalidated" });
const ok = { flag: false, reasons: [] as string[] };

function freeze<T>(value: T): T {
  if (value && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}

export const CUE_FIXTURES: readonly CueFixture[] = freeze<CueFixture[]>([
  { id: "calm", label: "Calm (no cue)", block: block([], ok) },
  { id: "nervous-like", label: "Nervous-like pattern", block: block([
    { name: "nervous-like", confidence: 0.8, evidence: ["p_neg_arousal=0.714", "head_jitter=0.135", "arousal=0.700", "coverage=1.000", "quality=0.948"] }], ok) },
  { id: "unsure-like", label: "Unsure-like pattern", block: block([
    { name: "unsure-like", confidence: 0.54, evidence: ["p_neutral=0.248", "mean_top_prob=0.350", "hesitation=1.000", "gaze_away[head-orientation proxy]=0.417", "arousal=0.150"] }], ok) },
  { id: "needs-help-like", label: "Needs-help-like escalation", block: block([
    { name: "unsure-like", confidence: 0.54, evidence: ["hesitation=1.000", "gaze_away[head-orientation proxy]=0.417"] },
    { name: "needs-help-like", confidence: 0.8, evidence: ["dwell_s=13.875", "gaze_away_episodes=3", "sustained_fraction=1.000"] }], ok) },
  { id: "insufficient", label: "Insufficient evidence (low quality)", block: block([], { flag: true, reasons: ["low_quality"] }) },
  { id: "multi-face", label: "More than one person in view", block: block([], { flag: true, reasons: ["multiple_faces", "no_face"] }) },
  { id: "dropout", label: "Intermittent dropout (face lost on many frames)", block: block([], { flag: true, reasons: ["low_coverage"] }) },
]);

/** Every fixture must pass the same validator as a live response. */
export function fixtureBlock(id: string): CueBlock | null {
  const f = CUE_FIXTURES.find(x => x.id === id);
  return f ? parseCueBlock(JSON.parse(JSON.stringify(f.block))) : null;
}
