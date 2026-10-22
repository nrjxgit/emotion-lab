// Manufactured response fixtures only. No browser, camera, transport or model.
import { describe, expect, it } from "vitest";
import { FER7, LEGACY8, presentResult, inspectionText } from "./resultPresentation";

const vector = (names: readonly string[], top: string) => Object.fromEntries(names.map(name => [name, Number(name === top)]));
function wire(face: Record<string, unknown> = {}, expression: Record<string, unknown> = {}) {
  return { selected_model: "mini-xception", ethics_notice: "Manufactured fixture — no real model reading",
    face: { face_found: true, box: null, aus: [], quality: .9,
      insufficient_evidence: { insufficient: false, reasons: [] }, ...face,
      expression: { probs: vector(LEGACY8, "happiness"), confidence: 1, uncertainty: 0,
        source: "trained-minixception-fer2013", ...expression } } };
}
const present = (response: unknown) => presentResult({ phase: "waiting", response });
const placeholder = { probs: vector(LEGACY8, "unknown"), confidence: .15, uncertainty: .85, source: "hdf5 missing" };

describe("pure evidence presentation", () => {
  it("distinguishes stopped/requesting/waiting without stale response", () => {
    for (const phase of ["stopped", "requesting-camera", "waiting"] as const) {
      const state = presentResult({ phase });
      expect(state.kind).toBe(phase); expect(state.historyPoint).toBeNull(); expect(state.response).toBeNull();
    }
    expect(presentResult({ phase: "stopped", response: wire() }).kind).toBe("stopped");
  });
  it("uses explicit unavailable/invalid > no-face > abstained > insufficient > usable precedence", () => {
    const overlap = wire({ face_found: false, abstained: true, abstain_threshold: .7,
      insufficient_evidence: { insufficient: true, reasons: ["high-uncertainty"] } });
    expect(presentResult({ phase: "waiting", response: overlap, error: "transport unavailable" }).kind).toBe("unavailable");
    expect(present({ ...overlap, face: { ...overlap.face, expression: { ...overlap.face.expression, probs: null } } }).kind).toBe("invalid");
    expect(present(overlap).kind).toBe("no-face");
    expect(present(wire({ abstained: true, abstain_threshold: .7, insufficient_evidence: { insufficient: true, reasons: ["below-abstention-threshold"] } })).kind).toBe("abstained");
    expect(present(wire({ insufficient_evidence: { insufficient: true, reasons: [] } })).kind).toBe("insufficient");
    expect(present(wire()).kind).toBe("usable");
  });
  it("recognises the documented not-run sentinel without a source whitelist", () => {
    const unavailable = present(wire({ insufficient_evidence: { insufficient: true, reasons: ["high-uncertainty"] } }, placeholder));
    expect(unavailable.kind).toBe("unavailable"); expect(unavailable.historyPoint).toBeNull(); expect(unavailable.scores).toEqual([]);
    expect(present(wire({ face_found: false }, placeholder)).kind).toBe("no-face");
  });
  it("no-face suppresses placeholder confidence and charts even without abstained fields", () => {
    const result = present(wire({ face_found: false, insufficient_evidence: { insufficient: true, reasons: ["no-face-detected"] } }, placeholder));
    expect(result.message).toContain("no expression inference ran");
    expect(result.confidence).toBeNull(); expect(result.uncertainty).toBeNull(); expect(result.historyPoint).toBeNull(); expect(result.scores).toEqual([]);
    expect(inspectionText(result.raw)).toContain("0.15");
  });
  it.each([undefined, null, NaN, Infinity, -1, 2])("missing/nonfinite/out-of-bounds confidence %s is unavailable, never zero", confidence => {
    const result = present(wire({}, { confidence }));
    expect(result.kind).toBe("invalid"); expect(result.message).toContain("unavailable"); expect(result.confidence).toBeNull(); expect(result.historyPoint).toBeNull();
  });
  it.each([null, "1", NaN, Infinity, -.01, 1.01])("rejects invalid probability %s without a chart", value => {
    const probs = { ...vector(LEGACY8, "happiness"), happiness: value };
    const result = present(wire({}, { probs }));
    expect(result.kind).toBe("invalid"); expect(result.scores).toEqual([]);
  });
  it("rejects wrong mass, mixed/partial class spaces, bad structure and malformed display quantities", () => {
    for (const response of [null, {}, wire({}, { probs: { happy: 1 } }), wire({}, { probs: { ...vector(LEGACY8, "happiness"), happy: 0 } }),
      wire({}, { probs: { ...vector(LEGACY8, "happiness"), happiness: .8 } }), wire({}, { uncertainty: null }),
      wire({ quality: NaN }), wire({ aus: [{ code: "AU01", intensity: null }] }), wire({ insufficient_evidence: { insufficient: false, reasons: [123] } })]) {
      expect(present(response).kind).toBe("invalid");
    }
  });
  it("names FER7 separately from the current legacy eight; does not invent a backend change", () => {
    const legacy = present(wire());
    const fer = present(wire({}, { probs: vector(FER7, "happy") }));
    expect(legacy.classSpace).toBe("legacy-eight"); expect(fer.classSpace).toBe("FER7");
    expect(legacy.scores.find(x => x.label === "happiness")?.value).toBe(1);
    expect(fer.scores.find(x => x.label === "happy")?.value).toBe(1);
    expect(legacy.scores.find(x => x.label === "unknown")?.support).toBe("residual");
  });
  it("keeps zero scores when support is not reported; explicit unsupported classes are unavailable", () => {
    const ordinary = present(wire());
    expect(ordinary.scores.find(x => x.label === "anger")).toMatchObject({ value: 0, support: "not-reported" });
    expect(ordinary.supportText).toContain("not reported");
    const explicit = present(wire({}, { supported_classes: ["happy"] }));
    expect(explicit.scores.find(x => x.label === "anger")).toMatchObject({ value: null, support: "unsupported" });
    expect(explicit.scores.find(x => x.label === "happiness")).toMatchObject({ value: 1, support: "supported" });
    const topLevel = { ...wire(), supported_classes: ["happiness"] };
    expect(present(topLevel).scores.find(x => x.label === "anger")?.value).toBeNull();
    expect(present(wire({}, { supported_classes: ["anger"] })).kind).toBe("invalid");
  });
  it("does not mutate/renormalise even near-unit values or explicit support", () => {
    const response = wire({}, { probs: { ...vector(FER7, "happy"), happy: .999999 }, confidence: .999999, uncertainty: .000001, supported_classes: [...FER7] });
    const before = JSON.stringify(response);
    Object.freeze(response.face.expression.probs); Object.freeze(response.face.expression); Object.freeze(response.face); Object.freeze(response);
    const state = present(response);
    expect(state.kind).toBe("usable"); expect(state.raw).toBe(response);
    expect(state.scores.find(x => x.label === "happy")?.value).toBe(.999999);
    expect(JSON.stringify(response)).toBe(before);
  });
  it("unknown reasons stay visible and cannot hit inherited dictionary properties", () => {
    for (const reason of ["new-rule", "toString", "constructor"]) {
      const state = present(wire({ insufficient_evidence: { insufficient: false, reasons: [reason] } }));
      expect(state.kind).toBe("insufficient"); expect(state.reasons).toEqual([`Unknown evidence status: ${reason}`]);
      expect(state.historyPoint).toBeNull();
    }
  });
  it("only stable category text enters status; numbers and thresholds stay outside it", () => {
    const first = present(wire()), second = present(wire({}, { probs: vector(LEGACY8, "neutral"), confidence: .8, uncertainty: .2 }));
    expect(first.statusText).toBe(second.statusText);
    const a = present(wire({ abstained: true, abstain_threshold: .7 })), b = present(wire({ abstained: true, abstain_threshold: .9 }));
    expect(a.statusText).toBe(b.statusText); expect(a.message).not.toBe(b.message);
    expect(a.historyPoint).toBeNull();
  });
});


describe("H1 source-shaped synthetic response fixtures", () => {
  it("accepts the real ViT shape: unknown zero, null heuristics and no profile", () => {
    const response = { ...wire({}, { probs: { ...vector(LEGACY8, "happiness"), happiness: .8, neutral: .2 },
      confidence: .8, uncertainty: .2, source: "synthetic ViT source", experimental: true,
      revision: "synthetic revision", calibrated: false, valence: null, arousal: null,
      expressiveness: null, warning: "apparent-expression scores; possible training overlap" }), selected_model: "vit" };
    expect(response.face.expression).not.toHaveProperty("profile");
    const state = present(response); expect(state.kind).toBe("usable");
    expect(state.historyPoint).toEqual({ conf: .8, top: "happiness" }); expect(state.raw).toBe(response);
  });
  it("accepts calibrated mini shape and preserves raw scores/temperature without substitution", () => {
    const probs = { ...vector(LEGACY8, "happiness"), happiness: .8, unknown: .2 };
    const probs_raw = vector(LEGACY8, "happiness");
    const response = wire({}, { probs, probs_raw, confidence: .8, uncertainty: .2, profile: "corrected",
      calibrated: true, temperature: 1.045, valence: .8, arousal: 0, expressiveness: 0 });
    const state = present(response); expect(state.kind).toBe("usable");
    expect(state.confidence).toBe(.8); expect(state.raw).toBe(response);
    expect(response.face.expression).toMatchObject({ probs_raw, temperature: 1.045 });
  });
  it("rejects the FER7-backed predictor not-run sentinel (served as legacy-eight), including error source", () => {
    for (const source of ["hdf5 missing", "error: synthetic failure"]) {
      const state = present(wire({ insufficient_evidence: { insufficient: true, reasons: ["high-uncertainty"] } },
        { ...placeholder, source, valence: 0, arousal: 0, expressiveness: 0 }));
      expect(state.kind).toBe("unavailable"); expect(state.historyPoint).toBeNull(); expect(state.scores).toEqual([]);
    }
  });
  it.each([-1, 1])("pins both sides of 1e-5 mass tolerance (direction %s)", sign => {
    for (const [delta, expected] of [[.000009999, "usable"], [.000010001, "invalid"]] as const) {
      const probs = { ...vector(LEGACY8, "happiness"), happiness: .5, neutral: .5 + sign * delta };
      const response = wire({}, { probs }); const before = JSON.stringify(response);
      expect(present(response).kind).toBe(expected); expect(JSON.stringify(response)).toBe(before);
    }
  });
  it.each(["quality", "valence", "arousal", "expressiveness"])("validates %s finite bounds and endpoints", key => {
    const low = key === "valence" ? -1 : 0;
    const run = (value: unknown) => present(key === "quality" ? wire({ quality: value }) : wire({}, { [key]: value }));
    for (const value of [low, 1]) expect(run(value).kind).toBe("usable");
    for (const value of [low - .000001, 1.000001, NaN, Infinity, -Infinity, true, "0"])
      expect(run(value).kind).toBe("invalid");
    if (key !== "quality") expect(run(null).kind).toBe("usable");
    else expect(run(null).kind).toBe("invalid");
  });
  it.each(["selected_model", "ethics_notice", "source", "AU code", "reason"])("pins 8192-character display text bound for %s", key => {
    const make = (n: number) => {
      const value = "x".repeat(n);
      if (key === "source") return wire({}, { source: value });
      if (key === "AU code") return wire({ aus: [{ code: value, intensity: .5 }] });
      if (key === "reason") return wire({ insufficient_evidence: { insufficient: true, reasons: [value] } });
      return { ...wire(), [key]: value };
    };
    expect(present(make(8192)).kind).toBe(key === "reason" ? "insufficient" : "usable");
    expect(present(make(8193)).kind).toBe("invalid"); expect(present(make(0)).kind).toBe("invalid");
  });
});
