// Static manufactured fixtures only. Never camera/media/network/model performance.
import { describe, expect, it, vi } from "vitest";
import { DEMO_FIXTURES, DEMO_LABEL, demoPresentation } from "./demoFixtures";

describe("synthetic demo fixtures", () => {
  it("has exactly the seven declared scenarios and the persistent disclosure", () => {
    expect(DEMO_LABEL).toBe("SYNTHETIC DEMO - fabricated scores, not accuracy or a camera reading");
    expect(DEMO_FIXTURES.map(f => f.id)).toEqual(["usable", "no-face", "low-quality", "abstained", "unavailable", "unsupported", "malformed"]);
    expect(Object.isFrozen(DEMO_FIXTURES)).toBe(true);
  });
  it.each(DEMO_FIXTURES.map(f => [f.id, f] as const))("%s flows through U1 without changing source data", (_name, fixture) => {
    const before = JSON.stringify(fixture);
    expect(demoPresentation(fixture).kind).toBe(fixture.expectedKind);
    expect(demoPresentation(fixture)).toEqual(demoPresentation(fixture));
    expect(JSON.stringify(fixture)).toBe(before);
    expect(Object.isFrozen(fixture)).toBe(true);
    if (fixture.response && typeof fixture.response === "object") expect(Object.isFrozen(fixture.response)).toBe(true);
  });
  it("keeps placeholders/failures out of history and explicitly reports unsupported classes", () => {
    const find = (id: string) => demoPresentation(DEMO_FIXTURES.find(f => f.id === id)!);
    expect(find("no-face").historyPoint).toBeNull(); expect(find("no-face").confidence).toBeNull();
    for (const id of ["low-quality", "abstained", "unavailable", "malformed"]) expect(find(id).historyPoint).toBeNull();
    expect(find("unsupported").scores.find(score => score.label === "anger")).toMatchObject({ value: null, support: "unsupported" });
    expect(find("usable").response?.selected_model).toBe("SYNTHETIC-demo");
    expect(find("unavailable").message).toContain("no backend request");
  });
});
