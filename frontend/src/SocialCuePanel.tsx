import { useEffect, useRef, useState } from "react";

import {
  CONFIDENCE_CAP, CONSENT_LABEL, CueController, ETHICS_NOTE, STATE_NAMES, SYNTHETIC_DEMO_LABEL, WORDING,
  cuesEnabled, presentCueBlock, shouldAnnounce, type Camera, type CueBlock, type CueStatus,
} from "./socialCueLogic";
import { CUE_FIXTURES, fixtureBlock } from "./socialCueFixtures";

/** Browser camera for the cue loop: own stream, own hidden canvas; all tracks stopped on close. */
export function makeCamera(video: { current: any }, canvas: { current: any }): Camera {
  let stream: any = null, epoch = 0;
  const releaseOwned = () => {
    stream?.getTracks().forEach((t: any) => t.stop());
    if (video.current && video.current.srcObject === stream) video.current.srcObject = null;
    stream = null;
  };
  return {
    async open() {
      const token = ++epoch;
      releaseOwned();
      const acquired = await navigator.mediaDevices.getUserMedia({ video: true });
      if (token !== epoch) { acquired.getTracks().forEach((t: any) => t.stop()); return; }
      if (!video.current) { acquired.getTracks().forEach((t: any) => t.stop()); throw new Error("video element unavailable"); }
      stream = acquired;
      video.current.srcObject = acquired;
    },
    async frame() {
      const v = video.current, c = canvas.current;
      if (!v || !c || !v.videoWidth) return null;
      c.width = 320; c.height = 240;
      c.getContext("2d").drawImage(v, 0, 0, 320, 240);
      return await new Promise<Blob | null>(resolve => c.toBlob(resolve, "image/jpeg", 0.8));
    },
    close() {
      ++epoch; // Invalidates every pending grant before releasing the owned stream.
      releaseOwned();
    },
  };
}

export default function SocialCuePanel({ api, enabled = cuesEnabled() }: { api: string; enabled?: boolean }) {
  return enabled ? <SocialCueInner api={api} /> : null;
}

export function SocialCueInner({ api }: { api: string }) {
  const video = useRef<any>(null);
  const canvas = useRef<any>(null);
  const controller = useRef<CueController | null>(null);
  const announced = useRef({ summary: "", at: -Infinity });
  const [consent, setConsent] = useState(false);
  const [status, setStatus] = useState<{ status: CueStatus; message: string }>({ status: "idle", message: "Stopped. The camera is off." });
  const [live, setLive] = useState<CueBlock | null>(null);
  const [demoId, setDemoId] = useState("");
  const [announcement, setAnnouncement] = useState("");

  const demoBlock = demoId ? fixtureBlock(demoId) : null;
  const view = presentCueBlock(demoId ? demoBlock : live);

  const announce = (summary: string) => {
    const now = Date.now();
    if (shouldAnnounce(announced.current.summary, summary, announced.current.at, now)) {
      announced.current = { summary, at: now };
      setAnnouncement(summary);
    }
  };
  // Clearing the display (stop, revoke, pause) must never be throttled or leave a stale live cue behind.
  const clearAnnouncement = () => { announced.current = { summary: "", at: -Infinity }; setAnnouncement(""); };
  if (controller.current === null) {
    controller.current = new CueController({
      api,
      fetchFn: (url, init) => fetch(url, init),
      camera: makeCamera(video, canvas),
      setTimer: (fn, ms) => window.setTimeout(fn, ms),
      clearTimer: id => window.clearTimeout(id),
      onBlock: block => { setLive(block); if (block === null) clearAnnouncement(); else announce(presentCueBlock(block).summary); },
      onStatus: (s, message) => { setStatus({ status: s, message }); if (s === "paused" || s === "error") setLive(null); if (s !== "running") clearAnnouncement(); },
      onConsent: value => setConsent(value),
    });
  }
  const ctl = controller.current;
  // Capture ownership survives recoverable transport errors.
  const running = ctl.running || status.status === "starting";
  const consentInput = useRef<HTMLInputElement>(null);
  const startButton = useRef<HTMLButtonElement>(null);
  const stopButton = useRef<HTMLButtonElement>(null);
  const errorDescription = status.status === "error" && !demoId ? " cue-error" : "";


  useEffect(() => {
    const onVisibility = () => ctl.setHidden(typeof document !== "undefined" && document.visibilityState === "hidden");
    if (typeof document !== "undefined") document.addEventListener("visibilitychange", onVisibility);
    onVisibility();
    return () => {
      if (typeof document !== "undefined") document.removeEventListener("visibilitychange", onVisibility);
      ctl.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section id="social-cues" className="cue-panel" aria-labelledby="cues-heading">
      <h2 id="cues-heading">Social cues (research prototype)</h2>
      <p id="cue-banner">{demoId ? SYNTHETIC_DEMO_LABEL : view.banner}</p>
      <p id="cue-ethics">{ETHICS_NOTE}</p>
      <p id="cue-capture-notice">
        Live: while running, camera frames are sent to the local backend at {api} about 8 times per second, one request at a time, only for this
        session. Nothing is sent before you consent. Stop or revoking consent ends capture, clears this display and asks the backend to drop the
        session; server logging and retention are not verified. Frames pause while this tab is hidden. The synthetic demo sends nothing.
      </p>
      <p id="cue-controls-help">
        {demoId ? "Live consent and Start are unavailable in the synthetic demo. Select off to use the camera."
          : running ? "A camera run is active or starting. Stop and clear ends it; the synthetic demo is unavailable until then."
          : consent ? "Ready to start. Stop and clear is unavailable because no camera run is active."
          : "Agree to the camera notice to enable Start. Stop and clear is unavailable because no camera run is active."}
      </p>
      <div className="cue-controls">
        <label htmlFor="cue-consent">
          <input ref={consentInput} id="cue-consent" type="checkbox" checked={consent} disabled={demoId !== ""} aria-describedby={"cue-capture-notice cue-controls-help" + errorDescription}
            onChange={e => { ctl.setConsent(e.target.checked); setConsent(e.target.checked); }} /> {CONSENT_LABEL}
        </label>{" "}
        <button ref={startButton} id="cue-start" type="button" aria-describedby={"cue-capture-notice cue-controls-help" + errorDescription} disabled={!consent || running || demoId !== ""}
          onClick={() => {
            const ownedFocus = typeof document !== "undefined" && document.activeElement === startButton.current;
            void ctl.start().then(started => {
              if (!started && ownedFocus && typeof document !== "undefined"
                && (document.activeElement === startButton.current || document.activeElement === document.body)) consentInput.current?.focus();
            });
          }}>Start social cues (camera)</button>{" "}
        <button ref={stopButton} id="cue-stop" type="button" aria-describedby={"cue-controls-help" + errorDescription} disabled={!running} onClick={() => {
          // Move before disabling Stop, while its focus ownership is still observable.
          if (typeof document !== "undefined" && document.activeElement === stopButton.current) consentInput.current?.focus();
          ctl.stop();
        }}>Stop and clear</button>{" "}
        <label htmlFor="cue-demo">Synthetic demo (no camera){" "}
          <select aria-describedby="cue-controls-help" id="cue-demo" value={demoId} disabled={running} onChange={e => { if (!ctl.running && status.status !== "starting") setDemoId(e.target.value); }}>
            <option value="">off</option>
            {CUE_FIXTURES.map(f => <option key={f.id} value={f.id}>{f.label}</option>)}
          </select>
        </label>
      </div>
      <p id="cue-status" role="status" aria-live="polite">{demoId ? SYNTHETIC_DEMO_LABEL : status.status === "running" && announcement ? announcement : status.message}</p>
      {status.status === "error" && !demoId && <p id="cue-error" className="error-message">{status.message}</p>}
      {status.status !== "error" && !demoId && <p id="cue-state-message">{status.message}</p>}
      <ul className="cue-rows" aria-label="Social cue hypotheses">
        {view.rows.map(row => (
          <li key={row.name} className={row.active ? "cue-row cue-active" : "cue-row"}>
            <h3>{row.label}</h3>
            <p>{row.text}</p>
            {row.confidence !== null && <p>
              <span className="cue-bar-track" aria-hidden="true"><span className="cue-bar" style={{ width: `${(row.confidence / CONFIDENCE_CAP) * 100}%` }} /></span>{" "}
              confidence {row.confidence.toFixed(2)} ({row.band}; scale 0 to {CONFIDENCE_CAP})
            </p>}
            {row.evidence.length > 0 && <details>
              <summary>Why (technical measurements)</summary>
              <p>These measurements are used by an unvalidated research rule (heuristic), HB-1, not probabilities of feelings. The names below are technical identifiers.
                p_neg_arousal: weighted expression-model score for selected negative-expression classes; dwell_s: seconds spent in the nervous-like or unsure-like pattern.</p>
              <ul>{row.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </details>}
          </li>
        ))}
      </ul>
      <p id="cue-body-status" role="status" aria-live="off">{demoId ? "Body evidence: not part of the synthetic demo" : live?.bodyText ?? "Body evidence: unavailable (no data yet)"}</p>
      <p><small>{WORDING} Window of about 3 s; confidence is capped at {CONFIDENCE_CAP}. {STATE_NAMES.length} hypotheses; none is a diagnosis.</small></p>
      <video ref={video} hidden aria-hidden="true" muted playsInline autoPlay />
      <canvas ref={canvas} hidden aria-hidden="true" />
    </section>
  );
}
