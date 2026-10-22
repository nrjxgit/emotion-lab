import { useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

import { presentResult, inspectionText, RequestFailure, httpFailure, cameraFailure, errorDiagnostic, sanitiseDiagnostic, responseProvenance, type InferResponse, type Presentation } from "./resultPresentation";

import { DEMO_FIXTURES, DEMO_LABEL, demoPresentation } from "./demoFixtures";
import SocialCuePanel from "./SocialCuePanel";
import { cuesEnabled, newSessionId, SESSION_HEADER } from "./socialCueLogic";
import "./App.css";

const API = "http://127.0.0.1:8000";

export default function App() {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const movementSession = useRef<string | null>(null);
  const [on, setOn] = useState(false);
  const [model, setModel] = useState("mini-xception");
  const [profile, setProfile] = useState("corrected");
  const [calibrated, setCalibrated] = useState(false);
  const [abstain, setAbstain] = useState("off");
  const [data, setData] = useState<(InferResponse & { requestProfile: string }) | null>(null);
  const [err, setErr] = useState("");
  const [view, setView] = useState<Presentation>(presentResult({ phase: "stopped" }));
  const [hist, setHist] = useState<{ t: number; conf: number; top: string }[]>([]);
  const [lat, setLat] = useState(0);
  const [fps, setFps] = useState(0);
  const cancelRun = useRef<(() => void) | null>(null);
  const [mode, setMode] = useState<"live" | "demo">("live");
  const [demoScenario, setDemoScenario] = useState<string | null>(null);

  const clearCurrent = () => {
    setData(null); setHist([]); setLat(0); setFps(0); setDemoScenario(null);
  };
  // Invalidate in the control handler, before React runs effect cleanup.
  const transition = (nextOn: boolean) => {
    cancelRun.current?.();
    clearCurrent(); setErr("");
    setView(presentResult({ phase: nextOn ? (mode === "demo" ? "waiting" : "requesting-camera") : "stopped" }));
  };

  useEffect(() => {
    if (!on) return;
    // The demo exits this branch before constructing a camera/transport run.
    if (mode === "demo") {
      let active = true, index = 0;
      let demoTimer: number | undefined;
      const disposeDemo = () => {
        if (!active) return;
        active = false;
        if (demoTimer !== undefined) window.clearTimeout(demoTimer);
        if (cancelRun.current === disposeDemo) cancelRun.current = null;
      };
      cancelRun.current = disposeDemo;
      const step = () => {
        if (!active) return;
        const fixture = DEMO_FIXTURES[index];
        const next = demoPresentation(fixture);
        setView(next); setDemoScenario(fixture.label);
        setData(next.response ? { ...next.response, requestProfile: "synthetic fixture" } : null);
        setErr(next.kind === "invalid" || next.kind === "unavailable" ? "Synthetic demo scenario: " + next.message : "");
        setLat(0); setFps(0);
        const point = next.historyPoint;
        if (point) setHist(h => [...h.slice(-59), { t: Date.now(), ...point }]);
        else if (next.kind === "invalid" || next.kind === "unavailable") setHist([]);
        index = (index + 1) % DEMO_FIXTURES.length;
        demoTimer = window.setTimeout(step, 800);
      };
      step();
      return disposeDemo;
    }
    let live = true, last = Date.now();
    let timer: number | undefined;
    let stream: MediaStream | null = null;
    const controller = new AbortController();
    const element = video.current;
    const dispose = () => {
      if (!live) return;
      live = false;
      if (timer !== undefined) window.clearTimeout(timer);
      controller.abort();
      stream?.getTracks().forEach(track => track.stop());
      if (stream && element?.srcObject === stream) element.srcObject = null;
      if (cancelRun.current === dispose) cancelRun.current = null;
    };
    cancelRun.current = dispose;
    (async () => {
      let sessionId: string;
      try {
        // One cryptographic id for this mounted page; memory only, never persisted.
        sessionId = movementSession.current ?? newSessionId();
        movementSession.current = sessionId;
        const acquired = await navigator.mediaDevices.getUserMedia({ video: true });
        if (!live) { acquired.getTracks().forEach(track => track.stop()); return; }
        stream = acquired;
        if (!element) throw new Error("video element unavailable");
        element.srcObject = stream;
        setView(presentResult({ phase: "waiting" }));
      } catch (e: any) {
        if (!live) return;
        const failure = cameraFailure(e);
        clearCurrent(); setErr(failure.message);
        setView({ ...presentResult({ phase: "waiting", error: failure.message, unavailableStatus: "Camera unavailable" }), diagnostic: failure.diagnostic }); dispose(); setOn(false); return;
      }
      const loop = async () => {
        if (!live) return;
        let accepted = false;
        try {
          const v = element!, c = canvas.current!;
          if (v.videoWidth) {
            let blob: Blob | null;
            try {
              c.width = 320; c.height = 240;
              c.getContext("2d")!.drawImage(v, 0, 0, 320, 240);
              blob = await new Promise<Blob | null>(resolve => c.toBlob(resolve, "image/jpeg", 0.8));
            } catch (e: unknown) {
              throw new RequestFailure("Inference unavailable: frame capture unavailable. Try Stop, then Start again.", errorDiagnostic(e));
            }
            if (!live) return;
            if (!blob) throw new RequestFailure("Inference unavailable: frame capture unavailable. Try Stop, then Start again.", "The browser produced no captured frame.");
            const fd = new FormData(); fd.append("image", blob, "f.jpg");
            const t0 = performance.now();
            let qs = `?model=${model}&profile=${profile}`;
            if (model === "mini-xception" && profile === "corrected" && calibrated) qs += "&calibrated=true";
            if (abstain !== "off") qs += `&abstain=${abstain}`;
            const res = await fetch(`${API}/api/infer${qs}`, { method: "POST", body: fd, headers: { [SESSION_HEADER]: sessionId }, signal: controller.signal });
            if (!live) return;
            let j: unknown;
            try { j = await res.json(); }
            catch { if (!live) return; throw httpFailure(res.status, undefined, true); }
            if (!live) return;
            if (!res.ok) throw httpFailure(res.status, j);
            const next = presentResult({ phase: "waiting", response: j });
            setView(next);
            setData(next.response ? { ...next.response, requestProfile: model === "vit" ? "n/a" : profile } : null);
            setErr(next.kind === "invalid" || next.kind === "unavailable" ? "Inference unavailable: " + next.message : "");
            setLat(Math.round(performance.now() - t0));
            const point = next.historyPoint;
            if (point) setHist(h => [...h.slice(-59), { t: Date.now(), ...point }]);
            else if (next.kind === "invalid" || next.kind === "unavailable") setHist([]);
            accepted = next.response !== null;
          }
        } catch (e: any) {
          if (!live) return;
          const failure = e instanceof RequestFailure ? e : new RequestFailure("Inference unavailable. Check the local backend and try again.", errorDiagnostic(e));
          clearCurrent(); setErr(failure.message);
          setView({ ...presentResult({ phase: "waiting", error: failure.message }), diagnostic: failure.diagnostic });
        }
        if (!live) return;
        const now = Date.now();
        if (accepted) setFps(Math.round(1000 / Math.max(1, now - last)));
        last = now;
        timer = window.setTimeout(loop, 800);
      };
      void loop();
    })();
    return dispose;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [on, model, profile, calibrated, abstain, mode]);

  const ex = data?.face.expression;
  const provenance = responseProvenance(data, data?.requestProfile);
  const quantity = (value: number | null | undefined) => value == null ? "unavailable" : value.toFixed(2);
  return (<><a className="skip-link" href="#main-content">Skip to main content</a>
    <main id="main-content" className="dashboard" aria-labelledby="dashboard-title" tabIndex={-1}>
    <header>
    <h1 id="dashboard-title">Emotion Research Dashboard <small>v0.6</small></h1>
    <p><i>Apparent-expression hypotheses only — never proof of internal feeling.</i></p>
    </header>
    <section aria-labelledby="controls-heading">
    <h2 id="controls-heading">Capture and display controls</h2>
    <p id="capture-notice">Live webcam: while active, captured frames are sent to the configured local backend at {API}. Stop ends capture and cancels pending browser requests; it does not undo work already received by the server. Server logging and retention are not verified. Demo sends no frames.</p>
    <p id="movement-retention">Starting live camera opts in to numeric body movement history: up to 30 frames per random page session in backend memory, expiring after 60 seconds idle and purged on later access. The id is never persisted. Stop ends capture; this dashboard does not send DELETE, so numeric history can remain until expiry/access or backend exit. Demo retains no backend history.</p>
    <p id="camera-help">Camera help: if access is denied, review this site's camera permission in your browser and check that the camera is connected and available. Start requests access; no settings are changed here.</p>
    <div className="controls">
      <button id="run-control" aria-describedby="capture-notice movement-retention camera-help" onClick={() => { transition(!on); setOn(!on); }}>{on ? "Stop" : "Start"}{mode === "demo" ? " synthetic demo" : " webcam"}</button>{" "}
      <label htmlFor="model-control">Analysis model <select id="model-control" disabled={mode === "demo"} value={model} onChange={e => { transition(on); setModel(e.target.value); }}>
        <option value="mini-xception">mini-xception (default)</option>
        <option value="vit">vit (experimental)</option></select></label>{" "}
      <label htmlFor="profile-control">Input processing profile {model === "vit" && <span>n/a for ViT (stored choice retained)</span>}<select id="profile-control" hidden={model === "vit"} disabled={mode === "demo" || model === "vit"} value={profile} onChange={e => { transition(on); setProfile(e.target.value); }}>
        <option value="corrected">corrected (default)</option>
        <option value="legacy">legacy (repro)</option></select></label>{" "}
      <label htmlFor="calibration-control"><input id="calibration-control" type="checkbox" disabled={mode === "demo" || model !== "mini-xception" || profile !== "corrected"} checked={calibrated} onChange={e => { transition(on); setCalibrated(e.target.checked); }} /> Use calibrated model scores (T=1.045; corrected profile only)</label>
      {model === "vit" ? <span>Calibration unavailable for ViT; the stored choice is retained and is not sent.</span> : profile !== "corrected" && <span>Calibration unavailable for legacy; only mini-xception corrected supports this option. Stored choice retained.</span>}{" "}
      <label htmlFor="abstain-control">Minimum top model score (abstain below threshold) <select id="abstain-control" disabled={mode === "demo"} value={abstain} onChange={e => { transition(on); setAbstain(e.target.value); }}>
        <option value="off">off</option><option value="0.5">0.5</option>
        <option value="0.7">0.7</option><option value="0.9">0.9</option><option value="0.95">0.95</option></select></label>{" "}
      <label htmlFor="mode-control">mode <select id="mode-control" aria-label="Input mode" value={mode} onChange={e => {
        transition(false); setOn(false); setMode(e.target.value as "live" | "demo");
      }}><option value="live">Live webcam (default)</option><option value="demo">Demo - synthetic, no camera</option></select></label>{" "}
      <span>{mode === "demo" ? "Demo timing: not measured" : <>latency {lat}ms · {fps}fps</>}</span>
    </div>
    </section>
    {mode === "demo" && <section aria-label="Synthetic demo notice">
      <p><strong>{DEMO_LABEL}</strong></p>
      <p>No camera, backend request or storage is used. Live model controls are inactive; fixtures do not use their settings.</p>
      <p>{demoScenario ? `Manufactured scenario: ${demoScenario}` : "Demo stopped. Select Start synthetic demo to step through the fixtures."}</p>
    </section>}
    <section aria-labelledby="evidence-heading">
    <h2 id="evidence-heading">Current evidence status</h2>
    <p role="status" aria-live="polite">{view.statusText}</p>
    {err && <p role="alert" className="error-message">{err}</p>}
    <p>{view.message}</p>
    {provenance.visibleWarning && <p aria-label="Visible source warning">{provenance.visibleWarning}</p>}
    {view.diagnostic && <details aria-label="Sanitised error diagnostic"><summary>Error diagnostic (sanitised; not a stack trace)</summary><p>{view.diagnostic}</p></details>}
    {view.reasons.length > 0 && <ul aria-label="Evidence status reasons">
      {view.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul>}
    </section>
    <section aria-labelledby="inspection-heading">
    <h2 id="inspection-heading">Response inspection</h2>
    <div className="response-layout">
      <section className="camera-panel" hidden={mode === "demo"} aria-labelledby="camera-heading">
        <h3 id="camera-heading">Live camera preview</h3>
        <video className="camera-preview" aria-label="Live camera preview" aria-describedby="capture-notice" ref={video} autoPlay playsInline /><canvas ref={canvas} style={{ display: "none" }} />
      </section>
      <section className="result-panel" aria-label="Expression response details">
        <h3>Expression ({data ? `${data.selected_model}/${data.requestProfile}` : "no current result"})</h3>
        <p>{view.supportText}</p>
        <details>
          <summary>Inspection details — raw response values, not proof of feeling</summary>
          <p>Class space: {view.classSpace === "legacy-eight" ? "legacy eight; unknown is an application residual, not a model class" : view.classSpace || "unavailable"}</p>
          {view.scores.map(score => <div key={score.label}>
            {score.label}{score.support === "residual" ? " (application residual)" : ""}: {score.value === null ? "unavailable — unsupported class" : <>
              <span className="score-bar" aria-hidden="true" style={{ width: score.value * 180 }} /> {score.value.toFixed(2)}</>}
            {score.support === "not-reported" ? " (support not reported)" : ""}
          </div>)}
          <p>Model score {quantity(view.confidence)} · source-reported uncertainty {quantity(view.uncertainty)} · source {sanitiseDiagnostic(ex?.source) || "unavailable"}</p>
          <p>Heuristic inspection quantities: valence {quantity(ex?.valence)} · arousal {quantity(ex?.arousal)} · expressiveness {quantity(ex?.expressiveness)}</p>
          {data?.face.box && <p>face box {JSON.stringify(data.face.box)} · face-detection confidence {quantity(data.face.quality)}</p>}
          <h4>Action units (observable, not emotions)</h4>
          <p>{data?.face.aus.map(a => `${a.code}:${a.intensity.toFixed(2)}`).join(" ") || "unavailable"}</p>
          <h4>Raw response — no-face values are placeholders, not inference</h4>
          <pre>{inspectionText(view.raw)}</pre>
        </details>
        <details aria-label="Response provenance and uncertainty">
          <summary>Response provenance and uncertainty (source-reported, unverified)</summary>
          <dl>
            <dt>Response model</dt><dd>{provenance.model}</dd>
            <dt>Captured request profile</dt><dd>{provenance.capturedProfile}</dd>
            <dt>Response profile</dt><dd>{provenance.responseProfile}</dd>
            <dt>Source</dt><dd>{provenance.source}</dd>
            <dt>Raw/calibrated flag</dt><dd>{provenance.calibration}</dd>
            <dt>Temperature</dt><dd>{provenance.temperature}</dd>
            <dt>Revision</dt><dd>{provenance.revision}</dd>
            <dt>Warning</dt><dd>{provenance.warning}</dd>
            <dt>Model score</dt><dd>{quantity(view.confidence)}</dd>
            <dt>Source-reported uncertainty</dt><dd>{quantity(view.uncertainty)}</dd>
          </dl>
          <p>{provenance.uncertaintyDefinition}</p>
          <p>These quantities are not calibrated probabilities of feelings or of being correct. Scores from different models do not share comparable calibration.</p>
          <p>Model, source and revision are response metadata; runtime identity and asset provenance are not verified.</p>
        </details>
      </section>
    </div>
    </section>
    <section aria-labelledby="history-heading">
    <h2 id="history-heading">{mode === "demo" ? "Synthetic fixture score history (fabricated)" : "Usable model-score history (not an emotion reading)"}</h2>
    <p id="history-description">Previous usable frames in this run only, newest last (up to 60 points). Non-usable evidence frames omit a point; Stop and settings/mode changes clear the history. The table contains the same model scores as the chart.</p>
    <div className="history-chart" aria-hidden="true">
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={hist}><XAxis dataKey="t" hide /><YAxis domain={[0, 1]} /><Tooltip /><Line type="monotone" dataKey="conf" dot={false} /></LineChart>
      </ResponsiveContainer>
    </div>
    <table className="history-table" aria-describedby="history-description">
      <caption>{mode === "demo" ? "SYNTHETIC DEMO - fabricated score points" : "Usable model-score points - text alternative to chart"}</caption>
      <thead><tr><th scope="col">Sample</th><th scope="col">Model score</th><th scope="col">Top class (score label)</th></tr></thead>
      <tbody>{hist.length ? hist.map((point, index) => <tr key={`${point.t}-${index}`}>
        <th scope="row">{index + 1}</th><td>{point.conf.toFixed(2)}</td><td>{point.top}</td>
      </tr>) : <tr><td colSpan={3}>No usable score points for this run.</td></tr>}</tbody>
    </table>
    </section>
    {cuesEnabled() && <SocialCuePanel api={API} enabled />}
    <footer><p><small>{data?.ethics_notice}</small></p></footer>
  </main></>);
}
