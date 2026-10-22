<!--
Installed as docs/backend-capability-semantics.md (EL-005 P5-install, 2026-10-20T20:24:47Z) from the audited P5-v3 draft plus the audit-v4 fixes F3-F4. Cited main.py lines were re-read at install against HEAD eae0289884d27a95dbc3f4b04fe4256869b84985 (commit eae0289); main.py is byte-identical to its 63e5810 version (sha256 72799b2d...).
DESIGN ONLY. No endpoint, schema, response or default is changed by this text. Nothing was run to write it: no server, no get_predictor, no import of a model library, no weight stat, no /api/health or /api/model-info call.
-->
# Backend capability semantics (U6) — design only

**Status: DESIGN ONLY — NO RUNTIME IMPLEMENTATION, NO ENDPOINT, NO MODEL, NO ASSET AUTHORISATION.** This document defines *words*: what a statement such as "available", "supported" or "ready" may mean when a backend reports it, so that a future service and the UI do not claim more than was checked. It adds no code, changes no response, enables no model and probes nothing. Readiness of a model for integration is a separate, gated decision (`docs/integration-readiness.md`).

Labels: **[V]** verified in the cited source line at HEAD `eae0289` (`main.py` is unchanged since `63e5810`). `backend/app/face/expression.py` and `docs/API.md` are byte-identical to their `e2913b5`/`e09a40f` versions (hashes `a9569e39…` and `38a89788…`). `backend/app/main.py` is **not**: the social-cue commits edited it (lines 1, 131-132, 291 and 299-342: the import, the two extra `/api/infer` parameters, the `resp` assignment, the optional block and the new delete route), so every `main.py` line number below was re-read at HEAD; **[I]** inference with reason; **[U]** unknown.

## 1. Why this document exists

[V] `GET /api/models` returns a hard-coded literal: `mini-xception` `available: true`, `vit` `available: true` (experimental), `hsemotion` `available: false` (`backend/app/main.py:96-121`). Those flags are written in the source; **no code checks that a weight file exists, that a library imports, or that a model loads** before returning them. The same file shows the gap: `POST /api/infer` with `model=vit` can still fail at run time with HTTP 503 "requested ViT unavailable" (`main.py:194-202` (the 503 raise is at 200-202)), although `/api/models` said `available: true`. [V] For mini-Xception, a missing weight file does **not** raise at all: the predictor returns an HTTP 200 sentinel (all mass on `unknown`, confidence 0.15, `source` such as "hdf5 missing") (`backend/app/face/expression.py:137-145`; `EL-004-cycle\C0\ui-contract.md` §1.6).

So the word "available" in the current API means only **"this route name is implemented in the source and not deliberately blocked"**. It must not be read as "weights are present", "the model loads", "scores are meaningful" or "healthy".

## 2. Dimensions that must stay separate

A capability report should never be a single boolean. Each dimension below is independent; a value on one says nothing about the others. Each has a **status vocabulary** (never `true`/`false` alone) and an **evidence note**.

| # | Dimension | Question it answers | Allowed values | What produces the value |
|---|---|---|---|---|
| D1 | **route implementation** | does the route's source exist and is it not blocked by design? | `IMPLEMENTED`, `BLOCKED_BY_DESIGN`, `NOT_IMPLEMENTED` | static fact about the code; today `mini-xception` and `vit` are implemented, `hsemotion` returns 501 |
| D2 | **asset presence** | is a file at the expected local path? | `NOT_CHECKED`, `PRESENT`, `ABSENT` | a filesystem check — **never performed implicitly**; the default is `NOT_CHECKED` |
| D3 | **verified pin** | does the file's hash equal a trusted, authority-approved pin? | `NOT_CHECKED`, `MATCH`, `MISMATCH`, `NO_AUTHORITY` | needs an approved pin and the authority to read the asset; today there is none for real Combined/HSE assets |
| D4 | **loadability** | can the runtime construct the model from that asset? | `NOT_CHECKED`, `LOADS`, `FAILS(reason)` | requires actually loading — a heavy, gated step |
| D5 | **execution availability** | is the model currently able to produce a score for a request? | `UNKNOWN`, `AVAILABLE`, `UNAVAILABLE(reason)` | the result of D1–D4 plus runtime state; derived, never asserted from D1 alone |
| D6 | **profile applicability** | which preprocessing profile(s) apply to this model? | per model, e.g. `corrected`, `legacy`, `n/a` | static (for ViT the `profile` query is ignored: `main.py:148,190`) |
| D7 | **supported classes** | which class names can this model output a score for? | an explicit list, or `NOT_REPORTED` | static or declared by the adapter; today **not reported** by the API |
| D8 | **calibration** | are scores raw or transformed, by what, fitted on what? | `RAW`, `TRANSFORMED(T, scope)`, `NOT_REPORTED` | today only `calibrated` and `temperature` in some responses |
| D9 | **provenance** | where did the weights and code come from? | `source`, `revision`, licence reference, `hash` **only when verified (D3)**, else `null` | metadata the response may carry; unverified metadata is labelled so |
| D10 | **blocked reason** | if not usable, why — in a stable, human-readable form? | enumerated reason + text (e.g. `NOT_INTEGRATED`, `ASSET_ABSENT`, `RIGHTS_GATE`, `AUTHORITY_REQUIRED`) | static or from the failing check |

[I] D2–D4 are the dimensions the current API silently collapses into the static `available` flag. D10 is today carried only as free-text in 501/503 `detail` strings.

## 3. What the current API actually conveys, per dimension

| Dimension | Today | Source |
|---|---|---|
| D1 | implicit: 501 for `hsemotion`; static `available` flags | `main.py:134-137,103-119` |
| D2–D4 | **not reported**; mini-Xception weights are loaded when the predictor is first constructed (`get_predictor`, `main.py:22-32`, which runs `ExpressionPredictor.__init__`, `expression.py:34-53`); a missing or unloadable file is recorded in the predictor's status/`source` and surfaces as a 200 sentinel, not an error; ViT load failure is 503 | `expression.py:34-53,137-145`; `main.py:22-32,194-202` |
| D5 | not reported; observable only by calling `/api/infer` | — |
| D6 | `profile` echoed for mini-Xception; absent for ViT; calibrated only for `mini-xception` + `corrected` (else 422) | `main.py:138-148,190,207-219` |
| D7 | **not reported.** The response always carries the legacy eight class names; `unknown` is an application residual (mini) or constant 0 (ViT) | `expression.py:5-14,91-95`; `main.py:203-206,218` |
| D8 | `calibrated` boolean, `temperature` and `probs_raw` only when calibrated | `expression.py:117-125` |
| D9 | `source` always; `revision` and `warning` for ViT; no hashes | `main.py:207-219` |
| D10 | free-text `detail` in 422/501/503; `warning`/`note` strings | `main.py` |

[V] Side effects to remember: `GET /api/health` **imports** `mediapipe`, `torch` and `onnxruntime` to report booleans (`main.py:44-59`), and `GET /api/model-info` **constructs** the corrected predictor, which loads weights (`main.py:62-93`). Neither is a free read, so a "readiness refresh" or documentation pass must not call them.

[V] Outside the model-capability dimensions defined here, `POST /api/infer` also accepts a `socialcue` query parameter (only the exact value `1` enables it) and the header `X-Socialcue-Session`, and then adds an optional `socialcue` block (`main.py:131-132,301-327`); `DELETE /api/socialcue/session` exists (`main.py:330-342`). The block is an opt-in research prototype: it states `heuristic: "HB-1 hypothesis, unvalidated"` and `proxy_eval_ref: null` (`backend/app/socialcue_session.py:52,71-72`; the last-resort fallback literal is at `main.py:323-324`; `docs/SOCIAL_CUES.md` §3). It does not change the default response and is not a model capability in the sense of D1-D10.

## 4. Rules for any future capability report

1. **No boolean "ready".** A report states each dimension's value with its vocabulary; "ready" is not a field. A consumer that wants one answer must combine dimensions itself and keep unknowns as unknowns.
2. **Default to `NOT_CHECKED`/`UNKNOWN`, not `false`.** A check that was not performed is not a failure and not a pass.
3. **Hashes are reported only when D3 was actually performed against an approved pin**, with the pin's source. Otherwise `hash: null`. Never invent or echo an unverified hash.
4. **Scope every claim to a model and source**: a statement about `mini-xception` says nothing about `vit`.
5. **Routes are not rights.** `IMPLEMENTED` does not mean licensed, consented or scientifically validated. Rights gates (data, weights, intended use) are **procedural and human-held** and stay procedural even if a path or a hash exists on disk. Having a file, being able to hash it, or being able to load it does not satisfy a rights gate.
6. **No silent fallback.** A request for a model that cannot run returns a typed failure for *that* model; it never substitutes `mini-xception` for a blocked challenger (also the contract in `docs/API.md:20-21`).
7. **The legacy eight-class API and the FER7 benchmark contract are different spaces** and must not be mixed in one report. The served API emits the eight names {happiness, sadness, anger, surprise, fear, disgust, neutral, unknown}; FER7 {angry, disgust, fear, happy, sad, surprise, neutral} exists in adapters and benchmarks. Any report that lists classes names the space it is in.
8. **Generating this documentation never probes the service.** No `get_predictor`, no import of a model library, no `stat` of a weight path, no call to `/api/health` or `/api/model-info`.
9. **Error classes keep their meaning:** `422` = the request is invalid (unknown profile, calibration combination, bad abstain value, undecodable image); `501` = the route is deliberately not integrated; `503` = integrated but currently unavailable (missing dependency or asset). Detail text is human-readable; diagnostics are sanitised and kept apart from the user message.
10. **Status is point-in-time.** A report carries an `as_of` time and the source revision that produced it; an old report is not current evidence.

## 5. Hypothetical additive schema (illustration only — not a patch, not an endpoint)

These examples show the vocabulary; they are **not** the wire format of anything that exists. Values are placeholders. The existing `expression.probs`, `ethics_notice` and legacy fields would stay as they are; any new object would be optional and ignored by old clients.

```json
{
  "capability_report_example": true,
  "as_of": "NOT_REPORTED",
  "source_revision": "NOT_REPORTED",
  "models": {
    "mini-xception": {
      "route": "IMPLEMENTED",
      "asset_presence": "NOT_CHECKED",
      "verified_pin": "NOT_CHECKED",
      "loadability": "NOT_CHECKED",
      "execution_availability": "UNKNOWN",
      "profiles": ["corrected", "legacy"],
      "supported_classes": { "space": "legacy-eight", "names": "NOT_REPORTED" },
      "calibration": { "default": "RAW", "available": "corrected_only", "temperature_scope": "NOT_REPORTED" },
      "provenance": { "source": "NOT_REPORTED", "revision": null, "hash": null, "licence_reference": "NOT_REPORTED" },
      "blocked_reason": null
    },
    "vit": {
      "route": "IMPLEMENTED",
      "asset_presence": "NOT_CHECKED",
      "verified_pin": "NOT_CHECKED",
      "loadability": "NOT_CHECKED",
      "execution_availability": "UNKNOWN",
      "profiles": ["n/a"],
      "supported_classes": { "space": "legacy-eight", "names": "NOT_REPORTED", "unknown_class": "never_emitted" },
      "calibration": { "default": "RAW", "available": "none" },
      "provenance": { "source": "NOT_REPORTED", "revision": "NOT_REPORTED", "hash": null, "licence_reference": "NOT_REPORTED" },
      "experimental": true,
      "blocked_reason": null
    },
    "hsemotion": {
      "route": "BLOCKED_BY_DESIGN",
      "execution_availability": "UNAVAILABLE",
      "blocked_reason": { "code": "NOT_INTEGRATED", "text": "separate implementation; no merge performed", "gate": "RIGHTS_AND_ASSET_AUTHORITY_REQUIRED" }
    }
  }
}
```

Example of a *response-level* statement the UI could render without claiming health (all fields optional, unknown stays unknown):

```json
{ "execution_status": "no_face", "supported_classes": "NOT_REPORTED", "score_kind": "RAW", "calibration": { "applied": false },
  "provenance": { "source": "<as returned>", "verified": false } }
```

Cross-reference: this document is installed as `docs/backend-capability-semantics.md` together with `docs/STATUS.md` and the `docs/integration-readiness.md` refresh; those documents cite it under that name.

Fixture-only note: the first block deliberately contains **no** boolean named `available`, `ready` or `healthy` and no non-null hash; a consumer that finds `true` in such a field would be inventing it.

## 6. Mapping to the UI and to errors

| UI/consumer need | Dimension | Rule |
|---|---|---|
| show "Unsupported class is unavailable, not 0" | D7 | until `supported_classes` is reported, the UI says "support not reported" (the U1 design in `EL-004-cycle\C0\ui-contract.md` §6/OQ1) |
| disable the calibrated control for ViT/legacy | D6/D8 | derive from D6 (profile applicability), not from a probe |
| "model unavailable" message | D5/D10 | map 501 → "not integrated", 503 → "temporarily unavailable"; show the stable reason code, not a stack |
| model dropdown | D1 | list only routes with `route: IMPLEMENTED`; do not display `available` as health |
| provenance panel | D9 | show only what the response carries; label it "source-reported, unverified" unless D3 was performed |

## 7. What this document does not decide

It does not decide whether to build a capability endpoint, which fields to ship, how checks would be authorised, or any asset pin. It does not promote or enable any model, does not touch the default, does not change `main.py`, `adapters.py` or the UI, and does not verify any model. Real availability checks (D2–D5) require explicit asset and runtime authority and a separate order; until then every value above that depends on them is `NOT_CHECKED`/`UNKNOWN`.

## 8. Trace table

| Claim | Source (HEAD eae0289; `main.py` unchanged since 63e5810) | Evidence level |
|---|---|---|
| static `available` literals | `backend/app/main.py:96-121` | SRC |
| ViT can fail with 503 after `available: true` | `main.py:194-202` (the 503 raise is at 200-202) | SRC |
| `hsemotion` → 501 | `main.py:134-137` | SRC |
| calibration only for mini + corrected | `main.py:138-142`; `expression.py:77-86` | SRC |
| missing weights → 200 sentinel | `expression.py:137-145` | SRC |
| no `supported_classes` in the response | `EL-004-cycle\C0\ui-contract.md` §1.6 (grep of the sources) | SRC |
| `/api/health` imports libraries; `/api/model-info` loads weights | `main.py:44-59,62-93` | SRC |
| no-fallback contract | `docs/API.md:20-21` | SRC |
| source hash of `main.py` at install (HEAD `eae0289`) | `72799b2da6a03fa76bf5cf0828226e2c2ebb881315b1b0a74a3413700b86f0e1` | file hash |
