# Social cues: operator guide

## What you are looking at

This is a research prototype. Its three rows describe signs consistent with nervous-like, unsure-like or needs-help-like behaviour, not feelings or a diagnosis. The hand-written thresholds were never tuned on real data; no real accuracy figure exists, and none should be quoted. Never use it for employment, school, healthcare or policing decisions; monitoring staff, students or job candidates; anyone without informed consent; bystanders; or claims that it detects nervousness, deceit, intent or need for help. [Source: [reference section 1](SOCIAL_CUES.md#1-what-this-is-and-what-it-is-not), [reference section 8](SOCIAL_CUES.md#8-never-use-it-for)]

## Explore without a camera

On an available dashboard, `?socialcues=1` URL flag reveals the separate Social cues panel. Keep camera capture stopped and choose **Synthetic demo (no camera)**: calm, nervous-like, unsure-like, needs-help-like escalation, insufficient evidence, multi-person or dropout. These fabricated examples send no inference request. They show presentation, not detected behaviour, live timing or model performance. [Source: [reference section 3](SOCIAL_CUES.md#3-using-the-flagged-api-and-the-panel), [reference section 11](SOCIAL_CUES.md#11-d6-live-demo-checklist-operator-with-one-consenting-person)]

The demo body line says **Body evidence: not part of the synthetic demo**. It cannot demonstrate a positive body cue. The scripted route rehearsal uses manufactured inputs and fake models; it checks route plumbing, not real models, transport, frame rates or accuracy. [Source: [reference section 11](SOCIAL_CUES.md#11-d6-live-demo-checklist-operator-with-one-consenting-person), [reference section 10 rehearsal](SOCIAL_CUES.md#scripted-route-rehearsal-s75-plan-item-u1-synthetic-only)]

## Read the display

Each row is a hedged hypothesis. Confidence bars stop at 0.8; the heuristic is unvalidated. Through the synthetic 8 fps API ring, needs-help-like confidence cannot exceed about 0.55 (dwell margin cap 0.575 x coverage x quality, plus rounding), while a dense full-history reference can reach 0.8; this is the S56 ring definition, not a defect or a tuned threshold. **Why** lists contributing features. **No cue / insufficient evidence** is the expected default: the system makes no cue claim, rather than finding that someone is calm or fine. Displayed reasons can include collecting frames, low detection quality or too few usable frames. More than one face gives **More than one person in view: not analysed**. [Source: [reference section 1](SOCIAL_CUES.md#1-what-this-is-and-what-it-is-not), [reference section 3](SOCIAL_CUES.md#3-using-the-flagged-api-and-the-panel)]

The small line under the three rows explains body evidence. **no pose model installed - expected** means expected face-only operation. Other messages explain missing evidence, such as collecting frames, hands not visible or frame rate too low. Available evidence counts present body cues and names supported face-based cues, or **none**. With zero present cues, a nonzero count of cues marked **none** or **unavailable** is also shown. Body evidence can support an existing face-based cue; it cannot create one by itself. [Source: [reference section 11, body-status explanation](SOCIAL_CUES.md#11-d6-live-demo-checklist-operator-with-one-consenting-person)]

## Understand the live path

Live use follows the [D6 checklist](SOCIAL_CUES.md#11-d6-live-demo-checklist-operator-with-one-consenting-person) and its own conditions. The panel's live path requires the consent checkbox and one person in view. Stop or revoking consent ends capture and clears the display; pause and request errors clear old cue/body information. Real browser, camera and model behaviour remain unverified. [Source: [reference section 3](SOCIAL_CUES.md#3-using-the-flagged-api-and-the-panel), [reference section 11](SOCIAL_CUES.md#11-d6-live-demo-checklist-operator-with-one-consenting-person)]

## Know the privacy limits

The session path retains derived numeric records in memory, not image/video frames, and writes nothing to disk through this path. Idle expiry is 60 seconds but physical removal waits for later store access. The panel uses no browser storage. Stop requests session deletion only after a frame was sent; deletion is best effort, not proof of physical erasure. Server/proxy logs and machine retention are unaudited. The ordinary dashboard's Stop does not send that deletion request. [Source: [reference section 6](SOCIAL_CUES.md#6-privacy), [reference section 3 dashboard](SOCIAL_CUES.md#primary-dashboard-movement-session-s39), [reference section 11](SOCIAL_CUES.md#11-d6-live-demo-checklist-operator-with-one-consenting-person)]

Session ids are bearer secrets: possession identifies a session, not an authenticated caller. DELETE is unauthenticated; anyone who can reach the backend and has the id can delete that session, and callers can evict others through the global session cap. CORS is open to any origin. Treat this as a single-user local prototype only, not a shared service. [Source: [reference section 3](SOCIAL_CUES.md#3-using-the-flagged-api-and-the-panel), [reference section 6](SOCIAL_CUES.md#6-privacy)]

Data-rights and evaluation status: reference sections 4-5. [Source: [reference section 4](SOCIAL_CUES.md#4-data-and-rights-posture), [reference section 5](SOCIAL_CUES.md#5-evaluation-status)]
