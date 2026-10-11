# Frontend A — screen, layout, and user instructions

## Mission

Frontend A owns the visible Gradio experience for LocalRecall: the screen structure, visual hierarchy, labels, instructions, empty states, and presentation flow.

The interface should make the product understandable to a first-time presenter. A user should know what to upload, when the laptop camera is active, what LocalRecall observed, where the answer came from, and how privacy is being handled.

Follow [`LocalRecall_Codex_Master_Prompt.md`](./LocalRecall_Codex_Master_Prompt.md) for the product requirements, including the model-result terminology and confidence behavior.

## Owns

Frontend A owns:

- The Gradio Blocks page structure and five-tab layout.
- Headings, labels, helper text, instructions, warnings, badges, and empty states.
- Visual hierarchy for the Watch, Memory, Ask, Setup, and System & Privacy screens.
- How BEFORE/AFTER evidence, event details, confidence, and source records are presented.
- Clear distinction between live browser-camera mode, manual image-pair mode, fixture mode, mock mode, and backend mode.
- Accessible, readable local styling using Gradio-supported components and local assets only.
- The presenter/demo flow and fallback instructions when camera access or streaming is unavailable.
- UI-level acceptance scenarios and screenshots or visual handoff notes.

Typical files:

```text
app/ui/app.py
app/ui/transforms.py       # display formatting and presentation helpers
TASK_UI.md
UI_HANDOFF.md
tests/test_ui_contract.py  # presentation-facing contract checks, coordinated with Frontend B
```

Frontend A may propose component IDs and output groupings, but callback wiring belongs to Frontend B. Shared facade methods and schemas belong to the backend owner.

## Does not own

Frontend A does not own:

- Backend model calls, prompt construction, response validation, or Qwen configuration.
- SQLite, object memory, event state, query logic, or evidence persistence.
- Browser-frame ingestion implementation, source leases, motion detection, or Watch Mode state.
- The facade protocol or backend schemas without coordination with Backend B.
- Long-running work inside layout construction, timers, or camera stream handlers.
- A custom React/Vue/Vite frontend, custom JavaScript canvas, FastAPI layer, WebSocket service, or public Gradio share link.
- Fake health, fake offline status, or fabricated events in backend mode.

The UI displays facts supplied by the facade. It must not infer a location, rewrite an event, or use `gr.State` as the source of truth.

## Product screen contract

Build a single Python Gradio Blocks application with these five tabs.

### 1. Watch — capture and analyze

Primary purpose: guide the user through a reliable event capture.

Include:

- A concise explanation of the product and a visible local/offline status indicator.
- Optional browser-owned live webcam input for Watch Mode.
- Clear instructions such as: “Start the laptop camera, allow browser permission, then begin Watch Mode.”
- Two separate non-streaming image inputs labeled `BEFORE` and `AFTER`.
- A prominent manual fallback instruction: upload or take two snapshots even if live streaming is unavailable.
- Motion/source health text showing connected, stale, disconnected, or unavailable states.
- A latest event summary with object, source zone, destination zone, confidence, and evidence.
- A clear distinction between “camera preview,” “frame received,” and “event analyzed.”

The manual BEFORE/AFTER path is first-class. It must work without an active stream, a server camera, or Watch Mode.

### 2. Memory — saved facts and evidence

Primary purpose: show what LocalRecall has actually remembered.

Include:

- Current object locations in a readable table.
- Recent event history with event IDs, timestamps, event type, object, and zones.
- Review items separated from accepted events.
- BEFORE, AFTER, and composite evidence where available.
- Approve and reject controls supplied by Frontend B.
- An empty state explaining that no accepted events have been saved yet.
- A note that locations come from accepted observed events, not continuous video or guesses.

Use the model’s vocabulary consistently: `move`, `appear`, `disappear`, `no_change`, `uncertain`, and the confidence bands defined by Backend A.

### 3. Ask — answer from saved memory

Primary purpose: answer questions from SQLite-backed records.

Include:

- A question field with examples such as “Where are my keys?” and “What moved most recently?”
- A readable answer area.
- Event ID and evidence references supporting the answer.
- An explicit unknown/no-evidence state.
- A short instruction that answers are based on saved observations and may be unknown when no event supports them.

Never make a model answer look like a database fact unless the returned response includes its supporting records.

### 4. Setup — objects and zones

Primary purpose: establish the bounded tabletop vocabulary.

Include:

- A control to load the three demo zones: `left_tray`, `center_mat`, and `right_tray`.
- A simple zone list or form showing names and normalized rectangle coordinates.
- Object enrollment with reference-image upload or snapshot, display name, aliases, and a visible saved reference card.
- A concise explanation that registered objects make event interpretation more conservative.
- A source indicator showing browser camera, fixture, or paired-image mode.

Do not require a JavaScript drawing canvas for the MVP. Keep the setup understandable and bounded to the documented object limit.

### 5. System & Privacy — honest status and controls

Primary purpose: make local operation and retention behavior inspectable.

Include:

- Actual model name and endpoint locality, without exposing secrets.
- Offline/local status and application health.
- Browser-frame health, model latency, event counts, and retained evidence count when supplied by the facade.
- Export Session control.
- Delete All Memory control with explicit confirmation and clear consequence text.
- Privacy explanation: no continuous application video by default; event evidence is retained according to the app policy; transient browser/Gradio files may exist and are cleaned up according to the documented policy.

Do not claim that no image bytes ever exist. Use the project’s precise privacy wording.

## Interaction and layout rules

- Use Python Gradio Blocks only.
- Keep primary actions visible and place diagnostic details in an optional secondary section.
- Use separate BEFORE and AFTER inputs; do not merge them into one ambiguous upload control.
- Make the manual Analyze action visually distinct from Start Watch.
- Show busy, success, warning, error, and disconnected states explicitly.
- Keep important instructions near the control they explain.
- Preserve evidence proportions and readable labels in previews.
- Mark mock data clearly as `MOCK`; never present fixtures as real memory.
- Keep all assets local. Do not load remote fonts, scripts, analytics, or CDN resources.
- Disable Gradio public sharing.
- Make the layout usable at the presenter laptop’s browser size without requiring terminal access from nontechnical users.

## Camera and fallback instructions

The camera belongs to the presenter laptop browser, not the headless Spark.

The Watch screen must communicate this sequence:

1. Open the app through the local browser/SSH-forwarded localhost URL.
2. Start the laptop webcam input and grant browser permission.
3. Confirm that frames are arriving before starting Watch Mode.
4. For a dependable demo, use the two explicit BEFORE/AFTER inputs first.
5. If the camera is denied, disconnected, or stale, upload PNG/JPEG images and continue manually.

Do not tell users to open the raw Spark address when browser webcam permissions require localhost or HTTPS. Do not suggest that the Spark has a USB camera.

## Coordination with Frontend B

Frontend B owns buttons, callbacks, event handlers, and callback return wiring. Frontend A should provide a screen map before implementation:

```text
screen/tab -> component ID -> displayed purpose -> expected facade data -> empty/error state
```

Coordinate before changing:

- Component IDs or callback input/output ordering.
- Facade method names or shared Pydantic models.
- The meaning of status badges, confidence labels, event evidence, or review states.
- The browser-camera stream and timer components.
- Any layout change that moves a component or changes its input/output contract.

Frontend A may specify that a control exists and how it should read; Frontend B implements the callback that invokes the facade. Neither frontend role changes Backend A’s model contract directly.

## Coordination with Backend A and Backend B

Coordinate with Backend A when presenting:

- Model health or latency.
- Event type and confidence labels.
- Uncertain results, repair failures, or model errors.
- Evidence wording derived from visible observations.

Coordinate with Backend B when presenting:

- Current locations and accepted events.
- Review/approve/reject states.
- Query answers and event IDs.
- Deletion, export, and retained evidence counts.

The UI must consume the shared facade. It must not import `VLMClient`, write SQLite, inspect model responses, or maintain authoritative event history.

## Accessibility and presentation checklist

- Every image input has a clear label and a short instruction.
- BEFORE and AFTER are visually distinguishable.
- Status colors are accompanied by text; color is not the only signal.
- Error text explains what the user can do next.
- Empty states explain why the area is empty and which action fills it.
- Destructive actions state what will be removed before confirmation.
- Technical terms such as `uncertain` and `review` have a plain-language explanation.
- The main demo can be completed without reading raw JSON or opening a terminal.

## Acceptance checklist

Verify in mock mode:

1. `python -m app.ui.app --mock --port 7861` launches without a model, camera, or database setup.
2. All five tabs are present and readable.
3. Mock status and fixture records are clearly marked `MOCK`.
4. A user can upload or capture two distinct BEFORE/AFTER images and see an evidence result without starting a stream.
5. The Ask screen shows an answer and supporting event evidence.
6. Memory shows current locations, event history, and review states.
7. Setup exposes demo zones and object enrollment instructions.
8. System & Privacy shows real-looking mock status without claiming it is live.
9. Camera-denied, stale, empty, and backend-error states are understandable.
10. Delete All Memory requires explicit confirmation.
11. The UI does not call inference from timers or stream callbacks.
12. `pytest tests/test_ui_contract.py` passes, with the exact result recorded in the handoff.

## Definition of done

Frontend A is complete when:

- The five screens provide a coherent end-to-end user journey.
- Manual image-pair analysis is obvious and usable as the reliable fallback.
- Live camera instructions correctly identify the browser laptop as the camera owner.
- Saved memory and answers visibly link back to evidence.
- Mock, empty, error, stale-camera, review, and destructive-action states are designed.
- The layout and component contract are documented for Frontend B.
- No UI text contradicts Backend A’s validation, confidence, or uncertainty behavior.
- Tests and visual/demo notes are recorded in `UI_HANDOFF.md`.

## Handoff

Record:

- Screen map and component IDs.
- Changed files.
- Mock launch command and test command.
- Exact test results.
- Screenshots or a concise visual review, if available.
- Required facade methods or missing backend data.
- Any unresolved callback wiring that Frontend B must complete.

Do not claim camera streaming or backend inference works based only on static layout inspection; record those as unverified until exercised through the browser and facade.
