# GAZELINK — Project Instructions

## 1. Mission

GAZELINK enables a person to operate a Windows computer using their eyes only.

The product is not complete when it merely detects eyes or estimates gaze. The target outcome is independent, reliable, and safe computer use for people who cannot use their hands.

Optimize decisions for these priorities, in order:

1. User safety and the ability to pause or regain control.
2. Reliable behavior and prevention of unintended actions.
3. Accessibility and ease of use.
4. Gaze accuracy and low latency.
5. Maintainability and extensibility.

## 2. Sources of truth

The project uses three coordinated sources of truth:

- `spec.MD` — product vision, milestone scope, demos, and product-level Definition of Done.
- `TECHNICAL_SPEC.md` — approved architecture, contracts, safety rules, quality targets, and technical acceptance criteria. Items explicitly marked as proposed remain proposals until approved or validated.
- `TASKS.md` — execution plan, task dependencies, current status, checks, and completion evidence.

Product requirements in `spec.MD` take precedence over technical interpretation. Technical work must conform to `TECHNICAL_SPEC.md`; execution status must reflect repository reality in `TASKS.md`.

Before meaningful work:

1. Read the exact task and its dependencies in `TASKS.md`.
2. Read the relevant milestone in `spec.MD` and the matching sections in `TECHNICAL_SPEC.md`.
3. Inspect the actual repository structure and current implementation.
4. Resolve the Git root before running broad Git commands. Treat it as the project repository only if it is the project directory or an intentional project parent.
5. Identify the task ID, milestone, requested outcome, demo, technical acceptance criteria, and Definition of Done.
6. State important assumptions when the specification or implementation is incomplete.

At present, the project directory may sit below a user-home Git root. Do not scan, stage, diff, commit, or report unrelated user-home files as project changes. If GAZELINK has no dedicated repository yet, use direct workspace file inspection and state that Git status is unavailable for this project.

Do not invent requirements, dependencies, architecture, hardware assumptions, commands, or files that are not present in the repository or approved by the user.

If repository reality conflicts with any source-of-truth document, report the conflict before making a product-level decision. Do not silently rewrite the roadmap or technical baseline. Update `spec.MD` only when the user has approved a product decision or explicitly asked for a roadmap change.

After working on a tracked task, update its entry in `TASKS.md` in the same turn. Preserve the original `What to do` and `Definition of Done`; update status and add completion evidence, files changed, checks run, deviations, and remaining risks. Never mark a task `DONE` based only on code being written.

## 3. Product roadmap

Keep work aligned with the six milestones:

| Milestone | Outcome |
| --- | --- |
| M1 — Vision | The system reliably detects the face, eyes, iris, eyelids, eye openness, and head pose. |
| M2 — Gaze | The system maps calibrated gaze to a confident screen coordinate. |
| M3 — Control | The user can safely control the Windows cursor and perform mouse actions using gaze and intentional eye gestures. |
| M4 — Usability | The user can browse, scroll, type, and use applications for sustained real-world tasks. |
| M5 — Platform | External applications can consume gaze, selection, blink, confidence, and calibration events through stable interfaces. |
| M6 — Product | A new user can install, calibrate, and use GAZELINK safely without a developer present. |

Camera access, MediaPipe, head pose, EAR, filtering, smoothing, state machines, APIs, and similar technologies are implementation tools inside a milestone—not milestones by themselves.

Every substantial task must connect to a user-visible capability, measurable quality improvement, or explicit Definition of Done.

## 4. Core product constraints

### 4.1 Safety

- Never allow low-confidence or missing tracking data to create clicks, drags, or uncontrolled cursor movement.
- Freeze or safely disengage control when the face or gaze signal is lost.
- Preserve a reliable pause/emergency mechanism.
- Treat click, double-click, right-click, drag, dwell, and scroll as stateful safety-sensitive actions.
- Prevent repeated or stuck actions after exceptions, camera loss, focus changes, or shutdown.
- Keep a manual recovery path during development and testing.
- Any code that moves the real cursor or emits OS input must be opt-in and clearly separated from simulation/test modes.

### 4.2 Intentional input

- Do not treat every blink as a command.
- Distinguish natural blinks from intentional gestures using duration, confidence, cooldowns, and state where appropriate.
- Avoid false activation even if that slightly increases deliberate activation time.
- Make thresholds configurable or calibratable rather than scattering unexplained constants.

### 4.3 Privacy

- Process camera video locally by default.
- Do not store or transmit raw video, face images, biometric samples, or calibration data unless the user explicitly approves a documented requirement.
- Avoid logging raw frames or sensitive biometric data.
- Store only the minimum profile/calibration data required, with clear ownership and lifecycle.

### 4.4 Performance and accuracy

- Measure latency, FPS, confidence, jitter, calibration error, and loss/recovery behavior where relevant.
- Prefer measured improvements over visual impressions alone.
- Keep capture, inference, gaze estimation, smoothing, UI, and OS-input concerns separable so bottlenecks can be isolated.
- Do not hide low confidence with aggressive smoothing; expose and handle uncertainty explicitly.

### 4.5 Accessibility

- Core flows must not require a mouse or keyboard once eye control is active.
- Provide clear visual feedback for focus, dwell progress, activation, pause state, calibration quality, and tracking loss.
- Keep targets, timing, contrast, RTL behavior, and fatigue in mind.
- Preserve Hebrew RTL and English support when editing user-facing interfaces or the on-screen keyboard.

## 5. Architecture principles

Until the repository defines a more specific architecture, preserve these logical boundaries:

1. Camera capture
2. Face, eye, iris, eyelid, and head-pose detection
3. Feature extraction and confidence
4. Calibration and gaze estimation
5. Filtering and smoothing
6. Gesture and interaction state machines
7. Cursor/OS input adapter
8. User interface and accessibility feedback
9. Profiles and settings
10. External API/SDK interfaces

Keep hardware/model-specific code behind adapters where practical. Keep pure math and state transitions independent of the camera and OS so they can be tested deterministically.

Avoid premature platform/API work before the current milestone's user outcome is working, unless the user explicitly requests it.

## 6. Multi-agent orchestration

Use the full multi-agent workflow when the user asks for agents, sub-agents, parallel work, delegation, or splitting the task. Also use it when a large task contains genuinely independent workstreams that can be completed without overlapping edits.

For small, tightly coupled, or single-file tasks, prefer one agent. More agents are useful only when their scopes can be isolated and their outputs can be independently verified.

### 6.1 Lead/orchestrator responsibilities

The lead agent must use **Codex Opus** or **OpenAI GPT-5.6 Sol** when either is available. The orchestrator owns the final result. It must:

1. Read `spec.MD` and inspect repository reality.
2. Identify the milestone, dependencies, risks, and acceptance criteria.
3. Divide work into narrow, independently verifiable scopes.
4. Assign exclusive file ownership whenever agents will edit in parallel.
5. Prevent two agents from editing the same file unless one is explicitly read-only.
6. Review every returned diff and not rely on summaries alone.
7. Integrate results in dependency order.
8. Run independent technical and scenario-level QA.
9. Send work back for correction when QA finds a problem.
10. Produce the final report and clearly distinguish verified facts from remaining risks.

Sub-agent output is a proposal until the lead has reviewed the actual code, tests, and behavior.

### 6.2 Model roles

When model selection is available, use these provider-equivalent roles deliberately:

- **Codex Opus or OpenAI GPT-5.6 Sol — lead/orchestrator:** architecture, milestone decomposition, cross-cutting state and safety decisions, calibration strategy, privacy decisions, integration review, and final scenario QA. One orchestrator owns integration and completion even when several agents participate.
- **Codex Sonnet or OpenAI GPT-5.6 Terra — implementation agent:** focused services, adapters, UI components, tests, refactors, and other well-scoped implementation with clear acceptance criteria.
- **Codex Haiku or OpenAI GPT-5.6 Luna — lightweight support:** very simple, low-risk tasks such as read-only file discovery, exact reference searches, inventories, test-output summaries, and narrow consistency checks.

Do not use Haiku or Luna for implementation that changes behavior, architecture, biometric/privacy decisions, OS-input safety, gesture state machines, calibration model selection, integration, or final QA. Escalate the task to Sonnet/Terra or Opus/Sol when investigation reveals non-trivial judgment or code changes. Model choice never replaces file isolation, precise prompts, diff review, or independent verification.

### 6.3 Recommended decomposition

Split by stable boundary, not by arbitrary file count. Examples:

- Vision pipeline and camera-loss recovery
- Calibration and gaze mapping
- Filtering, smoothing, and confidence policy
- Blink/gesture state machine
- Windows cursor/input adapter
- Accessible UI and calibration feedback
- Profiles/settings persistence
- API/SDK contract
- Automated tests and deterministic fixtures
- Performance/accuracy measurement
- Read-only security, privacy, or architecture review

Do not split coupled state transitions across separate agents. One owner should control an end-to-end state machine and its tests.

### 6.4 Required sub-agent prompt

Every delegated task must include:

- Exact task ID from `TASKS.md`
- Milestone and task goal
- User-visible outcome or measurable result
- Exact allowed files or subsystem
- Files/subsystems to avoid
- Relevant `spec.MD` constraints
- Known dependencies and risks
- Tests or measurements to run
- Expected deliverable
- Whether edits are allowed or the task is read-only
- Required report format

Each sub-agent must report:

- What it changed or found
- Files touched
- Tests/measurements run and their results
- Assumptions and deviations
- Remaining risks
- Impact on the roadmap or other workstreams

### 6.5 Parallel work and conflict rules

Before parallel edits, the lead must create a file-ownership map. Shared/high-conflict files—such as central configuration, dependency manifests, application entry points, shared state models, roadmap files, and public API contracts—should normally have one editing owner.

Other agents may review a high-conflict file or propose a patch in prose, but the owner performs the integration.

Agents must not:

- Undo or overwrite changes they did not create.
- Reformat unrelated files.
- broaden their scope without approval from the lead.
- Commit, merge, or declare the overall task complete unless explicitly assigned that responsibility.

The lead must inspect `git status` and the actual diff after every integration step.

### 6.6 Dependency order

Use this default order when applicable:

1. Contracts, data shapes, and acceptance metrics
2. Pure algorithms and state machines
3. Hardware/model/OS adapters
4. UI and end-to-end wiring
5. Tests, performance measurements, and scenario QA
6. Documentation and final integration review

Parallelize only work that does not depend on an unfinished contract or shared implementation decision.

## 7. Quality assurance

Passing compilation or unit tests is necessary but not sufficient for eye-control behavior.

### 7.1 Technical QA

Run the repository's existing relevant commands. Do not invent commands before inspecting the project configuration.

Verify, as applicable:

- Focused unit and integration tests pass.
- Type checking, linting, and formatting checks pass.
- Camera/model resources are released on shutdown and error.
- Threads, async tasks, timers, and event listeners terminate cleanly.
- Numerical code handles missing values, invalid calibration, low confidence, and edge coordinates.
- OS input can be replaced by a fake/simulation adapter in tests.
- New dependencies are justified and compatible with the target platform.
- No unrelated files or user changes were modified.

### 7.2 Mandatory scenario QA

For user-facing or safety-sensitive changes, verify relevant scenarios rather than only individual functions:

- Normal happy path
- No face detected
- Face temporarily lost and recovered
- One or both eyes occluded
- Low-confidence or noisy gaze
- Calibration incomplete, invalid, or stale
- Natural blink versus intentional gesture
- Gesture cooldown and repeated input
- Pause and emergency recovery
- Camera unavailable or disconnected
- Application shutdown during an active interaction
- Screen edges and multi-resolution behavior
- Real OS input disabled in automated tests

For M2 and later, report measurable results when the repository provides the required harness or data: calibration error, validation error, jitter, latency, FPS, false activations, and recovery time.

### 7.3 Test quality review

Tests should assert behavior and safety properties, not merely that a function returned a value.

For stateful interactions, test transitions, forbidden transitions, cooldowns, reset behavior, and recovery after loss/error. A test should fail if low-confidence input can trigger an action or if a stale gesture can be replayed.

Prefer deterministic prerecorded landmarks/features or synthetic fixtures for algorithmic tests. Do not require a live camera for the normal automated test suite unless the project explicitly defines a hardware test tier.

### 7.4 Review after sub-agents

Before accepting delegated implementation, the lead must answer:

- Which files changed, and why?
- Did the agent stay within scope and file ownership?
- Does the implementation match the relevant milestone and Definition of Done?
- Are confidence, safety, and recovery paths correct?
- Are tests meaningful and capable of detecting wrong wiring or unsafe fallback behavior?
- Were privacy or accessibility assumptions introduced?
- Are there duplicate pipelines, hidden global state, unexplained thresholds, or platform-specific coupling?
- What was manually inspected, and what behavior was actually exercised?

If any answer is uncertain, the work is not yet verified.

## 8. Working practices

- Prefer small, reviewable changes connected to one outcome.
- Read before editing and follow established conventions once code exists.
- Preserve user changes and avoid destructive Git operations.
- Never silently replace an existing implementation or dependency choice.
- Add dependencies only when they materially improve the requested outcome; explain the tradeoff.
- Put configurable thresholds and tunable parameters in one discoverable place.
- Document units for time, distance, angles, screen coordinates, and confidence values.
- Use monotonic time for gesture durations and cooldowns.
- Keep logs useful but free of raw biometric/video data.
- Do not claim behavior was tested with real hardware or users unless it actually was.

## 9. Completion checklist

Before the final response:

1. Re-read the requested outcome, exact `TASKS.md` entry, relevant `spec.MD` milestone, and matching `TECHNICAL_SPEC.md` sections.
2. Review `git status` and the complete diff.
3. Confirm only intended files changed.
4. Run relevant tests and static checks available in the repository.
5. Verify applicable safety, confidence, recovery, privacy, and accessibility scenarios.
6. Confirm no real cursor/click action can occur unexpectedly during automated testing.
7. Record anything that could not be tested, especially hardware-dependent behavior.
8. Update the task status and completion evidence in `TASKS.md` without removing the original acceptance criteria.

The final response must report:

- Task IDs addressed
- Milestone/outcome addressed
- Files changed
- What was manually reviewed
- Scenarios verified
- Tests/checks run and results
- Measured performance/accuracy results, if applicable
- Local/commit status
- Remaining risks, hardware validation, or follow-up work

If sub-agents were used, also report:

- Each agent's scope
- Files each agent touched
- What the lead independently reviewed and verified
- Corrections made after integration QA
- Final orchestration QA result

Never present delegated work as complete before lead-agent verification.
