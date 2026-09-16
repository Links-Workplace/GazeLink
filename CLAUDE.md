GAZELINK — Project Instructions

Mission

GAZELINK enables a person to operate a Windows computer using their eyes only.

The product is not complete when it merely detects eyes or estimates gaze. The target outcome is independent, reliable, and safe computer use for people who cannot use their hands.

Optimize decisions for these priorities, in order:

User safety and the ability to pause or regain control.

Reliable behavior and prevention of unintended actions.

Accessibility and ease of use.

Gaze accuracy and low latency.

Maintainability and extensibility.

Sources of truth

The project uses three coordinated sources of truth:

spec.MD — product vision, milestone scope, demos, and product-level Definition of Done.

TECHNICAL_SPEC.md — approved architecture, contracts, safety rules, quality targets, and technical acceptance criteria. Items explicitly marked as proposed remain proposals until approved or validated.

TASKS.md — execution plan, task dependencies, current status, checks, and completion evidence.

Product requirements in spec.MD take precedence over technical interpretation. Technical work must conform to TECHNICAL_SPEC.md; execution status must reflect repository reality in TASKS.md.

Before meaningful work:

Read the exact task and its dependencies in TASKS.md.

Read the relevant milestone in spec.MD and the matching sections in TECHNICAL_SPEC.md.

Inspect the actual repository structure and current implementation.

Resolve the Git root before running broad Git commands. Treat it as the project repository only if it is the project directory or an intentional project parent.

Identify the task ID, milestone, requested outcome, demo, technical acceptance criteria, and Definition of Done.

State important assumptions when the specification or implementation is incomplete.

At present, the project directory may sit below a user-home Git root. Do not scan, stage, diff, commit, or report unrelated user-home files as project changes. If GAZELINK has no dedicated repository yet, use direct workspace file inspection and state that Git status is unavailable for this project.

Do not invent requirements, dependencies, architecture, hardware assumptions, commands, or files that are not present in the repository or approved by the user.

If repository reality conflicts with any source-of-truth document, report the conflict before making a product-level decision. Do not silently rewrite the roadmap or technical baseline. Update spec.MD only when the user has approved a product decision or explicitly asked for a roadmap change.

After working on a tracked task, update its entry in TASKS.md in the same turn. Preserve the original What to do and Definition of Done; update status and add completion evidence, files changed, checks run, deviations, and remaining risks. Never mark a task DONE based only on code being written.

Product roadmap

Keep work aligned with the six milestones:

Milestone

Outcome

M1 — Vision

The system reliably detects the face, eyes, iris, eyelids, eye openness, and head pose.

M2 — Gaze

The system maps calibrated gaze to a confident screen coordinate.

M3 — Control

The user can safely control the Windows cursor and perform mouse actions using gaze and intentional eye gestures.

M4 — Usability

The user can browse, scroll, type, and use applications for sustained real-world tasks.

M5 — Platform

External applications can consume gaze, selection, blink, confidence, and calibration events through stable interfaces.

M6 — Product

A new user can install, calibrate, and use GAZELINK safely without a developer present.

Camera access, MediaPipe, head pose, EAR, filtering, smoothing, state machines, APIs, and similar technologies are implementation tools inside a milestone—not milestones by themselves.

Every substantial task must connect to a user-visible capability, measurable quality improvement, or explicit Definition of Done.

Core product constraints

4.1 Safety

Never allow low-confidence or missing tracking data to create clicks, drags, or uncontrolled cursor movement.

Freeze or safely disengage control when the face or gaze signal is lost.

Preserve a reliable pause/emergency mechanism.

Treat click, double-click, right-click, drag, dwell, and scroll as stateful safety-sensitive actions.

Prevent repeated or stuck actions after exceptions, camera loss, focus changes, or shutdown.

Keep a manual recovery path during development and testing.

Any code that moves the real cursor or emits OS input must be opt-in and clearly separated from simulation/test modes.

4.2 Intentional input

Do not treat every blink as a command.

Distinguish natural blinks from intentional gestures using duration, confidence, cooldowns, and state where appropriate.

Avoid false activation even if that slightly increases deliberate activation time.

Make thresholds configurable or calibratable rather than scattering unexplained constants.

4.3 Privacy

Process camera video locally by default.

Do not store or transmit raw video, face images, biometric samples, or calibration data unless the user explicitly approves a documented requirement.

Avoid logging raw frames or sensitive biometric data.

Store only the minimum profile/calibration data required, with clear ownership and lifecycle.

4.4 Performance and accuracy

Measure latency, FPS, confidence, jitter, calibration error, and loss/recovery behavior where relevant.

Prefer measured improvements over visual impressions alone.

Keep capture, inference, gaze estimation, smoothing, UI, and OS-input concerns separable so bottlenecks can be isolated.

Do not hide low confidence with aggressive smoothing; expose and handle uncertainty explicitly.

4.5 Accessibility

Core flows must not require a mouse or keyboard once eye control is active.

Provide clear visual feedback for focus, dwell progress, activation, pause state, calibration quality, and tracking loss.

Keep targets, timing, contrast, RTL behavior, and fatigue in mind.

Preserve Hebrew RTL and English support when editing user-facing interfaces or the on-screen keyboard.

Architecture principles

Until the repository defines a more specific architecture, preserve these logical boundaries:

Camera capture

Face, eye, iris, eyelid, and head-pose detection

Feature extraction and confidence

Calibration and gaze estimation

Filtering and smoothing

Gesture and interaction state machines

Cursor/OS input adapter

User interface and accessibility feedback

Profiles and settings

External API/SDK interfaces

Keep hardware/model-specific code behind adapters where practical. Keep pure math and state transitions independent of the camera and OS so they can be tested deterministically.

Avoid premature platform/API work before the current milestone's user outcome is working, unless the user explicitly requests it.

5.1 Mandatory OOP and modularity policy

Apply this policy to every new feature, fix, experiment, recorder, analysis tool, and refactor. Architecture is part of completion, not a later cleanup task. Deliver modular code in the current task; do not defer an avoidable monolith to a future refactor or wait for the user to request modularity again.

Use the repository's approved architecture and existing packages as the starting point. These rules extend the architecture principles above; they do not authorize replacing the approved design or reorganizing unrelated subsystems.

5.2 Responsibility and dependency map before implementation

Before meaningful code changes, inspect the affected code and include a short responsibility map in the implementation plan:

Existing components to reuse and their actual paths.

Each new or changed module/class, its single primary responsibility, and why it belongs in that package.

Input/output contracts, units, dependencies, and ownership of mutable state or resources.

The entry point that wires the components together and the relevant behavior checks.

For a small fix, a few sentences suffice. Do not invent a new framework or produce a large design document for routine work. Make routine implementation decisions autonomously within the approved scope.

5.3 OOP where it owns behavior; functions where they suffice

Give each class one cohesive responsibility and explicit state ownership. Encapsulate state transitions behind methods that preserve invariants; avoid unrelated public mutable flags.

Use classes for stateful services, resource lifecycles, interchangeable adapters, and domain behavior. Use typed data structures for configuration and results, and pure functions for stateless calculations. A class containing only unrelated static methods is not modular OOP.

Prefer composition over inheritance. Introduce a small Protocol or interface at an actual substitution boundary, such as a real/fake camera, model, clock, storage, or OS-input adapter. Do not build speculative class hierarchies or a dependency-injection framework.

Pass dependencies explicitly through constructors or function parameters. Do not hide them in mutable globals, service locators, singleton registries, or long closures sharing nonlocal state.

Keep domain calculations and state machines independent of CLI parsing, UI frameworks, camera libraries, file I/O, and OS input. Outer adapters depend on core contracts; core modules must not import entry-point scripts or concrete UI/OS adapters.

Create and wire resources in a clear composition root. One component owns each resource's startup and cleanup; cleanup must still occur when reporting or other shutdown steps fail.

Keep per-frame paths efficient. Modularity does not justify unnecessary copies, allocations, wrappers, or synchronization in the live pipeline.

5.4 Package layout, entry points, and file-size guardrails

Place code under the existing package for its responsibility. Reuse the post-refactor core packages where present and verified. Do not rebuild a parallel architecture inside a gf_*.py script, experiment, or scratchpad.

Keep CLI and launcher files thin: parse arguments, build configuration, wire components, invoke a use case, and report its result. Move algorithms, state machines, persistence, protocol definitions, rendering, and scoring into cohesive modules.

Use responsibility-based names. Do not split a large file into part1/part2, unrelated utils/helpers modules, or one equally large class. Folder organization must reflect real dependency and responsibility boundaries.

Review size before a module exceeds 400 physical lines, a class exceeds 200 lines, or a function/method exceeds 60 lines. These are review triggers, not evidence of a defect on their own.

Do not create or grow a hand-written source module beyond 600 physical lines by default. Split at a real responsibility boundary before delivery. A narrowly justified cohesive exception must be documented in the task evidence and examined by the existing section 7.5 reviewer; routine exceptions do not introduce a separate user-approval round.

Exclude generated code, vendored files, and data-only fixtures from these size limits. Do not evade the limits by compressing code, deleting useful documentation, or scattering one responsibility across arbitrary files. Organize tests by behavior or subsystem as they grow.

Existing oversized files are not permission to add unrelated behavior. For new responsibilities, extract or create the relevant component and leave only delegation/wiring in the old file. A small bug fix may remain local; do not launch a repository-wide refactor merely because a legacy file is large.

Preserve public entry points and compatibility with thin wrappers where needed. Keep extraction scoped to the current task and verify affected behavior.

5.5 Experiments and analysis tools follow the same rules

Temporary location is not an architecture exemption. Reusable experiment logic belongs in cohesive modules, with a thin runner; it must reuse core functionality instead of copying the production pipeline.

For a multi-pose experiment, for example, separate declarative pose definitions and instructions, recording orchestration, feature/coverage calculations, model comparison and decision rules, and report output. Choose concrete files and classes only after inspecting existing components; this is a responsibility map, not a requirement to create five new services.

Keep experiment constants and selection criteria in typed, discoverable configuration. Do not mix fitting, metric computation, acceptance decisions, plots, UI events, and file writes in one main() function. A genuinely short one-off stateless calculation may remain a function or small script.

5.6 Architecture acceptance gate

Before calling an implementation complete, the lead and the existing section 7.5 reviewer must inspect the actual changes and confirm:

New behavior is in the correct package, with a clear owner and focused interfaces.

Entry points remain thin; no duplicate pipeline, circular dependency, hidden mutable state, or oversized catch-all class/module was introduced.

Newly introduced responsibilities were not appended to an existing monolith for convenience.

Size-triggered modules/classes/functions were examined; any exception has a concrete reason and review outcome.

Relevant tests check public behavior, state transitions, contracts, and wiring. Test core logic without a live camera or real OS input; do not add brittle tests solely to assert a class exists or a file has a particular length.

Extracted behavior and public commands remain compatible, except for explicitly intended changes. Report checks actually run and any unverified hardware behavior.

Include a concise architecture summary in the final task report: responsibilities, affected paths, reuse decisions, size exceptions, and verification. Unresolved responsibility violations or unjustified monolith growth are acceptance blockers even when tests pass. Address them within the current task before marking it DONE. This gate is part of existing QA, not an additional review loop.

Multi-agent orchestration

Use the full multi-agent workflow when the user asks for agents, sub-agents, parallel work, delegation, or splitting the task. Also use it when a large task contains genuinely independent workstreams that can be completed without overlapping edits.

For small, tightly coupled, or single-file tasks, prefer one agent. More agents are useful only when their scopes can be isolated and their outputs can be independently verified.

6.1 Lead/orchestrator responsibilities

Claude Code is the main agent and owns planning, implementation, delegation, integration, and the final report. Use Claude Opus for the lead when available. A separate Claude Opus session with Extra High Effort when supported is the read-only QA reviewer described in section 7.5; it does not take over orchestration or edit the implementation. The Claude orchestrator owns the final result. It must:

Read spec.MD and inspect repository reality.

Identify the milestone, dependencies, risks, and acceptance criteria.

Divide work into narrow, independently verifiable scopes.

Assign exclusive file ownership whenever agents will edit in parallel.

Prevent two agents from editing the same file unless one is explicitly read-only.

Review every returned diff and not rely on summaries alone.

Integrate results in dependency order.

Run independent technical and scenario-level QA.

Send work back for correction when QA finds a problem.

Produce the final report and clearly distinguish verified facts from remaining risks.

Sub-agent output is a proposal until the lead has reviewed the actual code, tests, and behavior.

6.2 Model roles

When model selection is available, use these roles deliberately. In the active Claude-main/Opus-QA workflow, use Claude agents for implementation and a separate read-only Opus reviewer. The OpenAI implementation/support alternatives below apply only if the user explicitly changes this role allocation:

Claude Opus — lead/orchestrator: architecture, milestone decomposition, cross-cutting state and safety decisions, calibration strategy, privacy decisions, integration review, and final scenario QA. One orchestrator owns integration and completion even when several agents participate.

Claude Opus — separate read-only QA reviewer: review plans, actual changes, test quality, acceptance evidence, and unresolved risks. Use Extra High Effort when supported, following the availability and reporting rules in section 7.5. Findings return directly to the lead for evaluation and correction. Opus approval does not replace required user approval or hardware validation.

Claude Sonnet or OpenAI GPT-5.6 Terra — implementation agent: focused services, adapters, UI components, tests, refactors, and other well-scoped implementation with clear acceptance criteria.

Claude Haiku or OpenAI GPT-5.6 Luna — lightweight support: very simple, low-risk tasks such as read-only file discovery, exact reference searches, inventories, test-output summaries, and narrow consistency checks.

Do not use Haiku or Luna for implementation that changes behavior, architecture, biometric/privacy decisions, OS-input safety, gesture state machines, calibration model selection, integration, or final QA. Escalate the task to Sonnet/Terra or Opus/Sol when investigation reveals non-trivial judgment or code changes. Model choice never replaces file isolation, precise prompts, diff review, or independent verification.

6.3 Recommended decomposition

Split by stable boundary, not by arbitrary file count. Examples:

Vision pipeline and camera-loss recovery

Calibration and gaze mapping

Filtering, smoothing, and confidence policy

Blink/gesture state machine

Windows cursor/input adapter

Accessible UI and calibration feedback

Profiles/settings persistence

API/SDK contract

Automated tests and deterministic fixtures

Performance/accuracy measurement

Read-only security, privacy, or architecture review

Do not split coupled state transitions across separate agents. One owner should control an end-to-end state machine and its tests.

6.4 Required sub-agent prompt

Every delegated task must include:

Exact task ID from TASKS.md

Milestone and task goal

User-visible outcome or measurable result

Exact allowed files or subsystem

Files/subsystems to avoid

Relevant spec.MD constraints

Known dependencies and risks

Tests or measurements to run

Expected deliverable

Whether edits are allowed or the task is read-only

Required report format

Each sub-agent must report:

What it changed or found

Files touched

Tests/measurements run and their results

Assumptions and deviations

Remaining risks

Impact on the roadmap or other workstreams

6.5 Parallel work and conflict rules

Before parallel edits, the lead must create a file-ownership map. Shared/high-conflict files—such as central configuration, dependency manifests, application entry points, shared state models, roadmap files, and public API contracts—should normally have one editing owner.

Other agents may review a high-conflict file or propose a patch in prose, but the owner performs the integration.

Agents must not:

Undo or overwrite changes they did not create.

Reformat unrelated files.

broaden their scope without approval from the lead.

Commit, merge, or declare the overall task complete unless explicitly assigned that responsibility.

The lead must inspect project-scoped git status and the actual diff after every integration step, subject to the Git-root restrictions in section 2. If no valid project repository exists, inspect the changed project files directly.

6.6 Dependency order

Use this default order when applicable:

Contracts, data shapes, and acceptance metrics

Pure algorithms and state machines

Hardware/model/OS adapters

UI and end-to-end wiring

Tests, performance measurements, and scenario QA

Documentation and final integration review

Parallelize only work that does not depend on an unfinished contract or shared implementation decision.

Quality assurance

Passing compilation or unit tests is necessary but not sufficient for eye-control behavior.

7.1 Technical QA

Run the repository's existing relevant commands. Do not invent commands before inspecting the project configuration.

Verify, as applicable:

Focused unit and integration tests pass.

Type checking, linting, and formatting checks pass.

Camera/model resources are released on shutdown and error.

Threads, async tasks, timers, and event listeners terminate cleanly.

Numerical code handles missing values, invalid calibration, low confidence, and edge coordinates.

OS input can be replaced by a fake/simulation adapter in tests.

New dependencies are justified and compatible with the target platform.

No unrelated files or user changes were modified.

7.2 Mandatory scenario QA

For user-facing or safety-sensitive changes, verify relevant scenarios rather than only individual functions:

Normal happy path

No face detected

Face temporarily lost and recovered

One or both eyes occluded

Low-confidence or noisy gaze

Calibration incomplete, invalid, or stale

Natural blink versus intentional gesture

Gesture cooldown and repeated input

Pause and emergency recovery

Camera unavailable or disconnected

Application shutdown during an active interaction

Screen edges and multi-resolution behavior

Real OS input disabled in automated tests

For M2 and later, report measurable results when the repository provides the required harness or data: calibration error, validation error, jitter, latency, FPS, false activations, and recovery time.

7.3 Test quality review

Tests should assert behavior and safety properties, not merely that a function returned a value.

For stateful interactions, test transitions, forbidden transitions, cooldowns, reset behavior, and recovery after loss/error. A test should fail if low-confidence input can trigger an action or if a stale gesture can be replayed.

Prefer deterministic prerecorded landmarks/features or synthetic fixtures for algorithmic tests. Do not require a live camera for the normal automated test suite unless the project explicitly defines a hardware test tier.

7.4 Review after sub-agents

Before accepting delegated implementation, the lead must answer:

Which files changed, and why?

Did the agent stay within scope and file ownership?

Does the implementation match the relevant milestone and Definition of Done?

Are confidence, safety, and recovery paths correct?

Are tests meaningful and capable of detecting wrong wiring or unsafe fallback behavior?

Were privacy or accessibility assumptions introduced?

Are there duplicate pipelines, hidden global state, unexplained thresholds, or platform-specific coupling?

What was manually inspected, and what behavior was actually exercised?

If any answer is uncertain, the work is not yet verified.

7.5 Mandatory Claude Opus QA workflow

This workflow replaces the former Codex QA workflow with the user's explicit authorization. Claude Code remains the lead and owns implementation and corrections. A separate Claude Opus reviewer session or sub-agent performs read-only QA. A reread by the implementation agent alone does not satisfy this requirement.

Model and effort

Use Claude Opus with Extra High Effort for every required review when that combination is supported by the installed runtime. Inspect the available model and effort controls; do not invent model IDs, effort values, flags, or commands. If Extra High Effort is unavailable, use the highest supported effort for Opus and explicitly report the actual model and effort and the limitation. Do not silently substitute another model. If Opus or a separate review context is unavailable, record the required review as blocked rather than passed.

Authorized migration from Codex

The user has already authorized replacing the Codex review gate with this Opus workflow. Do not request the same approval again. Inspect the installed plugin's supported configuration and disable the legacy Codex review gate that enforces review for this project. Remove or update only the corresponding legacy review triggers and project instructions; preserve unrelated hooks, settings, and user changes. If the only supported switch has wider scope, report that scope and use the narrowest supported change. Do not change Codex authentication, billing, or model settings to repair a workflow that is being retired. Cancel obsolete project review jobs through supported mechanisms and verify that the old trigger is inactive. Do not keep invoking failed Codex reviews or repeat waiting messages.

Editing CLAUDE.md alone does not disable a runtime hook or configure a model. Report separately what instruction changes were made, what runtime configuration was verified, and what remains unavailable. If configuration cannot be changed with available access, report the precise blocker once and continue independent authorized work; do not mark missing Opus QA as passed.

Required review checkpoints

Before implementation: prepare the plan and send it to a separate Opus reviewer with the task requirements, relevant specification excerpts, affected components, assumptions, proposed tests, and acceptance criteria. Resolve material findings before implementing. Repeat plan review when architecture, scope, or safety assumptions materially change. A short plan is sufficient for a trivial change.

After each coherent unit of work: run relevant checks, then request Opus review of the actual scoped changes and evidence before starting dependent work. A unit is a bounded task, bug fix, or integrated sub-agent result, not every file write. Independent workstreams may continue under section 6's ownership rules.

After corrections: the lead evaluates findings and implements justified fixes. Send the corrected changes and updated checks to the reviewer. Earlier approval does not cover later edits.

Before completion: obtain final Opus review of the current scoped changes, acceptance criteria, test evidence, and relevant safety scenarios. Record the result in TASKS.md. No Codex approval or Codex Stop gate is required.

Review inputs and scope

Give the reviewer the task ID, milestone, requested outcome, relevant source-of-truth sections, exact scope, plan or implementation, changed files, check results, known limitations, and unresolved findings. Have the reviewer inspect actual code, tests, and permitted evidence directly, not only the implementer's summary. Request actionable findings, locations, reasoning, missing evidence, and a distinction between acceptance blockers and optional improvements.

Resolve the project Git root under section 2. Never review an unrelated user-home repository. If no valid project repository exists, use explicitly scoped file inspection. Share only the minimum necessary code, documentation, and non-sensitive evidence. Preserve section 4.3: do not send raw video, face images, biometric samples, calibration data, credentials, or unrelated user files without the required authorization.

The reviewer is read-only: no implementation edits, dependency changes, roadmap rewrites, commits, or real OS input. The lead owns corrections. Use fake OS-input adapters for automated checks. Separate review context reduces reliance on the implementer's narrative; it is not a guarantee of independent judgment or correctness.

Findings, retries, and completion

Evaluate findings against actual code and approved requirements. Fix substantiated defects; explain disputed findings with evidence and request re-evaluation. Do not expand product scope merely to satisfy the reviewer.

Allow at most three correction-and-review cycles per unit. If a blocker remains, findings repeat without progress, or a product decision is required, report the unresolved issue and stop dependent implementation. Do not loop failed reviews or repeatedly ask for an already granted approval. This is a workflow rule, not a claim about automatic hook enforcement.

Record the review scope, reviewer session or agent identifier when exposed, actual model, actual effort (or explicitly unverified), outcome, findings, corrections, checks, and limitations in TASKS.md. Never claim Opus or Extra High Effort was used without runtime evidence. Never mark DONE while acceptance-blocking findings remain or required review is missing.

Replacing the reviewer does not weaken regression gates, privacy rules, acceptance criteria, required live validation, or explicit-user-approval requirements for real OS input and commits. Opus review complements technical and scenario QA; it cannot certify hardware tests, usability, gaze accuracy, or safety that have not actually been measured.

Working practices

Prefer small, reviewable changes connected to one outcome.

Read before editing and follow established conventions once code exists.

Preserve user changes and avoid destructive Git operations.

Never silently replace an existing implementation or dependency choice.

Add dependencies only when they materially improve the requested outcome; explain the tradeoff.

Put configurable thresholds and tunable parameters in one discoverable place.

Document units for time, distance, angles, screen coordinates, and confidence values.

Use monotonic time for gesture durations and cooldowns.

Keep logs useful but free of raw biometric/video data.

Do not claim behavior was tested with real hardware or users unless it actually was.

Completion checklist

Before the final response:

Re-read the requested outcome, exact TASKS.md entry, relevant spec.MD milestone, and matching TECHNICAL_SPEC.md sections.

Review project-scoped git status and the complete diff only after resolving a valid project Git root; otherwise inspect the scoped files directly as required in section 2.

Confirm only intended files changed.

Verify the mandatory OOP and modularity policy in sections 5.1-5.6: responsibility boundaries, thin entry points, reuse, size exceptions, and architecture review evidence.

Run relevant tests and static checks available in the repository.

Verify applicable safety, confidence, recovery, privacy, and accessibility scenarios.

Confirm no real cursor/click action can occur unexpectedly during automated testing.

Record anything that could not be tested, especially hardware-dependent behavior.

Complete the Opus QA checkpoints in section 7.5 for the current work; resolve acceptance-blocking findings or report QA as blocked.

Update the task status and completion evidence in TASKS.md without removing the original acceptance criteria, including the Opus review outcome and remaining limitations.

The final response must report:

Task IDs addressed

Milestone/outcome addressed

Files changed

What was manually reviewed

Scenarios verified

Tests/checks run and results

Measured performance/accuracy results, if applicable

Opus QA outcome, actual reviewer model and effort, corrections, disputed findings, and any review blocker

Local/commit status

Remaining risks, hardware validation, or follow-up work

If sub-agents were used, also report:

Each agent's scope

Files each agent touched

What the lead independently reviewed and verified

Corrections made after integration QA

Final orchestration QA result

Never present delegated work as complete before lead-agent verification.