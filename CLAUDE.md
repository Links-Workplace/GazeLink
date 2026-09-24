# GAZELINK — Project Instructions

## 0. Mission

GAZELINK enables a person to operate a Windows computer **using their eyes only**.

The product is not complete when it detects eyes or estimates gaze. The target is
**independent, reliable, safe computer use for people who cannot use their hands.**

Optimize every decision in this order:

1. User safety and the ability to pause or regain control
2. Reliable behavior; prevention of unintended actions
3. Accessibility and ease of use
4. Gaze accuracy and low latency
5. Maintainability and extensibility

---

## 1. Sources of truth

| Document | Authority |
|---|---|
| `spec.MD` | Product vision, milestone scope, demos, product-level DoD. **Takes precedence over technical interpretation.** |
| `TECHNICAL_SPEC.md` | Approved architecture, contracts, safety rules, quality targets. Items marked *proposed* stay proposals until approved. |
| `TASKS.md` | Execution status and completion evidence. **Must reflect repository reality.** |
| `README.md` | Orientation: what actually works today, what is blocked, where the code is. |
| `COMMANDS.MD` | Every command, verified against the code. |

**Before meaningful work:**

1. Read the exact task and its dependencies in `TASKS.md`.
2. Read the relevant milestone in `spec.MD` and the matching `TECHNICAL_SPEC.md` sections.
3. **Inspect the actual repository.** Do not trust a document over the code.
4. Identify: task ID, milestone, requested outcome, demo, acceptance criteria, DoD.
5. State important assumptions where the spec or implementation is incomplete.

**Never invent** requirements, dependencies, architecture, hardware assumptions,
commands, or files that are not in the repository or approved by the user.

**On conflict between repository reality and a source-of-truth document:** report the
conflict before making a product-level decision. Do not silently rewrite the roadmap.
Update `spec.MD` only on explicit user approval.

**After working a tracked task, update its `TASKS.md` entry in the same turn.** Preserve
the original *What to do* and *Definition of Done*; add status, completion evidence,
files changed, checks run, deviations, and remaining risks.
**Never mark DONE because code was written.**

### Git scope

Resolve the Git root before running broad Git commands. Treat it as the project repo
only if it is the project directory or an intentional project parent. The project may
sit below a user-home Git root — **do not scan, stage, diff, or report unrelated
user-home files as project changes.** If there is no dedicated repo, inspect workspace
files directly and say that Git status is unavailable.

---

## 2. Where the code is

```
focus/
├── gazefollower/     ← all active work
│   └── gazelink_core/
└── src/gazelink/     ← suspended legacy path. Do not develop here.
```

Active engine: GazeFollower library (`MGazeNet`) + our `gazelink_core/` layer.
Suspended and not deleted: the standalone engine, the EyeGestures adapter, Tobii Nexus.
**Do not develop in the suspended paths and do not quote numbers from them.**

---

## 3. Roadmap

| Milestone | Outcome |
|---|---|
| **M1 — Vision** | Reliable detection of face, eyes, iris, eyelids, openness, head pose |
| **M2 — Gaze** | Calibrated gaze mapped to a confident screen coordinate |
| **M3 — Control** | Safe cursor control and mouse actions via gaze and intentional gestures |
| **M4 — Usability** | Browse, scroll, type, use applications for sustained real tasks |
| **M5 — Platform** | External apps consume gaze, selection, blink, confidence, calibration events |
| **M6 — Product** | A new user can install, calibrate, and use GAZELINK safely, unaided |

Camera access, MediaPipe, head pose, EAR, filtering, smoothing, state machines and APIs
are **implementation tools inside a milestone — never milestones themselves.**

Every substantial task must connect to a user-visible capability, a measurable quality
improvement, or an explicit DoD.

---

## 4. Product constraints

### 4.1 Safety

- Low-confidence or missing tracking data must **never** produce clicks, drags, or
  uncontrolled cursor movement.
- Freeze or safely disengage control when face or gaze signal is lost.
- Preserve a reliable pause/emergency mechanism.
- Click, double-click, right-click, drag, dwell and scroll are **stateful,
  safety-sensitive** actions.
- Prevent repeated or stuck actions after exceptions, camera loss, focus changes, or
  shutdown. **No mouse button may remain held.**
- Keep a manual recovery path during development and testing.
- **Any code that moves the real cursor or emits OS input is opt-in and clearly
  separated from simulation/test modes.**

### 4.2 Intentional input

- Do not treat every blink as a command.
- Separate natural blinks from intentional gestures using duration, confidence,
  cooldowns and state.
- **Avoid false activation even at the cost of slightly slower deliberate activation.**
- Thresholds are configurable or calibratable — never unexplained constants scattered
  through the code.

### 4.3 Privacy

- Camera video is processed **locally** by default.
- **Never** store or transmit raw video, face images, biometric samples, landmarks or
  calibration data without explicit, documented user approval.
- Never log raw frames or biometric data.
- Recordings hold derived embeddings only, under `recordings/` with `.gitignore`, purged
  via `gf_purge.py`. **Never send them to any agent or external service.**
- Store the minimum profile/calibration data needed, with clear ownership and lifecycle.

### 4.4 Performance and accuracy

- Measure latency, FPS, confidence, jitter, calibration error, and loss/recovery
  behavior where relevant. **Prefer measured improvements over visual impressions.**
- Keep capture, inference, gaze estimation, smoothing, UI and OS-input separable so
  bottlenecks can be isolated.
- **Do not hide low confidence behind aggressive smoothing.** Expose and handle
  uncertainty explicitly.
- Numbers from two runs are not comparable unless recording, training budget, camera
  position and protocol match.

### 4.5 Accessibility

- Core flows must not require mouse or keyboard once eye control is active.
- Provide clear visual feedback for focus, dwell progress, activation, pause state,
  calibration quality and tracking loss.
- Keep target size, timing, contrast, RTL behavior and fatigue in mind.
- **Preserve Hebrew RTL and English support** in any user-facing interface or the
  on-screen keyboard.

---

## 5. Architecture and OOP policy

### 5.1 Logical boundaries

Camera capture · detection (face/eye/iris/eyelid/head pose) · feature extraction and
confidence · calibration and gaze estimation · filtering and smoothing · gesture and
interaction state machines · cursor/OS input adapter · UI and accessibility feedback ·
profiles and settings · external API/SDK.

Keep hardware/model-specific code behind adapters. **Keep pure math and state
transitions independent of the camera and OS so they can be tested deterministically.**
Avoid platform/API work before the current milestone's user outcome works.

These boundaries are enforced by `tests/test_architecture_boundaries.py`.

### 5.2 The policy applies to everything

Every feature, fix, experiment, recorder, analysis tool and refactor.
**Architecture is part of completion, not later cleanup.** Deliver modular code in the
current task; do not defer an avoidable monolith or wait to be asked again.

Start from the approved architecture and existing packages. These rules extend §5.1;
they do not authorize replacing the approved design or reorganizing unrelated subsystems.

### 5.3 Responsibility map before implementation

Before meaningful code changes, include a short responsibility map in the plan:

- Existing components to reuse, with actual paths
- Each new/changed module or class, its **single** primary responsibility, and why it
  belongs in that package
- I/O contracts, units, dependencies, ownership of mutable state and resources
- The entry point that wires it together, and the behavior checks

For a small fix, a few sentences suffice. **Do not invent a framework or write a large
design document for routine work.** Make routine decisions autonomously in scope.

### 5.4 OOP where it owns behavior; functions where they suffice

- One cohesive responsibility and explicit state ownership per class. Encapsulate state
  transitions behind methods that preserve invariants. No unrelated public mutable flags.
- **Classes** for stateful services, resource lifecycles, interchangeable adapters,
  domain behavior. **Typed data structures** for config and results. **Pure functions**
  for stateless calculations. A class of unrelated static methods is not modular OOP.
- **Composition over inheritance.** Introduce a Protocol/interface only at a real
  substitution boundary — real vs. fake camera, model, clock, storage, OS input.
  No speculative hierarchies, no DI framework.
- **Pass dependencies explicitly** through constructors or parameters. Never through
  mutable globals, service locators, singleton registries or long closures sharing
  nonlocal state.
- Domain calculations and state machines stay independent of CLI parsing, UI frameworks,
  camera libraries, file I/O and OS input. Outer adapters depend on core contracts;
  **core modules must never import entry-point scripts or concrete UI/OS adapters.**
- Create and wire resources in a clear **composition root**. One component owns each
  resource's startup and cleanup; **cleanup must still run when reporting or other
  shutdown steps fail.**
- Keep per-frame paths efficient. Modularity does not justify unnecessary copies,
  allocations, wrappers or synchronization in the live pipeline.

### 5.5 Layout, entry points, size guardrails

- Place code under the existing package for its responsibility. **Do not rebuild a
  parallel architecture inside a `gf_*.py` script, experiment or scratchpad.**
- **Keep CLI and launcher files thin:** parse args, build config, wire components,
  invoke a use case, report the result. Algorithms, state machines, persistence,
  protocol definitions, rendering and scoring go into cohesive modules.
- Responsibility-based names. Never `part1`/`part2`, never a generic `utils`/`helpers`
  dump, never one equally large class. Folders reflect real dependency boundaries.
- **Review triggers** (not defects on their own): module > 400 lines, class > 200 lines,
  function > 60 lines.
- **Hard default: do not create or grow a hand-written module beyond 600 lines.** Split
  at a real responsibility boundary before delivery. A narrowly justified cohesive
  exception must be documented in the task evidence and examined by the §7.5 reviewer —
  it does not require a separate user-approval round.
- Generated code, vendored files and data-only fixtures are excluded. **Do not evade the
  limits** by compressing code, deleting useful documentation, or scattering one
  responsibility across arbitrary files.
- **An existing oversized file is not permission to add unrelated behavior.** For a new
  responsibility, extract or create the component and leave only delegation in the old
  file. A small bug fix may stay local; do not launch a repo-wide refactor because a
  legacy file is large.
- Preserve public entry points with thin wrappers where needed. Keep extraction scoped
  to the current task and verify affected behavior.

### 5.6 Experiments follow the same rules

**Temporary location is not an architecture exemption.** Reusable experiment logic goes
into cohesive modules with a thin runner, and **reuses core functionality instead of
copying the production pipeline.**

Separate declarative definitions and instructions, recording orchestration, feature and
coverage calculations, model comparison and decision rules, and report output. Choose
concrete files and classes only after inspecting existing components.

Experiment constants and selection criteria go in typed, discoverable configuration.
**Do not mix fitting, metrics, acceptance decisions, plots, UI events and file writes in
one `main()`.** A genuinely short one-off stateless calculation may stay a function.

### 5.7 Architecture acceptance gate

Before calling an implementation complete, the lead **and** the §7.5 reviewer inspect the
actual changes and confirm:

- New behavior is in the correct package, with a clear owner and focused interfaces
- Entry points remain thin; no duplicate pipeline, circular dependency, hidden mutable
  state, or oversized catch-all class
- New responsibilities were **not** appended to an existing monolith for convenience
- Size-triggered modules/classes/functions were examined; exceptions have a concrete
  reason and a review outcome
- Tests check public behavior, state transitions, contracts and wiring — **without a live
  camera or real OS input.** No brittle tests asserting a class exists or a file length
- Extracted behavior and public commands stay compatible, except where change was intended
- Checks actually run are reported, along with any unverified hardware behavior

Include a concise architecture summary in the final report: responsibilities, affected
paths, reuse decisions, size exceptions, verification.
**Unresolved responsibility violations or unjustified monolith growth are acceptance
blockers even when tests pass.** Fix them inside the current task before DONE.
This gate is part of existing QA, not an extra review loop.

---

## 6. Multi-agent orchestration

Use the full workflow when the user asks for agents, sub-agents, parallel work,
delegation or splitting — **or** when a large task contains genuinely independent
workstreams with no overlapping edits.

For small, tightly coupled or single-file tasks, **prefer one agent.** More agents help
only when scopes can be isolated and outputs independently verified.

### 6.1 Lead responsibilities

Claude Code is the main agent and owns planning, implementation, delegation, integration
and the final report. Use Claude Opus for the lead when available. The lead must:

- Read `spec.MD` and inspect repository reality
- Identify milestone, dependencies, risks, acceptance criteria
- Divide work into narrow, independently verifiable scopes
- **Assign exclusive file ownership** whenever agents edit in parallel
- Prevent two agents editing the same file unless one is explicitly read-only
- **Review every returned diff — never rely on summaries**
- Integrate in dependency order
- Run independent technical and scenario QA
- Send work back for correction when QA finds a problem
- Produce the final report, separating **verified facts** from **remaining risks**

**Sub-agent output is a proposal until the lead has reviewed the actual code, tests and
behavior.**

### 6.2 Model roles

The active workflow is Claude-main implementation + a separate read-only Opus reviewer.
The OpenAI alternatives below apply **only** if the user explicitly changes this.

| Role | Model | Scope |
|---|---|---|
| Lead / orchestrator | **Claude Opus** | Architecture, decomposition, cross-cutting state and safety, calibration strategy, privacy, integration review, final scenario QA. **One orchestrator owns integration and completion.** |
| Read-only QA reviewer | **Claude Opus**, separate session | Plans, actual changes, test quality, acceptance evidence, unresolved risks. Extra High Effort where supported (§7.5). Findings go to the lead. **Opus approval never replaces user approval or hardware validation.** |
| Implementation agent | Claude Sonnet / GPT-5.6 Terra | Focused services, adapters, UI components, tests, refactors with clear acceptance criteria |
| Lightweight support | Claude Haiku / GPT-5.6 Luna | Read-only discovery, exact reference searches, inventories, test-output summaries, narrow consistency checks |

**Never use Haiku/Luna** for anything changing behavior, architecture, biometric/privacy
decisions, OS-input safety, gesture state machines, calibration model selection,
integration or final QA. Escalate to Sonnet/Terra or Opus when non-trivial judgment
appears. **Model choice never replaces file isolation, precise prompts, diff review or
independent verification.**

### 6.3 Decomposition

Split by stable boundary, not file count: vision pipeline and camera-loss recovery ·
calibration and gaze mapping · filtering, smoothing, confidence policy · blink/gesture
state machine · Windows input adapter · accessible UI and calibration feedback ·
profiles/settings persistence · API/SDK contract · tests and deterministic fixtures ·
performance/accuracy measurement · read-only security, privacy or architecture review.

**Never split coupled state transitions across agents.** One owner controls an
end-to-end state machine and its tests.

### 6.4 Required sub-agent prompt

Every delegated task must state: task ID from `TASKS.md` · milestone and goal ·
user-visible outcome or measurable result · **exact allowed files or subsystem** ·
files/subsystems to avoid · relevant `spec.MD` constraints · known dependencies and
risks · tests or measurements to run · expected deliverable · **whether edits are
allowed or the task is read-only** · required report format.

Every sub-agent must report: what it changed or found · files touched · tests and
measurements run with results · assumptions and deviations · remaining risks · impact on
the roadmap or other workstreams.

### 6.5 Parallel work and conflict rules

Before parallel edits, the lead creates a **file-ownership map**. High-conflict files —
central configuration, dependency manifests, entry points, shared state models, roadmap
files, public API contracts — normally have **one** editing owner. Others may review or
propose a patch in prose; the owner integrates.

Agents must not: undo or overwrite changes they did not create · reformat unrelated
files · broaden scope without lead approval · commit, merge, or declare the task
complete unless explicitly assigned that responsibility.

After every integration step the lead inspects project-scoped git status and the actual
diff (subject to §1 Git scope). With no valid project repo, inspect changed files directly.

### 6.6 Dependency order

Contracts, data shapes, acceptance metrics → pure algorithms and state machines →
hardware/model/OS adapters → UI and end-to-end wiring → tests, performance measurement,
scenario QA → documentation and final integration review.

**Parallelize only work that does not depend on an unfinished contract or shared
implementation decision.**

---

## 7. Quality assurance

**Passing compilation or unit tests is necessary but not sufficient for eye-control
behavior.**

### 7.1 Technical QA

Run the repository's existing commands. **Do not invent commands before inspecting the
project configuration.** Verify, as applicable:

- Focused unit and integration tests pass
- Type checking, linting and formatting pass
- **Camera/model resources are released on shutdown and on error**
- Threads, async tasks, timers and event listeners terminate cleanly
- Numerical code handles missing values, invalid calibration, low confidence, edge coords
- **OS input can be replaced by a fake/simulation adapter in tests**
- New dependencies are justified and compatible with the target platform
- No unrelated files or user changes were modified

### 7.2 Scenario QA — mandatory for user-facing or safety-sensitive changes

Verify scenarios, not only functions:

normal happy path · no face detected · face lost then recovered · one or both eyes
occluded · low-confidence or noisy gaze · calibration incomplete, invalid or stale ·
natural blink vs. intentional gesture · gesture cooldown and repeated input · pause and
emergency recovery · **camera unavailable or disconnected** · shutdown during an active
interaction · screen edges and multi-resolution · **real OS input disabled in automated
tests**.

For M2 and later, report measurable results where the repo provides the harness or data:
calibration error, validation error, jitter, latency, FPS, false activations, recovery time.

### 7.3 Test quality

Tests assert behavior and safety properties, not that a function returned a value.

For stateful interactions, test transitions, **forbidden** transitions, cooldowns, reset
behavior and recovery after loss or error. **A test must fail if low-confidence input can
trigger an action, or if a stale gesture can be replayed.**

Prefer deterministic prerecorded landmarks/features or synthetic fixtures. **Do not
require a live camera for the normal automated suite** unless the project defines a
hardware test tier.

### 7.4 Review after sub-agents

Before accepting delegated work, the lead answers:

Which files changed, and why? · Did the agent stay within scope and ownership? · Does it
match the milestone and DoD? · Are confidence, safety and recovery paths correct? · Are
tests capable of detecting wrong wiring or unsafe fallback? · Were privacy or
accessibility assumptions introduced? · Are there duplicate pipelines, hidden global
state, unexplained thresholds or platform coupling? · What was manually inspected, and
what behavior was actually exercised?

**If any answer is uncertain, the work is not verified.**

### 7.5 Mandatory Opus QA workflow

Claude Code is the lead and owns implementation and corrections. A **separate** Claude
Opus session or sub-agent performs read-only QA. **A reread by the implementation agent
does not satisfy this requirement.**

**Model and effort.** Use Claude Opus with Extra High Effort for every required review
where the installed runtime supports it. Inspect the available model and effort controls;
**do not invent model IDs, effort values, flags or commands.** If Extra High Effort is
unavailable, use the highest supported Opus effort and report the actual model, actual
effort and the limitation. **Do not silently substitute another model.** If Opus or a
separate review context is unavailable, **record the review as blocked, never as passed.**

**Codex migration (already authorized — do not ask again).** Disable the legacy Codex
review gate in the installed plugin's supported configuration. Remove or update only the
corresponding legacy triggers and instructions; preserve unrelated hooks, settings and
user changes. If the only supported switch has wider scope, report that scope and use the
narrowest supported change. Do not touch Codex authentication, billing or model settings
to repair a workflow being retired. Cancel obsolete review jobs through supported
mechanisms and verify the old trigger is inactive. **Do not keep invoking failed Codex
reviews or repeat waiting messages.**

**Editing `CLAUDE.md` does not disable a runtime hook or configure a model.** Report
separately: what instruction changes were made, what runtime configuration was verified,
what remains unavailable. If configuration cannot be changed with available access,
report the blocker **once** and continue independent authorized work.

**Checkpoints:**

1. **Before implementation** — send the plan to a separate Opus reviewer with task
   requirements, spec excerpts, affected components, assumptions, proposed tests and
   acceptance criteria. Resolve material findings before implementing. Repeat when
   architecture, scope or safety assumptions materially change. A short plan suffices
   for a trivial change.
2. **After each coherent unit of work** — run relevant checks, then request review of the
   actual scoped changes and evidence before starting dependent work. A unit is a bounded
   task, bug fix or integrated sub-agent result — **not every file write.**
3. **After corrections** — the lead evaluates findings and implements justified fixes,
   then sends corrected changes and updated checks. **Earlier approval does not cover
   later edits.**
4. **Before completion** — final review of scoped changes, acceptance criteria, test
   evidence and safety scenarios. Record the result in `TASKS.md`.

**Review inputs.** Give the reviewer: task ID, milestone, requested outcome, relevant
source-of-truth sections, exact scope, plan or implementation, changed files, check
results, known limitations, unresolved findings. **The reviewer inspects actual code and
tests, not only the implementer's summary.** Request actionable findings with locations
and reasoning, and a clear split between **acceptance blockers** and optional improvements.

Resolve the project Git root per §1. **Never review an unrelated user-home repository.**
Share only the minimum necessary code, documentation and non-sensitive evidence.
**§4.3 applies:** no raw video, face images, biometric samples, calibration data,
credentials or unrelated user files.

**The reviewer is read-only:** no implementation edits, dependency changes, roadmap
rewrites, commits or real OS input. The lead owns corrections. Automated checks use fake
OS-input adapters. A separate review context reduces reliance on the implementer's
narrative — it is **not** a guarantee of correctness.

**Findings and limits.** Evaluate findings against actual code and approved requirements.
Fix substantiated defects; dispute others with evidence and request re-evaluation.
**Do not expand product scope to satisfy the reviewer.**
**At most three correction-and-review cycles per unit.** If a blocker remains, findings
repeat without progress, or a product decision is needed — report it and stop dependent
implementation. Do not loop failed reviews or re-request a granted approval.

Record in `TASKS.md`: review scope, reviewer session/agent id when exposed, **actual
model, actual effort (or explicitly unverified)**, outcome, findings, corrections, checks,
limitations. **Never claim Opus or Extra High Effort without runtime evidence.
Never mark DONE while acceptance-blocking findings remain or required review is missing.**

Replacing the reviewer does not weaken regression gates, privacy rules, acceptance
criteria, required live validation, or explicit user approval for real OS input and
commits. **Opus review cannot certify hardware tests, usability, gaze accuracy or safety
that were not actually measured.**

---

## 8. Working practices

- Prefer small, reviewable changes connected to one outcome.
- Read before editing; follow established conventions.
- **Preserve user changes. Avoid destructive Git operations.**
- **Never silently replace an existing implementation or dependency choice.**
- Add dependencies only when they materially improve the requested outcome; explain the
  tradeoff.
- Put configurable thresholds and tunable parameters in **one discoverable place.**
- Document units for time, distance, angles, screen coordinates and confidence.
- **Use monotonic time** for gesture durations and cooldowns.
- Keep logs useful and free of raw biometric/video data.
- **Never claim behavior was tested with real hardware or users unless it actually was.**

---

## 9. Completion checklist

Before the final response:

1. Re-read the requested outcome, the `TASKS.md` entry, the `spec.MD` milestone and the
   matching `TECHNICAL_SPEC.md` sections
2. Review project-scoped git status and the full diff **after** resolving a valid project
   Git root; otherwise inspect the scoped files directly (§1)
3. Confirm only intended files changed
4. Verify the §5 OOP policy: responsibility boundaries, thin entry points, reuse, size
   exceptions, architecture review evidence
5. Run the relevant tests and static checks available in the repository
6. Verify applicable safety, confidence, recovery, privacy and accessibility scenarios
7. **Confirm no real cursor/click action can occur during automated testing**
8. Record what could not be tested, especially hardware-dependent behavior
9. Complete the §7.5 Opus checkpoints; resolve blockers or **report QA as blocked**
10. Update status and completion evidence in `TASKS.md` without removing the original
    acceptance criteria, including the Opus outcome and remaining limitations

**The final response must report:** task IDs · milestone/outcome · files changed · what
was manually reviewed · scenarios verified · tests and checks run with results ·
measured performance/accuracy where applicable · **Opus QA outcome, actual reviewer model
and effort, corrections, disputed findings, any review blocker** · local/commit status ·
remaining risks, hardware validation and follow-up work.

**If sub-agents were used, also report:** each agent's scope · files each touched · what
the lead independently reviewed and verified · corrections after integration QA · final
orchestration QA result.

**Never present delegated work as complete before lead verification.**
