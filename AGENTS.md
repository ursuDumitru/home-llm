# AGENTS.md

## Purpose

This repository is a learning-focused workspace for developing, training,
fine-tuning, evaluating, and running local language models.

The repository may eventually contain multiple model experiments. Code should
therefore remain modular and reusable without introducing unnecessary
abstractions too early.

The repository owner is learning AI/ML development. Generated code must favor
clarity, explicit behavior, and incremental progress.

## Active Development Roadmap

This section defines the development order as of 2026-09-14. Follow it unless
the owner changes priorities. It defines a finite local CLI release.
Follow the collaboration rules below when implementing
requested changes; the roadmap does not authorize unrelated work.

### Completion status (updated 2026-09-18)

Record completion using existing tests and owner-reported runs; do not reopen
completed milestones merely to repeat testing. DONE implementation does not
mean every live acceptance step has been observed.

- [x] DONE: Phase 0 testing checkpoint, including existing local inference,
  GPU-monitoring, evaluation, cancellation, and bounded handoff evidence.
- [x] DONE: Phase 1 JSON registry, validated profiles, and configuration-only
  model additions through the existing backend.
- [x] DONE: Phase 2 model commands, runtime status, validated selection,
  enable/disable persistence, and state preservation on failure.
- [x] DONE: Phase 3 explicit session saving, listing, loading, saved settings,
  unsaved-work guards, and new conversations.
- [x] DONE: Phase 4 context-management implementation: full transcript retained,
  persisted clearing boundary, estimated token budget, complete recent-turn
  selection, omission notices, and oversized-prompt rejection.
- [x] DONE: Makefile lint, format, format-check, and all/named unit-test commands.
- [x] DONE: README usage guide and final acceptance instructions.
- [x] DONE: Fast verification after context changes: Ruff checks passed;
  99 tests passed, with one opt-in live integration test skipped.
- [x] DONE: Latest live run completed two 4B turns, saved, restarted, listed
  sessions, loaded four messages with the saved 64-token output limit, then
  cleared request context and saved the same session again.
- [x] DONE: Latest live run demonstrated useful errors for missing `/load`
  arguments, unknown `/list`, switching with retained history, and invalid
  `/new qwen-9b` syntax. Idle Ctrl+C exited after saving.
- [x] DONE: Subsequent live run loaded the saved chat, used `/new`, and selected
  `qwen-9b` with its registry settings (4096 context, 512 output tokens).
  The repeated `/clear` did not change an already-saved boundary, so `/new`
  correctly required no additional save in this run.
- [ ] PENDING: Complete a generated 9B turn after this successful switch.
  The latest log stops at model selection. Reuse earlier live resumption and
  automated context/resumption evidence; this log does not show a resumed turn.
- [ ] PENDING: First-release sign-off after that remaining live workflow.

Keep later work deferred, not DONE: improved terminal input/history, current
date/runtime information, RAG, expanded evaluations, thinking profiles, GUI,
fine-tuning, and training. No application changes were made by this status update.

### Current goal and baseline

Build a practical local chat application using inference with existing models.
The first release must support selecting configured models, enabling or
disabling them, and saving and resuming conversations. Continue using Python,
Ollama, argparse, and the shared Rich console renderer.

The working baseline already includes streaming chat, conversation state,
application tests, GPU monitoring, performance comparison, a six-case quality
evaluation, and saved evaluation artifacts. Qwen 3.5 4B is the provisional
default because of its measured speed and memory cost; 9B remains available.
The small quality evaluation does not establish a general quality winner.

### Phase 0: close the current testing checkpoint — DONE

Use exactly these six acceptance scenarios to close this phase. Reuse existing
tests and reported results when the relevant code has not changed. These are
acceptance scenarios, not a requirement to create six new test files or to
reach a particular test count.

1. DONE: Static checks and existing unit tests pass. The live integration test may
   remain skipped in ordinary unit-test runs.
2. DONE: One local model completes a streamed CLI turn; a second turn receives the
   prior conversation. Use existing integration and conversation tests as
   evidence; do not repeat the two-model performance benchmark.
3. DONE: A failed or cancelled generation leaves no incomplete turn in conversation
   history and allows another prompt. Add a focused fake-backend cancellation
   check only if current tests do not cover it.
4. DONE: Optional GPU monitoring returns samples on this machine. A measurement
   failure is reported visibly without converting a successful inference test
   into a failure. Existing monitoring output counts as evidence.
5. DONE: One saved evaluation artifact parses as JSON and includes settings, model
   mapping, all six case prompts and answers, scores, manual ratings, timings,
   and GPU samples or explicit measurement errors. The 2026-09-14 artifact is
   existing evidence for the successful path; do not regenerate it merely to
   repeat verification.
6. DONE: Model handoff waits for the previous model to disappear from Ollama's
   running-model list and fails with a bounded, actionable timeout if it does
   not. Use mocked states for timeout coverage. GPU memory samples describe
   the whole device; an empty model list does not prove all VRAM is freed.

Do one gap review against this list, address uncovered behavior with small
changes, and move to Phase 1 when it passes. Fix actual failures, but do not
add another evaluation milestone as a condition for CLI development. A model's
wrong arithmetic answer is an evaluation finding, not an application blocker.

Defer larger question sets, versioned evaluation-suite files, academic
benchmarks, blind-rating refinements, and thinking-mode comparisons until the
first CLI release is usable or a specific feature requires them.

### Phase 1: model registry and validated configuration — DONE

Start with `config/models.json`, a small, versioned JSON registry. Use Python's
standard JSON library; no database service or new configuration dependency is
needed. Commit only non-sensitive configuration. Keep machine-specific or
private overrides outside Git if they are introduced.

Define a schema version, default model ID, and model profiles. Each profile
has a stable application ID, display name, backend ID (initially `ollama`),
runtime model name, enabled flag, and generation settings such as context
length, output limit, temperature, and optional seed. Keep the local Ollama
endpoint in shared runtime configuration. Preserve the existing loopback and
cloud-disabled constraints; registry entries must not bypass them.

Distinguish three states: enabled means selectable in this app; installed
means available in Ollama; loaded means currently resident in runtime memory.
An enabled flag neither downloads nor loads a model. Disabling a profile does
not delete weights or saved chats. Query installation/loading state when
needed instead of storing stale copies as registry truth.

Adding an installed model supported by an existing backend must require only
a JSON entry, with no Python edits or model-name branches. Adding a different
runtime/API requires one adapter implementing `ModelBackend` and registration
of that backend type. Configuration cannot implement a new protocol. Validate
unsupported capabilities and settings explicitly; the first release is text
chat. Vision, tools, and thinking output require separate feature work.

Acceptance: load a valid registry; reject malformed JSON, duplicate IDs,
unknown schema/backend, invalid numeric settings, and a missing or disabled
default; select an enabled profile; and prove with a fake backend that a newly
added profile works without changing source. Missing model weights must produce
an actionable message, not an automatic download.

### Phase 2: model selection in the interactive CLI — implementation DONE

Add `/models` to show profiles and enabled/installed/loaded status, `/model`
to show the current profile, and `/model <id>` to select an enabled profile.
Support a registry path argument and preserve startup `--model` selection;
document whether it accepts profile IDs, raw Ollama names, or both. Prefer
profile IDs with a clear migration message for any changed behavior.

Provide `/enable <id>` and `/disable <id>` backed by validated registry writes.
Reject disabling the active or default profile until a replacement is selected.
Persist updates atomically. Keep command parsing, backend creation,
conversation management, and Rich rendering separate. Do not build a plugin
framework or duplicate backend logic inside commands.

Initially allow model switching only in an empty conversation. Keep the current
model and messages unchanged when selection or loading fails. Once sessions
exist, offer save followed by `/new` before switching; do not silently reuse
history or a system prompt under a different model profile.

Acceptance: list and select a profile, reject disabled/unknown/uninstalled
choices, persist enable/disable state across restart, preserve state on failure,
and perform one real chat with a profile added solely through JSON.

### Phase 3: save and resume chat sessions — DONE

Use one versioned JSON file per session under `conversations/`, already ignored
by Git. Start with a single-process local store and scan files for listings;
SQLite is deferred until search, scale, or concurrent writes justify it.

Store a generated session ID, optional title, UTC creation/update timestamps,
model profile ID, runtime model name, an effective settings snapshot, system
prompt, and ordered role/content messages. Store text history, not GPU state,
model weights, tokenizer internals, or runtime attention caches. Resuming means
reconstructing the messages and submitting them on the next model request;
it does not guarantee identical future answers.

Introduce `/save`, `/sessions`, `/load <id>`, and `/new`. Make saving explicit
for the first release and warn before replacing unsaved conversation state.
Save complete turns only. Use generated IDs rather than titles as filenames,
validate loaded documents, and use a temporary file plus atomic replacement
in the same directory. Preserve the last valid file on a failed save and reject
unsupported future schema versions rather than guessing their meaning.

Do not silently substitute models when resuming. If the referenced profile is
disabled/missing or its weights are unavailable, explain the required action.
Show differences between current settings and the saved snapshot and use the
validated saved snapshot for resumption. Support one writer per session file;
do not claim concurrent-write safety merely because writes are atomic.

Acceptance: save/load round-trip preserves messages and settings; a resumed
turn receives the saved history; malformed or unsafe IDs/files fail clearly;
failed saves preserve the previous file; and `/new` or `/load` cannot silently
discard unsaved work. Use temporary directories and fake backends for tests.

### Phase 4: manage context and finish the first CLI release — implementation DONE

Context features and automated checks are DONE. Final live acceptance remains
PENDING as detailed in the completion status above; do not label the entire
release complete yet.

Separate the full saved transcript from the messages included in each model
request. The model's context budget must accommodate its prompt template,
system message, selected history, current prompt, and generated output.

Add `/context` to show the configured budget and history selection, and `/clear`
to start fresh request context while retaining the saved transcript. Persist
the context boundary so loading a chat does not restore deliberately excluded
messages. Keep `/new` as the operation that creates a separate conversation.

Use a deterministic history-selection policy that preserves the system prompt,
current user prompt, and complete recent turns. If older turns are omitted,
notify the user; never claim the model still sees them. Label approximate token
counts as estimates, reserve output space and a safety margin, and handle an
oversized current prompt explicitly. Exact token counting may require backend
support; do not present a character heuristic as a hard context guarantee.
Automatic summarization and persistent model memory are deferred.

Acceptance: history selection preserves turn boundaries and required messages,
reports omissions, retains the full transcript on disk, survives save/load, and
handles an oversized prompt without silently claiming full context retention.
Finish with one manual workflow: choose a configured model, chat for two turns,
save, restart, resume, clear request context, and start a new chat on another
model. Verify useful errors for an unavailable Ollama service and cancellation.

### Release discipline and later work

Each phase may take several small implementation steps. Add focused tests for
new behavior and regressions, run existing fast checks once after relevant
changes, and repeat live GPU checks only when model/backend behavior warrants
them. No coverage-percentage target, large model downloads, or repeated blind
comparisons are required to ship this local CLI release.

After these four feature phases, let practical CLI use determine the next
priority. Candidates include better terminal input, retrieval over local
documents, improved evaluations, thinking profiles, and a graphical interface.
RAG, fine-tuning, and training remain distinct projects and are not prerequisites
for this release. Preserve shared application APIs so a future interface can
reuse configuration, conversations, storage, and model adapters.

## Collaboration Rules

### Direct code changes

Agents may directly create and edit repository files, apply patches, and run
formatters, linters, tests, and relevant local verification within the scope of
the owner's requested task. Keep changes focused and incremental, preserve
unrelated owner changes, and avoid destructive operations without clear
authorization.

The owner does not need to copy, paste, or apply agent-proposed changes manually.
Honor requests for explanation-only answers or manual instructions instead of
editing when that is what the owner asks for.

Unless explicitly requested otherwise, agents must not:

- Install or remove dependencies.
- Download models or datasets.
- Run commands that modify Git state.
- Commit, push, or create branches.

Agents may inspect existing files and report their findings without making
changes when the request is for review, diagnosis, or explanation.

### Incremental development

Work in small, verifiable steps.

For each step:

1. State the goal of the step.
2. Explain why the step is needed.
3. Identify every file that must be created or changed.
4. Implement the requested changes directly unless manual instructions were requested.
5. Run appropriate verification and report the commands used.
6. Explain the actual results and any unverified behavior or limitations.
7. Ask the owner only when a required choice, permission, or unavailable manual
   check prevents further progress within the requested scope.

Do not generate a large project skeleton unless the owner specifically asks for
one.

When several implementations are possible, recommend one simple default and
briefly explain the important tradeoffs.

### Implementation handoff

After every implemented step, provide a self-contained description answering:

- What was done: the behavior changed and the files or components affected.
- Why it was done: the problem solved and how it advances the current goal.
- How it was verified: checks run, their results, and remaining limitations.

Do not report only that the work is done or that tests passed. Keep the
explanation concise, but make the purpose of the change explicit.

## Teaching Style

Assume the owner knows general programming concepts but is still learning
AI/ML engineering.

Explanations must:

- Use direct technical language.
- Avoid analogies.
- Introduce unfamiliar AI/ML terms before relying on them.
- Explain why a component is required, not only what it does.
- Distinguish required components from optional improvements.
- Include tensor shapes, data types, units, and device placement when relevant.
- Point out important resource costs such as VRAM, RAM, storage, and execution
  time.
- Clearly label assumptions and uncertainty.

For a non-obvious function or processing step, explain:

- Input: what data enters, including type and shape where relevant.
- Processing: what the code does to that data.
- Output: what it returns, including type and shape where relevant.
- Purpose: why the operation is needed by the model or pipeline.

Do not explain every line when the behavior is already clear. Focus on concepts,
data flow, design decisions, and likely sources of mistakes.

## Code Presentation

For direct edits, summarize the changes, link to the affected files, and report
verification results. Do not paste entire files unless the owner requests it.

When providing code for manual application, every code block must state:

- The target file path.
- Whether it creates a new file or changes an existing file.
- Whether the block is the complete file or only a replacement section.

All implemented or proposed code must:

- Be complete and executable.
- Contain no placeholder ellipses such as `...`.
- Use descriptive names.
- Prefer straightforward control flow over compact or clever code.
- Keep functions focused on one responsibility.
- Include comments where they clarify purpose, constraints, shapes, or
  non-obvious behavior.
- Avoid comments that merely repeat the syntax.
- Include type hints where they improve understanding.
- Use docstrings for public or non-obvious functions.

When the owner requests copy-paste instructions, provide the complete file for
small files. For larger existing files, provide an exact replacement section
with enough surrounding context to locate it.

Explain how the changed code connects to the rest of the project.

## AI/ML Scope

Do not treat the following as interchangeable:

- Running inference with an existing model.
- Prompting a model.
- Retrieval-augmented generation (RAG).
- Fine-tuning an existing model.
- Training a small model from scratch for learning.
- Pretraining a general-purpose foundation model.

Before proposing an implementation, state which of these activities it covers.

If the requested goal is ambiguous, explain the distinction and identify the
assumption being used.

Avoid implying that a small fine-tuning or educational training experiment
creates a general-purpose foundation model.

## Hardware Constraints

Development takes place inside Ubuntu 26.04 on WSL, hosted by Windows.

Available hardware:

- CPU: AMD Ryzen 7 9700X
- GPU: NVIDIA RTX 5070 12GB VRAM
- System RAM: 32 GB

Model and configuration recommendations must be realistic for this machine.
Account for:

- Model parameter count.
- Weight precision and quantization.
- GPU VRAM use.
- Activations and attention memory.
- Optimizer and gradient memory during training.
- CPU offloading costs.
- Dataset size.
- Checkpoint size.
- Expected training or inference time.

Prefer small models and small datasets for initial experiments. Use tiny smoke
test configurations before expensive runs.

When full training is unrealistic on this hardware, say so directly and propose
a smaller educational experiment, parameter-efficient fine-tuning, inference,
or another practical alternative.

GPU and CUDA compatibility changes over time. Verify current compatibility
using primary documentation before recommending exact versions of PyTorch,
CUDA, GPU drivers, quantization libraries, or attention libraries.

## WSL and Platform Rules

Commands intended for the project should use the Ubuntu/WSL shell unless
explicitly labeled as Windows PowerShell commands.

For every command block, state which shell should run it:

- Ubuntu/WSL shell
- Windows PowerShell

Use Linux-style paths inside project code and WSL commands.

Do not mix Windows and WSL installation instructions without explaining where
each command runs and why it is needed.

Prefer cross-platform Python code, but optimize the development instructions for
Ubuntu under WSL.

## Project Structure

The repository should be able to support multiple experiments and models.

Prefer separation between concerns such as:

- Configuration.
- Data loading and preprocessing.
- Tokenization.
- Model definitions or model adapters.
- Training.
- Inference and generation.
- Evaluation.
- Checkpoint and artifact handling.
- Shared utilities.
- Tests.
- Individual experiments.

Do not create all of these modules before they are needed. Start with the
smallest working vertical slice and extract reusable components when a second
use case demonstrates the need.

Shared code must not depend on one experiment's directory or configuration.

Experiment-specific settings should live in configuration files rather than
being scattered as hard-coded values throughout the code.

## Reproducibility

Experiments should be reproducible where reasonably possible.

Code should explicitly manage:

- Random seeds.
- Model and dataset identifiers.
- Dependency versions.
- Training configuration.
- Evaluation configuration.
- Checkpoint locations.
- Relevant hardware and precision settings.

Do not claim exact reproducibility when GPU operations or third-party libraries
can be nondeterministic. Explain such limitations when they matter.

Separate fast smoke-test settings from real experiment settings.

## Dependencies

Keep the dependency set small.

Before adding a dependency:

1. Explain what it provides.
2. Explain why the standard library or an existing dependency is insufficient.
3. Check compatibility with Python, WSL, the GPU, and other ML libraries.
4. Prefer maintained libraries with clear documentation.
5. Pin or constrain versions when compatibility requires it.

Do not add multiple tools that solve the same problem without a clear reason.

Do not silently change the Python, CUDA, PyTorch, or model-library version.

## Testing and Verification

Each implementation step must include an appropriate verification method.

Prefer this order:

1. Syntax or import check.
2. Small unit test for isolated logic.
3. Tiny CPU smoke test when possible.
4. Tiny GPU smoke test when GPU behavior matters.
5. Small end-to-end test.
6. Full experiment only after earlier checks pass.

Tests must not require downloading a large model or dataset unless that
requirement is clearly stated in advance.

When a test fails, explain the observed evidence and likely cause before
proposing a fix.

Do not report that code works unless it has actually been run, or clearly state
that the result is unverified and explain why it could not be run.

## Model and Dataset Safety

Before recommending or using a model or dataset, check:

- License and usage restrictions.
- Required authentication or access approval.
- Download size.
- Runtime memory requirements.
- Whether custom remote code is required.
- Whether the source is trustworthy.
- Whether personal or sensitive data is present.

Do not enable remote custom code without explaining the security implications.

Never place secrets, API tokens, private datasets, model weights, checkpoints,
or generated training artifacts in Git.

Large generated files should be stored outside the source tree or excluded
through `.gitignore`.

## Error Handling

Errors should be explicit and actionable.

Validation messages should state:

- What value or condition was invalid.
- What was expected.
- How the owner can correct it.

Do not catch broad exceptions unless they are re-raised with useful context or
handled at a clear application boundary.

Do not hide warnings related to numerical stability, incompatible model
settings, missing GPU support, or insufficient memory.

## Performance

Correctness and clarity come before optimization.

Before optimizing:

1. Establish a working baseline.
2. Measure the relevant bottleneck.
3. Explain the proposed optimization.
4. Describe its effect on readability, precision, and memory.
5. Provide a way to compare the result against the baseline.

Do not introduce distributed training, compilation, custom GPU kernels, or
complex caching until measurements justify them.

## Git and Generated Artifacts

Proposed Git changes should be small and focused.

Do not include the following in version control:

- Model weights and checkpoints.
- Downloaded datasets.
- Secrets or local credentials.
- Virtual environments.
- Python caches.
- Training logs.
- Generated evaluation outputs.
- Large temporary artifacts.

When one of these artifacts is first introduced, propose the corresponding
`.gitignore` entry.

## Decision Priority

When requirements compete, use this priority order:

1. Correctness.
2. Safety and data privacy.
3. Ability to run on the available hardware.
4. Clarity for learning.
5. Reproducibility.
6. Simplicity.
7. Reusability.
8. Performance.

If a recommendation compromises a higher-priority concern, explain the
tradeoff before providing the implementation.
