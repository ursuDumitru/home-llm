# HomeLLM

A local text-chat CLI for inference with existing models. Python coordinates
Ollama, a JSON model registry, explicit chat saving, and a shared Rich renderer.
This application does not train or fine-tune models.

## Run the existing setup

Use the Ubuntu/WSL shell from `/home/ursu/projects/home-llm`. These commands assume
the existing `.venv` is ready, Ollama is running locally, and the selected model
is already installed. They do not install dependencies or download models.

```bash
PYTHONPATH=src .venv/bin/python -m home_llm.cli
```

Select a profile or override its starting settings in the Ubuntu/WSL shell:

```bash
PYTHONPATH=src .venv/bin/python -m home_llm.cli --model qwen-4b --max-output-tokens 64
```

Startup `--model` accepts a profile ID or a unique configured runtime model name.
Interactive `/model <id>` accepts a profile ID. Startup setting overrides apply
only to the initial selection; switching models uses registry settings, and
loading a saved chat uses its saved settings.

## Everyday commands

Enter these at the application's `PROMPT >`, not in the shell:

| Command | Purpose |
| --- | --- |
| `/help` | Show available commands. |
| `/models` | List profiles and enabled, installed, and loaded status. |
| `/model` | Show the active model and effective settings. |
| `/model qwen-9b` | Switch to that profile in an empty conversation. |
| `/enable qwen-9b` | Enable the existing profile in the saved registry. |
| `/disable qwen-9b` | Disable a non-active, non-default profile; retain weights. |
| `/save My chat` | Save complete turns and optionally set the title. |
| `/save` | Update the current saved chat, preserving its ID and title. |
| `/sessions` | List saved session IDs and titles. |
| `/load <id>` | Resume a session; replace `<id>` with the full ID from `/sessions`. |
| `/context` | Show budget, eligible history, and last request selection. |
| `/clear` | Exclude previous turns from requests while retaining the transcript. |
| `/new` | Start a separate empty conversation without carrying a system prompt. |
| `/exit` | Exit the application. |

Saving is explicit, not automatic. `/new`, `/load`, and `/exit` guard unsaved
work. Save first, or append `--discard` only when you intend to lose unsaved
changes. `/clear` changes the request-context boundary; `/save` persists it.
Use `/new`, not `/clear`, before switching models after a conversation.

Press Ctrl+C during generation to cancel the incomplete turn. Previously
completed turns remain available; another prompt can be entered. At an idle
prompt, Ctrl+C exits only when there are no unsaved changes.

## Models and storage

Model profiles live in `config/models.json`. To add another already-installed,
local text-chat model supported by the Ollama adapter, copy an existing profile
and change its unique `id`, `display_name`, `model_name`, and appropriate settings.
No model-specific Python changes are required. Restart to read new profiles.
An unsupported runtime needs a new adapter; JSON alone cannot implement it.

Enabled means selectable in this app; installed means available in Ollama;
loaded is a runtime snapshot. Enabling a model neither downloads nor loads it.
Keep Ollama's cloud-disabled configuration in place. The application validates
loopback endpoints and rejects explicit cloud model identifiers; this is not
a network sandbox for the Ollama process.

Chats are stored as plaintext JSON under `conversations/`, ignored by Git.
`--sessions-dir` can choose a different directory; keep private chat files out
of version control. Saved chats include full text, effective model settings,
and the `/clear` boundary, not GPU state or runtime attention caches. Only one
writer per session is supported. Resumption does not promise identical answers.

## Context selection and limitations

Each request reserves the configured output limit and 128 template/safety
tokens. Input estimation uses three UTF-8 bytes per token plus eight framing
tokens per message. These are heuristics, not exact tokenizer counts or a hard
guarantee that the request fits the runtime context.

The system message and current prompt are retained. The largest fitting suffix
of complete recent turns is included; the CLI reports omitted older turns.
Omissions do not delete transcript content and are recalculated per request.
An oversized current prompt/system message is rejected before inference, with
instructions to shorten input or adjust limits.

`/context` distinguishes the full transcript, history excluded by `/clear`,
eligible history, and the last attempted request. After the first response,
one turn is eligible, but the last request selected zero previous turns.

The application does not currently supply the current date or live information
to the model. Use application commands for model/settings information rather
than trusting the model's self-description. Plausible but incorrect dates or
answers are model-quality findings, not proof of a context-selection failure.

## Fast development checks

Ubuntu/WSL shell, from the project directory:

```bash
make check
make format-check
make test
```

To format source and tests or run one test module, use the Ubuntu/WSL shell:

```bash
make format
make test TEST=test_context_budget
```

`make test` disables the opt-in live integration test. It does not require a
running model. A fully qualified test class or method is also accepted by `TEST`.

## First-release acceptance: one final manual workflow

The context-budget implementation's last full fast check ran 100 tests:
99 passed and the live integration test was skipped. Existing owner-reported
runs cover GPU inference, saved resumption, persisted clearing, and switching
profiles. Cancellation and failure recovery also have deterministic tests.
Those results should be reused unless relevant behavior changes.

### Recorded results (2026-09-18)

- [x] DONE: Two live 4B turns with a 64-token output limit.
- [x] DONE: Save, exit, restart, and list the saved conversation.
- [x] DONE: Load restores four messages and the saved 64-token output limit.
- [x] DONE: `/clear` retains the transcript; `/save` updates the same session.
- [x] DONE: Command errors protect state: `/load` needs an ID, `/list` is not
  supported, and model switching is blocked while the full transcript exists.
- [x] DONE: Idle Ctrl+C exits normally after the session has been saved.
- [x] DONE: Earlier live results and automated tests cover resumption and the
  persisted context boundary; context-selection tests cover omission and limits.
- [x] DONE: Subsequent live run loaded the chat, started a new conversation with
  `/new`, and selected 9B with its registry output limit restored to 512 tokens.
- [ ] PENDING: Complete a generated response on 9B after that switch. The latest
  log confirms selection, not generation. Earlier resumption evidence is reused.

Do not repeat the completed checks or benchmarks. In the current 9B chat, send
`Reply briefly: hello.`. Finish with `/save Second model check` and `/exit`.
The repeated `/clear` in the latest run did not change the already-saved boundary;
therefore `/new` correctly proceeded without another save.

`/new qwen-9b` is not supported: creating a conversation and selecting a model
are separate commands. `/clear` is not a substitute for `/new` because it retains
the transcript. The displayed `^[[A` is consistent with pressing the up-arrow
key; interactive input/history improvements remain deferred.

### Full workflow reference

The original workflow below is retained as a reference, not a requirement to
repeat the DONE items. Prompts may differ; verify conversation state rather than
requiring a particular model answer.

1. Start the CLI with `--model qwen-4b --max-output-tokens 64` using the shell
   command above. At `PROMPT >`, send `Remember this phrase: blue lantern.`
   followed by `What phrase did I ask you to remember?`.
2. Enter `/context`. Expect two completed transcript turns and one prior turn
   selected for the second request, with no budget omissions for this short chat.
3. Enter `/save Release check`, copy the generated ID, then `/exit`.
4. Restart the CLI with the default launch command. Enter `/load` followed by
   that ID. Expect the saved 64-token output limit to be restored. Ask again
   which phrase you supplied, then `/save`.
5. Enter `/clear`, then `/context`. Expect three transcript turns, three turns
   excluded by clearing, and zero eligible history turns. Enter `/save` to
   persist the boundary.
6. Enter `/new`, then `/model qwen-9b`. If the profile was deliberately disabled,
   enable it first with `/enable qwen-9b`. Send `Reply briefly: hello.`. Expect a
   new conversation on 9B with its registry settings and no prior chat history.
   Finish with `/save Second model check` and `/exit`.

A wrong model answer alone does not prove history was lost: inspect `/context`
and use the fake-backend tests as evidence of which messages were submitted.
Do not delete existing conversations or stop the Ollama service for this check.
Report command errors or unexpected state before declaring the release complete.
