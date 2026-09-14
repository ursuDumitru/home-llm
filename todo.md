# TODO List

## LLM Evaluations

- Standard academic benchmarks for comparison with other models.
- A private HomeLLM Eval for your actual goals.
- Blind comparisons against selected reference models.
- Speed, latency, memory, and model-size measurements.
- Regression testing after every training or architectural chang

## SSH - unrelated to agents, for user only

- you could have setup an ssh connection to the WSL instance and add the project as a remote connection  in chatGPT app :D - maybe look forward to this in the future

## Local Inference CLI

- Run existing open-weight language models locally; do not train a foundation model from scratch.
- Use Python for the application and Ollama as the initial inference backend.
- Run Ollama inside Ubuntu/WSL with cloud features disabled and its API bound to the local loopback address.
- Verify GPU inference first with a small Qwen 3.5 model before downloading larger models.
- Build an interactive streaming CLI with clear prompt, answer, error, and statistics output.
- Keep conversation logic, model backends, and console rendering in separate modules so future models and interfaces can reuse them.
- Keep downloaded models, checkpoints, conversations, and generated artifacts outside Git.