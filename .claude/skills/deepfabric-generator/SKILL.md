---
name: deepfabric-generator
description: Use this when the user gives you a topic and asks you to generate a dataset.
---

# DeepFabric Dataset Generator

## Prerequisites

If the user wants **mock files**, they will need the **Spin server** running before dataset generation. The Spin server provides a virtual filesystem that handles the tool calls (read_file, write_file, list_files) during generation.

**Setup for mock files — ask the user which method they want to use:**

Use `AskUserQuestion` to ask the user how they want to run the Spin server:
- **Docker** — description: "Run via Docker Compose — no Rust toolchain needed. Requires Docker installed and running."
- **Local Spin** — description: "Build and run Spin locally — requires Spin CLI (v2.0+) and Rust with the wasm32-wasip1 target."

**If the user chose Docker:**

1. Check that Docker is available: `docker --version` and `docker info`. If not installed or the daemon isn't running, tell the user to install/start Docker and re-run.
2. Locate the deepfabric tools-sdk directory. Check common locations: `~/projects/deepfabric/tools-sdk`, `../deepfabric/tools-sdk`, or ask the user for the path.
3. Start the Spin server: `cd <tools-sdk-path> && docker compose up -d`
4. The server runs on port **3030** → set `spin_endpoint` to `http://localhost:3030`
5. Verify it's running: `curl -sf http://localhost:3030/healthz || curl -sf http://localhost:3030/`
6. To stop it later: `cd <tools-sdk-path> && docker compose down`

**If the user chose Local Spin:**

1. **Spin CLI** (v2.0+) — check with `spin --version`. If not installed, tell the user to install it from https://developer.fermyon.com/spin and re-run.
2. **Rust with wasm target** — check with `rustup target list --installed | grep wasm32-wasip1`. If not installed, tell the user to install Rust and the wasm target, then re-run.
3. Locate the deepfabric tools-sdk directory. Check common locations: `~/projects/deepfabric/tools-sdk`, `../deepfabric/tools-sdk`, or ask the user for the path.
4. Start the Spin server in the background and capture its PID:
   ```
   cd <tools-sdk-path> && spin build && spin up &
   SPIN_PID=$!
   ```
   Save the PID — you will need it to stop the server after generation is complete.
5. The server runs on port **3000** → set `spin_endpoint` to `http://localhost:3000`

If the user does NOT want mock files, none of this is needed.

## Step 1: Gather Requirements

First, check if DeepFabric is installed by running `pip3 show deepfabric`. If it is not installed, install it with `pip3 install deepfabric`.

Next, collect the following information from the user before proceeding:

- **Topic** (required) — Ask the user exactly what the dataset topic should be about if they don't mention it. Only follow up with further questions after the topic is confirmed.

Once the topic is confirmed, briefly explain how DeepFabric uses a **knowledge graph** to generate diverse data, using a simple example based on their topic. For instance, if their topic is "Python programming":

> DeepFabric first builds a knowledge graph from your topic to make sure the dataset covers diverse subtopics. Think of it like a tree — your topic branches out into subtopics, and each subtopic branches further:
>
> ```
> Python Programming
> ├── Data Structures → Lists, Dicts, Sets
> ├── Control Flow    → Loops, Conditionals, Exceptions
> └── Functions       → Arguments, Decorators, Closures
> ```
>
> **Depth** = how many levels deep it goes (the example above has depth 2).
> **Degree** = how many branches at each level (the example above has degree 3).
>
> More depth/degree = more unique subtopics = more diverse (but larger) dataset.

Then use the `AskUserQuestion` tool to ask the user about depth and degree together:
- **Depth** — options: "2 (shallow)", "3 (balanced) (Recommended)", "4 (deep)"
- **Degree** — options: "2 (narrow)", "3 (balanced) (Recommended)", "4 (wide)"

Note: The total number of topics is approximately degree^depth. For example, depth=3 + degree=3 = ~27 topics; depth=3 + degree=2 = ~8 topics; depth=4 + degree=4 = ~256 topics.

After depth and degree are answered, use `AskUserQuestion` to ask the API provider and whether they want to customize further (in a single tool call with both questions):
- **API provider** — options: "Gemini (Recommended)", "Anthropic" — this is always asked, never defaulted
- **Customization** — options: "Pick the best defaults for me (Recommended)", "I want to customize further"
  - Description for defaults: Briefly list the defaults you'd pick for THIS topic (see below). Example: "I'll use: instruction-following, chain-of-thought, 50 samples, with mock files (since this is a coding topic)."
  - Description for customize: "I'll walk you through each remaining option: conversation type, reasoning style, number of samples, and mock files."

### If the user chose "Pick the best defaults for me"

**The defaults must be chosen based on the user's topic — they are NOT static.** Analyze the topic and pick the best fit for each setting. Here are guidelines:

- **Conversation type**:
  - Coding/building topics (e.g. "making an ML library", "building a REST API") → **Instruction-following** (users want step-by-step guidance)
  - Knowledge/theory topics (e.g. "history of AI", "linear algebra concepts") → **Q&A**
  - Casual/open-ended topics (e.g. "travel planning", "creative writing") → **Chat**

- **Reasoning style**:
  - Coding/technical/math topics → **Chain-of-thought** (step-by-step reasoning helps)
  - Simple factual Q&A → **Direct**

- **Number of samples**: Default to the number of generated topics (≈ degree^depth) so each topic gets at least one sample. For richer coverage, recommend 2× the topic count. Tell the user the recommended number and why.

- **Mock files**:
  - Topics involving code, projects, or files (e.g. "building an ML library", "web development") → **Yes** (the dataset will include realistic project files the AI can read/write)
  - Non-coding topics (e.g. "cooking recipes", "history") → **No**

Tell the user what defaults you picked and **why** in a brief summary so they know what's being used and can object if something feels off. Then skip to Step 2.

### If the user chose "I want to customize further"

Use `AskUserQuestion` to collect the remaining details (use up to two tool calls if needed to stay within the 4-question limit per call):
- **Conversation type** — options: "Q&A", "Instruction-following", "Chat"
- **Reasoning style** — options: "Chain-of-thought (Recommended)", "Direct", "Step-by-step"
- **Number of samples** — options: "Match topic count (~N) (Recommended)", "2× topic count (~M)", "Custom" — where N = degree^depth and M = 2 × degree^depth. Describe that each sample is one training conversation and more samples = better coverage but longer generation time.
- **Mock files** — options: "No mock files (Recommended)", "Yes, include mock files" — describe that mock files simulate a project directory the AI can read/write during conversations

Do NOT dump all these questions as plain text. Always use the `AskUserQuestion` tool so the user gets an interactive selection UI.

## Step 2: Validate API Key

Before creating any files, verify the API key is set:

- If Gemini: run `[ -n "$GEMINI_API_KEY" ] && echo "GEMINI_API_KEY is set" || echo "GEMINI_API_KEY is NOT set"`
- If Anthropic: run `[ -n "$ANTHROPIC_API_KEY" ] && echo "ANTHROPIC_API_KEY is set" || echo "ANTHROPIC_API_KEY is NOT set"`

If the key is missing, tell the user to set it:
```
export GEMINI_API_KEY="your-key-here"
```
or
```
export ANTHROPIC_API_KEY="your-key-here"
```

Do NOT proceed until the key is confirmed set.

## Step 3: Create the Config File

Once all requirements are gathered:

1. Create a new folder named after the topic (keep the name short). All generated files (YAML config, dataset, train script) go inside this folder.
2. Create a file called `<short_topic_name>_config.yaml` using `assets/config_template.yaml` as a reference for the structure. The config uses three main sections: `topics`, `generation`, and `output`.
3. Fill in the config fields based on the user's answers. For fields the user was not asked about (to avoid overwhelming them), use sensible defaults based on the use case.

### Conversation type mapping

Map the user's choice to config values:
- **Q&A** → `conversation.type: basic`
- **Instruction-following** → `conversation.type: cot`, `conversation.reasoning_style: freetext`
- **Chat** → `conversation.type: basic`

### Provider mapping

Map the user's API provider choice to BOTH `topics.llm` and `generation.llm`:
- **Gemini** → `provider: "gemini"`, `model: "gemini-2.5-flash"`
- **Anthropic** → `provider: "anthropic"`, `model: "claude-sonnet-4-6"`

### Setting num_samples

Set `output.num_samples` based on the user's answer from Step 1. If they chose defaults, use the topic count (degree^depth). The config template defaults to 50, but this should always be overridden to match the user's depth/degree choice.

### Important: save_as paths

The `topics.save_as` and `output.save_as` paths in the config are **relative to where `deepfabric generate` is run from**. Since we will `cd` into the topic folder before running the command, use just the filenames:
```yaml
topics:
  save_as: "topics.jsonl"
output:
  save_as: "dataset.jsonl"
```

### If the user wants mock files

Mock files simulate a project directory that the AI assistant can read/write during training conversations. When enabled, the dataset will contain tool-use conversations (read_file, write_file, list_files) instead of plain text Q&A.

To enable mock files:

1. First, verify the Spin server is reachable: `curl -sf http://localhost:3030/healthz || curl -sf http://localhost:3030/`. If it fails, tell the user to start the Spin server (see Prerequisites) and wait for confirmation.
2. Set `generation.conversation.type` to `cot` and `generation.conversation.reasoning_style` to `agent`.
3. Fill in the `generation.tools` section in the config. Use port **3030** if running via Docker, or **3000** if running Spin locally. The structure must be:
   ```yaml
   generation:
     tools:
       spin_endpoint: "http://localhost:3030"  # Use port 3000 for local Spin
       components:
         builtin:
           - read_file
           - write_file
           - list_files
       scenario_seed:
         files:
           "filename.py": |
             # file contents here
       max_per_query: 3
       max_agent_steps: 5
   ```
4. **CRITICAL indentation rule**: `spin_endpoint`, `components`, `scenario_seed`, `max_per_query`, and `max_agent_steps` are ALL direct children of `tools:`. Only `files:` goes under `scenario_seed`. Getting this wrong will silently break tool use.
5. Generate realistic mock file contents based on the user's topic. These files should look like a real project the user would be working on — include starter code with TODOs, incomplete implementations, config files, tests, etc.

## Step 4: Generate the Dataset

**Important**: `cd` into the topic folder first, then run the generate command so all output files land in the right place:

```
cd <topic_folder> && deepfabric generate <short_topic_name>_config.yaml --tui simple
```

Use `--tui simple` so progress is visible in Claude Code's terminal (the default rich TUI doesn't render properly here). If the `--tui` flag is not recognized (older deepfabric versions), drop it and run without it — the command will still work, the output will just use the default TUI.

Mention it may take anywhere between 25-30 minutes to generate the entire dataset.

Once the command finishes, **verify the output before continuing**:
1. Check that `dataset.jsonl` exists: `ls -la dataset.jsonl`
2. Check it's non-empty: `wc -l dataset.jsonl`
3. Check it has roughly the expected number of samples (should be close to `output.num_samples` from the config)

If the file is missing, empty, or has significantly fewer samples than expected, investigate the error output and try to fix it before proceeding. Common issues:
- API key expired or rate-limited — check the error logs
- Spin server went down mid-generation (for mock files) — restart and re-run
- Network timeout — re-run with `--topics-load topics.jsonl` to skip topic generation and resume from where it left off

## Step 5: Create the Training Script

1. Locate the generated `dataset.jsonl` file (should be in the topic folder).
2. Create a file called `<short_topic_name>_train.py` using `assets/template_train.py` as the template.
3. Update the `data_files` path in the script to point to the generated `dataset.jsonl`.
4. Tell the user which model is currently set in the template (default: `Qwen/Qwen3-0.6B`) and ask if they want to change it. Update accordingly.
5. If the user started the Spin server using **Local Spin**, stop it now:
   ```
   kill $SPIN_PID 2>/dev/null && echo "Spin server stopped" || echo "Spin server already stopped"
   ```
   If `$SPIN_PID` is no longer in scope, find and kill the process by name: `pkill -f "spin up"`.
6. Say "All done!" and give a brief summary of everything that was created (folder name, config file, dataset file with sample count, training script).

## Important: Error Handling

If you run into an error at any step, sort it out before continuing with the rest of the steps. Do NOT skip steps or create files that depend on a previous step that failed.

## Important: API Key Setup

The `deepfabric generate` command requires an API key (e.g., `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`) to be set as an environment variable. This **must be done in the terminal** before running the command.

- If using **Claude Code CLI**: export the key in your terminal before launching, e.g., `export GEMINI_API_KEY="your-key-here"`, or pass it to the CLI session.
- If using **Claude Code in a GUI (e.g., VS Code)**: the GUI may not have access to your terminal environment variables. You will need to either set the key in your shell profile (`.bashrc`, `.zshrc`, etc.) so it's available globally, or export it manually in the integrated terminal before running the generate command.
