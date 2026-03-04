# DeepFabric Dataset Generator — Claude Code Skill

A Claude Code skill that generates synthetic datasets using [DeepFabric](https://docs.deepfabric.dev/) and creates training scripts for fine-tuning LLMs.

## Setup

### 1. Set your API key

Before anything else, export the API key for the provider you plan to use:

```bash
# For Gemini (recommended)
export GEMINI_API_KEY="your-key-here"

# For Anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

To make the key persistent across sessions, add the export line to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

> **Note:** If you're using Claude Code in a GUI (e.g., VS Code), the integrated terminal may not inherit your shell environment variables. Either add the export to your shell profile so it's available globally, or run the export manually in the integrated terminal before generating.

### 2. Install DeepFabric

```bash
pip3 install deepfabric
```

### 3. Install the skill

Copy the `deepfabric-generator` folder into your Claude Code skills directory:

```bash
cp -r deepfabric-generator ~/.claude/skills/
```

Your directory structure should look like this:

```
~/.claude/skills/deepfabric-generator/
├── SKILL.md
├── README.md
└── assets/
    ├── config_template.yaml
    └── template_train.py
```

### 4. Mock files setup (optional)

If you want your dataset to include **mock files** (simulated project files the AI can read/write during training conversations), you'll need a running Spin server. Choose one of the following:

**Option A: Docker (preferred — no extra toolchain needed)**

```bash
cd /path/to/deepfabric/tools-sdk && docker compose up -d
```

The server runs on port **3030**.

**Option B: Local Spin server**

Requires [Spin CLI](https://developer.fermyon.com/spin) (v2.0+) and Rust with the `wasm32-wasip1` target:

```bash
# Install Spin (if needed)
curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash && sudo mv spin /usr/local/bin/

# Install Rust wasm target (if needed)
rustup target add wasm32-wasip1

# Build and start the server
cd /path/to/deepfabric/tools-sdk && spin build && spin up
```

The server runs on port **3000**.

If you don't need mock files, skip this step entirely.

## Usage

In Claude Code, tell the assistant you want to generate a dataset and provide a topic. For example:

> "I want to make a dataset about building REST APIs in Python"

The skill will walk you through choosing:

- **Knowledge graph depth & degree** — controls how many subtopics are generated
- **Conversation type** — Q&A, instruction-following, or chat
- **Turn structure** — single-turn or multi-turn
- **Reasoning style** — chain-of-thought, direct, or step-by-step
- **Mock files** — whether to include simulated file operations (requires Spin server)
- **API provider** — Gemini or Anthropic

It then generates a DeepFabric config, runs dataset generation (~25-30 minutes), and creates a training script using [TRL's SFTTrainer](https://huggingface.co/docs/trl/sft_trainer).

## Output

For each dataset, the skill creates a project folder containing:

| File | Description |
|------|-------------|
| `<topic>_config.yaml` | DeepFabric configuration |
| `topics.jsonl` | Generated topic graph |
| `dataset.jsonl` | The synthetic dataset |
| `<topic>_train.py` | Fine-tuning script (default model: `Qwen/Qwen3-0.6B`) |

## Requirements

- Python 3.8+
- `deepfabric` package
- `trl` and `datasets` packages (for training)
- A valid Gemini or Anthropic API key
- (Optional) Docker or Spin CLI for mock file support
