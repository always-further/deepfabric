<style>
  .md-typeset h1,
  .md-content__button {
    display: none;
  }
</style>
<div align="center">
    <p align="center">
        <img src="images/logo-light.png" alt="DeepFabric Logo" width="500"/>
    </p>
  <h3>Synthetic Training Data for Agents</h3>

  <p>
    <a href="https://github.com/always-further/deepfabric/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
      <img src="https://img.shields.io/badge/Contribute-Good%20First%20Issues-green?style=for-the-badge&logo=github" alt="Good First Issues"/>
    </a>
    &nbsp;
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/badge/Chat-Join%20Discord-7289da?style=for-the-badge&logo=discord&logoColor=white" alt="Join Discord"/>
    </a>
  </p>

  <p>
    <a href="https://pypi.org/project/deepfabric/">
      <img src="https://img.shields.io/pypi/v/deepfabric.svg" alt="PyPI Version"/>
    </a>
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/discord/1384081906773131274?color=7289da&label=Discord&logo=discord&logoColor=white" alt="Discord"/>
    </a>
        <a href="https://www.reddit.com/r/deepfabric/">
      <img src="https://img.shields.io/badge/Reddit-r%2Fdeepfabric-FF4500?logo=reddit&logoColor=white" alt="Reddit"/>
    </a>
  </p>
</div>

## Quick Start

```bash
pip install deepfabric
export OPENAI_API_KEY="your-key"

deepfabric generate \
  --topic-prompt "DevOps and Platform Engineering" \
  --generation-system-prompt "You are an expert in DevOps and Platform Engineering generate examples of issue resolution and best practices" \
  --mode graph \
  --depth 2 \
  --degree 2 \
  --provider openai \
  --model gpt-4o \
  --num-samples 2 \
  --batch-size 1 \
  --conversation-type chain_of_thought \
  --reasoning-style freetext \
  --output-save-as dataset.jsonl
```

## What Just Happened?

The key steps in this example were as follows:

1. **Topic Graph Generation**: A topic hierarchy was created starting from "DevOps and Platform Engineering". Topic graphs take a root prompt and recursively expand subtopics to form a DAG (Direct Acyclic Graph) structure. Here, we used a depth of 2 and degree of 2 to ensure coverage of subtopics.
2. **Dataset Generation**: For each node topic in the graph, a synthetic dataset sample was generated using a chain-of-thought conversation style. Each example includes reasoning traces to illustrate the thought process behind the answers. With the above example, 2 samples were generated per topic as per the `--num-samples` flag.
3. **Conversation and Reasoning Style**: The `chain_of_thought` conversation type with `freetext` reasoning style. This encourages the model to provide detailed explanations along with answers, enhancing the quality of the training data.

So lets' break down this down visually:

```
Topic Graph:
- DevOps and Platform Engineering
  - CI/CD Pipelines
    - Best Practices for CI/CD
    - Common CI/CD Tools
  - Infrastructure as Code
    - IaC Benefits
    - Popular IaC Tools
```

So as you can see we have a depth of 2 (root + 2 levels) and a degree of 2 (2 subtopics per topic). 

Each of these topics would then be used to generate a corresponding dataset samples.

**"Best Practices for CI/CD"** Sample:
```json
{
  "question": "What are some best practices for implementing CI/CD pipelines?",
  "answer": "Some best practices include automating testing, using version control, and ensuring fast feedback loops.",
  "reasoning_trace": [
    "The user is asking about best practices for CI/CD pipelines.",
    "I know that automation is key in CI/CD to ensure consistency and reliability.",
    "Version control allows tracking changes and collaboration among team members.",
    "Fast feedback loops help catch issues early in the development process."
  ]
}
```

## Dataset Types

DeepFabric supports multiple dataset types to suit different training needs:

| Type | Description | Use Case |
|------|-------------|----------|
| [Basic](dataset-generation/basic.md) | Simple Q&A pairs | Instruction following |
| [Reasoning](dataset-generation/reasoning.md) | Chain-of-thought traces | Step-by-step problem solving |
| [Agent](dataset-generation/agent.md) | Tool-calling with real execution | Building agents |


