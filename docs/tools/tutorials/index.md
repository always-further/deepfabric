# Custom Tools Tutorials

Step-by-step guides for building custom Spin components that add domain-specific tools to DeepFabric's Tool Execution framework.

## Choose Your Language

<div class="grid cards" markdown>

-   :fontawesome-brands-python: **Python**

    ---

    Create weather tools using componentize-py

    [:octicons-arrow-right-24: Python tutorial](python.md)

-   :fontawesome-brands-golang: **Go**

    ---

    Build weather tools with the Spin Go SDK

    [:octicons-arrow-right-24: Go tutorial](go.md)

-   :fontawesome-brands-rust: **Rust**

    ---

    Build weather tools with the Spin Rust SDK

    [:octicons-arrow-right-24: Rust tutorial](rust.md)

-   :fontawesome-brands-js: **TypeScript**

    ---

    Build weather tools with the Spin JS SDK

    [:octicons-arrow-right-24: TypeScript tutorial](typescript.md)

</div>

## Workflow Overview

Each tutorial follows the same pattern:

1. **Create a Spin component** - Implement tool handlers in your language of choice
2. **Register in spin.toml** - Add your component to the main Spin configuration
3. **Build and run** - Compile to WASM and start the Spin server
4. **Configure DeepFabric** - Define your tools in the YAML config with `component:` routing
5. **Generate datasets** - Run DeepFabric to create training data using your tools

## Prerequisites

All tutorials assume you have:

- [Spin CLI](../spin.md) installed
- Language-specific toolchain (Rust/TinyGo/Python/Node.js)
- DeepFabric CLI installed
