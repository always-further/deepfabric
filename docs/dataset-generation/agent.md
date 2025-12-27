# Agent Datasets

Agent datasets train models to use tools. DeepFabric supports two modes: single-turn (one-shot tool calls) and multi-turn (extended conversations with multiple tool interactions). If you're unsure which to use, start with single-turn, multi-turn is more brittle and requires careful configuration and is not needed for most training scenarios.

**Prerequisites**: Agent datasets require the Spin tool service. See [Tools](../tools/index.md) for setup instructions, or just use docker:

```bash
docker run -d -p 3000:3000 ghcr.io/always-further/deepfabric/tools-sdk:latest
```

## When to Use

- Training tool-calling capabilities
- Building agents that interact with APIs or systems
- ReAct-style reasoning with action-observation loops

## Single-Turn Agent

Single-turn mode generates complete tool workflows in one assistant response.

```yaml
topics:
  prompt: "Python Programming"
  graph: tree
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate tool usage examples with reasoning."
  instructions: "Create realistic scenarios requiring tools."

  conversation:
    type: chain_of_thought
    reasoning_style: agent
    agent_mode: single_turn

  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:
        - read_file
        - write_file
        - list_files
    max_per_query: 3
    max_agent_steps: 5

output:
  system_prompt: |
    You are an AI with access to tools. Analyze tasks, execute tools, and interpret results.
  num_samples: 4
  batch_size: 2
  save_as: "agent-dataset.jsonl"
```

### Single-Turn Output

```json
{
  "messages": [
    {"role": "system", "content": "You are an AI with access to tools."},
    {"role": "user", "content": "Find all while loops in main.py"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_0", "type": "function", "function": {"name": "search_file", "arguments": "{\"file_path\": \"main.py\", \"keyword\": \"while\"}"}}
    ]},
    {"role": "tool", "content": "[15, 42, 101]", "tool_call_id": "call_0"},
    {"role": "assistant", "content": "Found 3 while loops on lines 15, 42, and 101."}
  ],
  "reasoning": {
    "style": "agent",
    "content": [
      {"step_number": 1, "thought": "Need to search for 'while' keyword in main.py", "action": "search_file(file_path='main.py', keyword='while')"}
    ]
  },
  "tools": [...]
}
```

## Multi-Turn Agent

Multi-turn mode creates extended conversations with multiple tool interactions, following a ReAct pattern.

```yaml
topics:
  prompt: "Platform engineering tasks requiring tool usage"
  mode: graph
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate tool usage examples with reasoning."
  instructions: "Create realistic scenarios requiring tools."

  conversation:
    type: chain_of_thought
    reasoning_style: agent
    agent_mode: multi_turn

  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:
        - read_file
        - write_file
        - list_files
    max_per_query: 3
    max_agent_steps: 5

output:
  system_prompt: |
    You are an AI with access to tools. Analyze tasks, execute tools, and interpret results.
  num_samples: 4
  batch_size: 2
  save_as: "agent-platform-dataset.jsonl"
```

Multi-turn datasets include:
- Multiple tool call rounds
- Observation-based decisions
- Extended reasoning traces
- Planning over several steps

```json
{
    "messages": [
        {
            "content": "You are an AI with access to tools. Analyze tasks, execute tools, and interpret results.\n",
            "role": "system"
        },
        {
            "content": "\"I need to update the CI/CD pipeline configuration to integrate with the new version control system. Can you list the configuration files available?\"",
            "role": "user"
        },
        {
            "content": "",
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "arguments": "{}",
                        "name": "list_files"
                    },
                    "id": "5AgSQoude",
                    "type": "function"
                }
            ]
        },
        {
            "content": "[]",
            "role": "tool",
            "tool_call_id": "5AgSQoude"
        },
        {
            "content": "It seems that there are currently no configuration files available in the current session. Here are a few steps you can take to proceed:\n\n1. **Verify File Location:** Ensure that the configuration files are located in the correct directory that the tool is accessing. You might need to upload or move the files to the session's working directory.\n\n2. **Create a New Configuration File:** If the configuration files do not exist yet, you can create a new one. Let me know if you need assistance with writing a new configuration file for the CI/CD pipeline.\n\n3. **Check Permissions:** Ensure that the necessary permissions are set to access the files. Sometimes, files might be hidden or restricted due to permission settings.\n\n4. **Re-run the Tool:** Once you have verified the files are in place, you can re-run the `list_files` tool to check for their presence.\n\nIf you need further assistance or specific guidance on setting up the CI/CD pipeline configuration, feel free to ask!",
            "role": "assistant"
        },
        {
            "content": "\"I couldn't find the configuration files in the current session. Could you guide me on how to create a new version control configuration file for the CI/CD pipeline, or should I check the directory again for any hidden files?\"",
            "role": "user"
        },
        {
            "content": "To assist you in creating a new version control configuration file for your CI/CD pipeline, let's follow these steps:\n\n### Step 1: Determine the Requirements\n- **Version Control System (VCS):** Identify the VCS you are integrating with (e.g., Git, SVN).\n- **CI/CD Platform:** Specify the CI/CD platform you are using (e.g., Jenkins, GitLab CI, GitHub Actions).\n- **Pipeline Stages:** Define the stages of your pipeline (e.g., build, test, deploy).\n\n### Step 2: Create a Basic Configuration File\nLet's create a basic configuration file template. I'll guide you through writing a YAML file, which is commonly used for CI/CD configurations.\n\n#### Example YAML Configuration for GitLab CI\n```yaml\n# .gitlab-ci.yml\nstages:\n  - build\n  - test\n  - deploy\n\nbuild_job:\n  stage: build\n  script:\n    - echo \"Building the project...\"\n    - make build\n\nunit_tests:\n  stage: test\n  script:\n    - echo \"Running unit tests...\"\n    - make test\n\ndeploy_job:\n  stage: deploy\n  script:\n    - echo \"Deploying the project...\"\n    - make deploy\n```\n\n### Step 3: Write the Configuration File\nLet's write this configuration to a file. Please provide the desired file path, or I can use a default path for you.\n\n```python\n# Execute the write_file tool\nwrite_file(\n    file_path=\"./.gitlab-ci.yml\",\n    content=\"\"\"\n    stages:\n      - build\n      - test\n      - deploy\n    \n    build_job:\n      stage: build\n      script:\n        - echo \\\"Building the project...\\\"\n        - make build\n    \n    unit_tests:\n      stage: test\n      script:\n        - echo \\\"Running unit tests...\\\"\n        - make test\n    \n    deploy_job:\n      stage: deploy\n      script:\n        - echo \\\"Deploying the project...\\\"\n        - make deploy\n    \"\"\"\n)\n```\n\n### Step 4: Verify and Customize\n- **Verify:** Ensure the configuration file is correctly placed in your repository.\n- **Customize:** Modify the script commands to match your project's build, test, and deploy commands.\n\n### Step 5: Test the Configuration\n- **Commit and Push:** Add the configuration file to your version control and push it to trigger the CI/CD pipeline.\n- **Monitor:** Check the CI/CD platform for any errors and adjust the configuration as needed.\n\nIf you need further customization or encounter any issues, feel free to ask for more detailed guidance!",
            "role": "assistant"
        },
        {
            "content": "\"I've created the new version control configuration file using the template you provided. Could you guide me on how to integrate this file into the existing CI/CD pipeline, or should I run a test to ensure everything is working correctly?\"",
            "role": "user"
        },
        {
            "content": "",
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "arguments": "{\"file_path\":\"./.gitlab-ci.yml\"}",
                        "name": "read_file"
                    },
                    "id": "hEsYuJs4x",
                    "type": "function"
                }
            ]
        },
        {
            "content": "Error (FileNotFound): File not found: ./.gitlab-ci.yml",
            "role": "tool",
            "tool_call_id": "hEsYuJs4x"
        },
        {
            "content": "It seems there was an error when trying to read the configuration file, indicating that the file might not have been created or saved correctly. Let's address this step-by-step:\n\n### Step 1: Verify File Creation\n- **Check File Path:** Ensure that the file path `./.gitlab-ci.yml` is correct and that the file was created in the intended directory.\n- **Recreate the File:** If the file is missing, we can recreate it using the `write_file` tool.\n\n### Step 2: Recreate the Configuration File\nLet's recreate the configuration file using the template provided earlier. I'll execute the `write_file` tool for you:\n\n```python\nwrite_file(\n    file_path=\"./.gitlab-ci.yml\",\n    content=\"\"\"\n    stages:\n      - build\n      - test\n      - deploy\n    \n    build_job:\n      stage: build\n      script:\n        - echo \\\"Building the project...\\\"\n        - make build\n    \n    unit_tests:\n      stage: test\n      script:\n        - echo \\\"Running unit tests...\\\"\n        - make test\n    \n    deploy_job:\n      stage: deploy\n      script:\n        - echo \\\"Deploying the project...\\\"\n        - make deploy\n    \"\"\"\n)\n```\n\n### Step 3: Verify File Creation\n- **List Files:** Use the `list_files` tool to ensure the file is now present in the session.\n\n### Step 4: Integrate and Test\n- **Commit the File:** Add the `.gitlab-ci.yml` file to your version control system and push it to the repository.\n- **Trigger the Pipeline:** This should automatically trigger the CI/CD pipeline if it's set up to run on changes.\n- **Monitor the Pipeline:** Check your CI/CD platform for any errors or issues during the pipeline execution.\n\n### Step 5: Troubleshoot and Adjust\n- **Review Logs:** If there are errors, review the logs to identify any issues.\n- **Adjust Configuration:** Modify the configuration file as needed based on the feedback from the pipeline execution.\n\nIf you encounter any issues or need further assistance, feel free to ask!",
            "role": "assistant"
        }
    ],
    "metadata": {
        "conversation_type": "chain_of_thought",
        "topic": "Generate a multi-turn agent conversation with evolving tool usage.\n\nAvailable tools:\n- read_file(file_path: str) → File content as a string\n- write_file(file_path: str, content: str) → Confirmation message with bytes written\n- list_files() → JSON array of file paths\n\nYou may use 1 to 3 tools per query. Show tool dependencies and reasoning across conversation turns.\n\nARGUMENT REQUIREMENTS:\n- All argument values must be concrete and realistic (e.g., owner=\"acme-corp\", repo=\"web-app\", issue_number=42)\n- Never use template placeholders like {{owner}} or {{repo}}\n- Never use null values - omit optional parameters entirely if not needed\n- String fields must contain actual content, not empty strings\n\n\nHere are additional instructions:\n<instructions>\nCreate realistic scenarios requiring tools.\n</instructions>\n\n\nLastly, the topic of the training data should be related to the following subtopics: Platform engineering tasks requiring tool usage -> CI/CD pipeline management -> version control integration\n\nGenerate a complete multi-turn conversation using structured output with tool_executions list."
    },
    "reasoning": {
        "content": [
            {
                "action": "list_files({})",
                "step_number": 1,
                "thought": "To provide a list of configuration files available for the CI/CD pipeline, I need to first list all the files in the current session. This will help identify which files are relevant for the configuration update."
            },
            {
                "action": "Ready to respond",
                "step_number": 2,
                "thought": "The list_files tool returned an empty array, indicating that there are no files currently available in the session. This could mean that the files are not yet uploaded or accessible in this environment. To proceed, I need to inform the user about this situation and suggest uploading or checking the file paths."
            },
            {
                "action": "list_files({})",
                "step_number": 1,
                "thought": "To assist the user in creating a new version control configuration file for the CI/CD pipeline, I need to first ensure that there are no hidden files in the directory that might have been missed. If no configuration files are found, I can guide the user on creating a new configuration file."
            },
            {
                "action": "list_files({})",
                "step_number": 2,
                "thought": "I need to check if there are any hidden configuration files in the directory that might have been missed. If no configuration files are found, I will guide the user on creating a new configuration file."
            },
            {
                "action": "list_files({})",
                "step_number": 3,
                "thought": "I need to verify if there are any hidden configuration files in the directory that might have been missed. If no configuration files are found, I will guide the user on creating a new configuration file."
            },
            {
                "action": "Ready to respond",
                "step_number": 4,
                "thought": "Since the user couldn't find the configuration files, and I've attempted to list files multiple times without success, it's likely that the files are either not present or hidden. I'll guide the user on how to create a new version control configuration file for the CI/CD pipeline."
            },
            {
                "action": "read_file({\"file_path\":\"./.gitlab-ci.yml\"})",
                "step_number": 1,
                "thought": "Before integrating the new configuration file into the CI/CD pipeline, it's important to verify its content and ensure it aligns with the existing pipeline structure. Additionally, running a test can help confirm that the configuration is correct and functional."
            },
            {
                "action": "list_files({})",
                "step_number": 2,
                "thought": "Since the file was not found, it's possible that the file was not saved correctly or is in a different directory. Let's first list all files in the current session to verify the presence of the configuration file."
            },
            {
                "action": "list_files({})",
                "step_number": 3,
                "thought": "To proceed, I need to verify the presence of the newly created configuration file by listing all files in the current session. This will help determine if the file was saved correctly and is in the expected directory."
            },
            {
                "action": "list_files({})",
                "step_number": 4,
                "thought": "To ensure the configuration file is correctly integrated into the CI/CD pipeline, it's crucial to verify its existence and content. Since the file was not found earlier, listing all files will help confirm its presence and location. Once verified, I can guide you on testing the integration."
            },
            {
                "action": "list_files({})",
                "step_number": 5,
                "thought": "To ensure the configuration file is correctly integrated into the CI/CD pipeline, I need to verify its existence and content. Since the file was not found earlier, listing all files will help confirm its presence and location. Once verified, I can guide you on testing the integration."
            }
        ],
        "style": "agent"
    },
    "tool_context": {
        "executions": [
            {
                "arguments": "{}",
                "function_name": "list_files",
                "reasoning": "Listing all files will allow me to identify which ones are related to the CI/CD pipeline configuration.",
                "result": "[]"
            },
            {
                "arguments": "{\"file_path\":\"./.gitlab-ci.yml\"}",
                "function_name": "read_file",
                "reasoning": "To verify the content of the newly created configuration file and ensure it matches the intended setup.",
                "result": "Error (FileNotFound): File not found: ./.gitlab-ci.yml"
            }
        ]
    },
    "tools": [
        {
            "function": {
                "description": "Read content from a file",
                "name": "read_file",
                "parameters": {
                    "properties": {
                        "file_path": {
                            "description": "Path to the file to read",
                            "type": "string"
                        }
                    },
                    "required": [
                        "file_path"
                    ],
                    "type": "object"
                }
            },
            "type": "function"
        },
        {
            "function": {
                "description": "List all files in the current session",
                "name": "list_files",
                "parameters": {
                    "properties": {},
                    "required": [],
                    "type": "object"
                }
            },
            "type": "function"
        }
    ],
    "agent_context": {
        "execution_summary": "Completed 3-turn conversation",
        "mode": "multi_turn",
        "planning_trace": "### ScenarioOpenAICompat\n\n**Scenario Description:**\nA platform engineer is tasked with updating a configuration file in a CI/CD pipeline to integrate a new version control system. The engineer needs to ensure the changes are correctly implemented and verified.\n\n#### Turn 1: Initial Inquiry\n- **User:** \"I need to update the CI/CD pipeline configuration to integrate with the new version control system. Can you list the configuration files available?\"\n- **Tool Execution:**\n  - **Tool:** `list_files`\n  - **Parameters:** `{}`\n  - **Result:** `['/configs/pipeline.yml', '/configs/version_control.yml', '/configs/deploy.yml']`\n\n#### Turn 2: File Reading and Analysis\n- **User:** \"Please read the version control configuration file to check the current settings.\"\n- **Tool Execution:**\n  - **Tool:** `read_file`\n  - **Parameters:** `{\"file_path\": \"/configs/version_control.yml\"}`\n  - **Result:** \"version_control: git\\nbranch: master\\nrepository: https://github.com/acme-corp/web-app\"\n\n#### Turn 3: Modification and Verification\n- **User:** \"Update the version control configuration to use the new repository URL and branch. Then, confirm the changes.\"\n- **Tool Execution:**\n  - **Tool:** `write_file`\n  - **Parameters:** `{\"file_path\": \"/configs/version_control.yml\", \"content\": \"version_control: git\\nbranch: main\\nrepository: https://gitlab.com/acme-corp/web-app\"}`\n  - **Result:** \"File '/configs/version_control.yml' updated successfully with 98 bytes written.\"\n\n### Summary\nThis scenario demonstrates a multi-turn interaction where the user progressively builds on each step, requiring the listing of files, reading specific content, and then modifying that content based on the initial findings. Each tool execution is dependent on the results of the previous step, ensuring a coherent and logical flow of actions."
    },
    "question": "",
    "final_answer": ""
}
```

> [!NOTE]  
> You will sometimes see the model call files that do not exist in the tool context. This is expected behavior, as the model is forced to deal with hallucinated or missing files, just like in real-world scenarios. A FileNotFound error is a valid tool response that the model must handle appropriately.

## Tool Options

```yaml
tools:
  spin_endpoint: "http://localhost:3000"  # Spin service URL
  components:                              # Component-based tool routing
    builtin:                               # Built-in VFS tools -> /vfs/execute
      - read_file
      - write_file
      - list_files
  max_per_query: 3                         # Max tools per sample
  max_agent_steps: 5                       # Max reasoning iterations
  scenario_seed:                           # Pre-populate files
    files:
      "config.json": '{"debug": true}'
```

### Mock files

With `scenario_seed`, you can pre-create files in the agent's virtual filesystem:

```yaml
scenario_seed:
  files:
    "main.py": |
      for i in range(10):
          print(i)
    "Dockerfile": |
      FROM python:3.9-slim
      COPY . /app
      RUN pip install -r /app/requirements.txt
```

## CLI Usage

```bash
# Single-turn agent (requires Spin running)
deepfabric generate config.yaml \
  --conversation-type chain_of_thought \
  --reasoning-style agent \
  --agent-mode single_turn

# Multi-turn agent
deepfabric generate config.yaml \
  --agent-mode multi_turn \
  --min-turns 2 \
  --max-turns 4
```

## Next Steps

See [Tools](../tools/index.md) for:
- Installing and running Spin
- Available VFS tools (read_file, write_file, etc.)
- Creating custom tools
- Mock tool execution for testing
