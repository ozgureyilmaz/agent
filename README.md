# AI Coding Assistant

A simple AI agent. Supports Google Gemini and Anthropic Claude.

## Features

- Read, list, create, and edit files
- Support for Gemini and Claude APIs

## Quick Start

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
# Create a virtual environment and install dependencies
uv sync
```

Or manually create and activate:

```bash
# Create a virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Set Up API Key

Create a `.env` file:

```bash
# For Google Gemini
GEMINI_API_KEY=your_key_here

# OR for Anthropic Claude
ANTHROPIC_API_KEY=your_key_here
```

Get your API key:
- Gemini: https://makersuite.google.com/app/apikey
- Claude: https://console.anthropic.com/

### 4. Run

```bash
uv run main.py
```

## Usage

```
> List all Python files
> Read the config.py file
> Create a new file called test.txt with "Hello World"
> Change "old text" to "new text" in test.txt
```

Type `exit` to quit.
