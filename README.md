# RKLLM OpenAI API

An OpenAI-compatible API server for Rockchip RKLLM (Rockchip Large Language Model) inference.

This project provides a FastAPI-based server that wraps the RKLLM C++ runtime, allowing you to run LLMs on Rockchip NPUs (like RK3588, RK3576) using standard OpenAI clients and tools.

## Features

- 🚀 **OpenAI API Compatibility**: Full support for `/v1/chat/completions` and `/v1/models`.
- ⚡ **Hardware Acceleration**: Built on `librkllmrt` for NPU acceleration on Rockchip devices.
- 🌊 **Streaming Support**: Real-time token streaming (Server-Sent Events).
- 🛠️ **Function Calling**: Support for tool use/function calling.
- 🔌 **LoRA Support**: Dynamic loading of LoRA adapters.
- ⚙️ **Configurable**: Easy configuration via `.env` file.

## Prerequisites

- **Hardware**: Rockchip RK3588 or RK3576 based device.
- **System**: Linux (Ubuntu/Debian recommended).
- **Driver**: NPU driver (**v0.9.7 or higher**) and `librkllmrt.so` (Runtime library) installed.
- **Python**: Python 3.12 or higher.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/rkllm-openai.git
    cd rkllm-openai
    ```

2.  **Install Dependencies:**

    This project uses `uv` for dependency management.

    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Sync dependencies
    uv sync
    ```

3.  **Prepare RKLLM Runtime:**

    Ensure you have `librkllmrt.so` from the official Rockchip RKLLM SDK.
    Place it in the `lib/` directory or specify the path in `.env`.

    ```bash
    mkdir -p lib
    cp /path/to/librkllmrt.so lib/
    ```

4.  **Prepare Model:**

    Convert your model to `.rkllm` format using the RKLLM toolkit or download a pre-converted model.

## Configuration

Create a `.env` file in the root directory. You can copy the example:

```bash
cp .env.example .env
```

**Key Settings:**

```ini
# Path to your converted RKLLM model file
MODEL_PATH=/path/to/your/model.rkllm

# Target Platform (rk3588 or rk3576)
TARGET_PLATFORM=rk3588

# Path to the RKLLM runtime library
RKLLM_LIB_PATH=lib/librkllmrt.so

# Server Configuration
HOST=0.0.0.0
PORT=8080

# Generation Parameters (Defaults)
MAX_CONTEXT_LEN=4096
MAX_NEW_TOKENS=4096
TOP_K=1
TOP_P=0.9
TEMPERATURE=0.8
```

## Running the Server

Start the API server using `uv run`:

```bash
uv run python main.py
```

Or directly with `uvicorn` via `uv run` (useful for development):

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Usage

### CURL Example

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rkllm-model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello! Who are you?"}
    ],
    "stream": true
  }'
```

### Python Client Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="rkllm" # API key is optional/ignored by default
)

response = client.chat.completions.create(
    model="rkllm-model",
    messages=[
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Rockchip RKLLM SDK](https://github.com/airockchip/rknn-llm)
