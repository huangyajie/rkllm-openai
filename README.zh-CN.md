# RKLLM OpenAI API

[English](README.md) | [简体中文](README.zh-CN.md)

一个面向 Rockchip RKLLM（Rockchip Large Language Model）推理的 OpenAI 兼容 API 服务。

这个项目提供了一个基于 FastAPI 的服务，对 RKLLM C++ Runtime 进行了封装，使你能够在 Rockchip NPU（如 RK3588、RK3576）上运行 LLM，并使用标准 OpenAI 客户端和工具。

## 功能特性

- 🚀 **OpenAI API 兼容**：完整支持 `/v1/chat/completions` 和 `/v1/models`
- 🖼️ **多模态支持（VLM）**：支持带图像输入能力的视觉语言模型（例如 Qwen2-VL）
- 💭 **思考过程**：支持推理模型的 `<think>` 标签解析和 `reasoning_content` 输出
- ⚡ **硬件加速**：基于 `librkllmrt`，可在 Rockchip 设备上使用 NPU 加速
- 🌊 **流式支持**：实时 token 流式输出（Server-Sent Events）
- 🛠️ **函数调用**：支持工具使用 / 函数调用
- 🔌 **LoRA 支持**：动态加载 LoRA 适配器
- ⚙️ **可配置**：通过 `config.yaml` 文件轻松配置

## 环境要求

- **硬件**：基于 Rockchip RK3588 或 RK3576 的设备
- **系统**：Linux（推荐 Ubuntu / Debian）
- **驱动**：已安装 NPU 驱动（**v0.9.7 或更高版本**）以及 `librkllmrt.so`（运行时库）
- **Python**：Python 3.12 或更高版本

## 安装

1.  **克隆仓库：**

    ```bash
    git clone https://github.com/huangyajie/rkllm-openai.git
    cd rkllm-openai
    ```

2.  **安装依赖：**

    这个项目使用 `uv` 进行依赖管理。

    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Sync dependencies
    uv sync
    ```

3.  **准备 RKLLM Runtime：**

    确保你已从官方 Rockchip RKLLM SDK 获取 `librkllmrt.so`。
    将其放入 `lib/` 目录，或者在 `config.yaml` 中指定路径。

    ```bash
    mkdir -p lib
    cp /path/to/librkllmrt.so lib/
    ```

4.  **准备模型：**

    使用 RKLLM toolkit 将模型转换为 `.rkllm` 格式，或者下载已转换好的模型。

## 配置

在仓库根目录中的 `config.yaml` 里配置 rkllm 相关设置。

**关键配置项：**

```yaml
# Path to your converted RKLLM model file
MODEL_PATH: "/path/to/your/model.rkllm"

# Path to the vision encoder .rknn model file (optional for multimodal/VLM)
# VISION_MODEL_PATH: "/path/to/your/vision_model.rknn"

# Target Platform (rk3588 or rk3576)
TARGET_PLATFORM: "rk3588"

# Path to the RKLLM runtime library
RKLLM_LIB_PATH: "lib/librkllmrt.so"

# Path to librknnrt.so used for vision encoder (optional for VLM)
# RKNN_LIB_PATH: "lib/librknnrt.so"

# Server Configuration
HOST: "0.0.0.0"
PORT: 8080

# Generation Parameters (Defaults)
MAX_CONTEXT_LEN: 4096
MAX_NEW_TOKENS: 4096
TOP_K: 1
TOP_P: 0.9
TEMPERATURE: 0.8
```

## 运行服务

使用 `uv run` 启动 API 服务：

```bash
uv run python main.py
```

或者通过 `uv run` 直接使用 `uvicorn` 启动（适合开发环境）：

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```

## 使用方法

### CURL 示例

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

### Python 客户端示例（OpenAI SDK）

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

### 多模态（VLM）示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="rkllm-model",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }]
)
print(response.choices[0].message.content)
```

### 思考过程（Reasoning）示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="rkllm-model",
    messages=[{"role": "user", "content": "Which is larger, 9.11 or 9.9?"}],
    extra_body={"enable_thinking": True},
    stream=True
)

is_thinking = False
for chunk in response:
    delta = chunk.choices[0].delta
    # Handle reasoning content
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        if not is_thinking:
            print("\nThinking:\n", end="", flush=True)
            is_thinking = True
        print(delta.reasoning_content, end="", flush=True)
    
    # Handle normal content
    if delta.content:
        if is_thinking:
            print("\n\nAnswer:\n", end="", flush=True)
            is_thinking = False
        print(delta.content, end="", flush=True)
```

## Docker 部署

这个项目为 RK3588 / RK3576 设备提供了预配置的 Docker 部署方案。

### Docker 前置要求
- 你的 Rockchip 设备上已安装 Docker
- 主机系统已加载 NPU 驱动（存在 `/dev/dri`）
- 主机上已提供 `librkllmrt.so`

### 使用 Docker Compose 快速开始

1.  **准备你的环境：**
    确保你已经准备好模型文件和 `librkllmrt.so` 库文件。

2.  **编辑 `docker/docker-compose.yml`：**
    更新 volume 路径，使其指向你的本地文件：
    ```yaml
    volumes:
      - /path/to/host/lib:/app/lib:ro
      - /path/to/host/models:/app/models:ro
      - /path/to/host/config.yaml:/app/config/config.yaml:ro
    ```

3.  **运行服务：**
    ```bash
    cd docker
    docker compose up -d
    ```

## License
本项目基于 MIT License 开源，详情见 [LICENSE](LICENSE) 文件。

## 致谢

- [Rockchip RKLLM SDK](https://github.com/airockchip/rknn-llm)
