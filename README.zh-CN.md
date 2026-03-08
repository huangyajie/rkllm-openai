# RKLLM OpenAI API

[English](README.md) | [简体中文](README.zh-CN.md)

面向 Rockchip RKLLM（Rockchip Large Language Model）推理的 OpenAI 兼容 API 服务。

这个项目基于 FastAPI，对 RKLLM C++ Runtime 做了一层封装，使你可以在 Rockchip NPU 设备（如 RK3588、RK3576）上运行 LLM，并直接复用标准 OpenAI 客户端和工具链。

## 项目简介

`rkllm-openai` 将本地 RKLLM 推理能力暴露为 OpenAI 风格接口，当前主要提供：

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

适合以下场景：

- 将 Rockchip 设备接入现有 OpenAI SDK 或兼容客户端
- 在边缘设备上部署本地大模型或多模态模型
- 通过统一接口接入流式输出、工具调用、思维链输出等能力

## 功能特性

- **OpenAI API 兼容**：支持 `/v1/chat/completions` 和 `/v1/models`
- **多模态支持（VLM）**：支持视觉语言模型输入，例如 Qwen2-VL / Qwen3-VL
- **Thinking / Reasoning 输出**：解析 `<think>` 标签，并通过 `reasoning_content` 返回
- **NPU 硬件加速**：基于 `librkllmrt` 在 Rockchip NPU 上运行
- **流式输出**：支持 Server-Sent Events（SSE）
- **Function Calling / Tools**：支持通过 `tools` 字段传入工具定义
- **LoRA 支持**：支持加载 LoRA 适配器
- **Prompt Cache 支持**：可向底层运行时传入 `PROMPT_CACHE_PATH`
- **YAML 配置**：通过 `config.yaml` 或 `CONFIG_FILE` 环境变量管理配置

## 环境要求

- **硬件**：Rockchip RK3588、RK3576，或当前 RKLLM Runtime 支持的其他平台
- **系统**：Linux（推荐 Ubuntu / Debian）
- **驱动与运行时**：
  - `librkllmrt.so`
  - NPU driver `v0.9.7` 或更高版本
  - 多模态场景还需要 `librknnrt.so`
- **Python**：Python 3.12 或更高版本

## 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/huangyajie/rkllm-openai.git
   cd rkllm-openai
   ```

2. **安装依赖**

   项目使用 `uv` 管理 Python 依赖。

   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Sync dependencies
   uv sync
   ```

3. **准备 RKLLM Runtime**

   从官方 Rockchip RKLLM SDK 获取 `librkllmrt.so`，并将其放到 `lib/` 目录，或者在 `config.yaml` 中填写绝对路径。

   如果要启用多模态模型，还需要准备 `librknnrt.so`。

   ```bash
   mkdir -p lib
   cp /path/to/librkllmrt.so lib/
   cp /path/to/librknnrt.so lib/
   ```

4. **准备模型文件**

   - 文本模型：准备 `.rkllm` 模型文件
   - 多模态模型：除 `.rkllm` 外，还需要对应的视觉编码器 `.rknn` 文件

   你可以使用 RKLLM toolkit 自行转换模型，或者直接使用已经转换好的模型文件。

5. **修改配置**

   编辑仓库根目录下的 `config.yaml`，填入模型路径、运行时路径和服务参数。

## 配置说明

默认情况下，程序会读取仓库根目录的 `config.yaml`。如果需要加载其他配置文件，可以通过环境变量 `CONFIG_FILE` 覆盖：

```bash
CONFIG_FILE=/path/to/config.yaml uv run python main.py
```

当前代码实际支持的主要配置项如下：

```yaml
MODEL_PATH: "/path/to/your/model.rkllm"
VISION_MODEL_PATH: null
MODEL_NAME: "rkllm-model"
LORA_MODEL_PATH: null
PROMPT_CACHE_PATH: null

RKLLM_LIB_PATH: "/path/to/librkllmrt.so"
RKNN_LIB_PATH: "/path/to/librknnrt.so"
TARGET_PLATFORM: "rk3588"
RKNN_CORE_NUM: 3

IMG_START: "<|vision_start|>"
IMG_END: "<|vision_end|>"
IMG_CONTENT: "<|image_pad|>"

HOST: "0.0.0.0"
PORT: 8080
API_KEY: null

MAX_CONTEXT_LEN: 4096
MAX_NEW_TOKENS: 4096
TOP_K: 1
TOP_P: 0.9
TEMPERATURE: 0.8
REPEAT_PENALTY: 1.1
FREQUENCY_PENALTY: 0.0
PRESENCE_PENALTY: 0.0

LOG_LEVEL: "INFO"
LOG_DIR: "logs"
LOG_MAX_BYTES: 1048576
LOG_BACKUP_COUNT: 5
```

配置项说明：

- `MODEL_PATH`：`.rkllm` 模型文件路径
- `VISION_MODEL_PATH`：视觉编码器 `.rknn` 模型路径；启用多模态时需要配置
- `MODEL_NAME`：对外暴露的模型名称，`/v1/models` 会返回这个值
- `LORA_MODEL_PATH`：可选的 LoRA 适配器路径
- `PROMPT_CACHE_PATH`：可选的 Prompt Cache 路径
- `RKLLM_LIB_PATH`：`librkllmrt.so` 路径
- `RKNN_LIB_PATH`：`librknnrt.so` 路径；启用多模态时需要配置
- `TARGET_PLATFORM`：目标平台，例如 `rk3588`、`rk3576`
- `RKNN_CORE_NUM`：视觉编码器使用的 NPU 核心数，`3` 表示自动/全部
- `IMG_START`、`IMG_END`、`IMG_CONTENT`：多模态输入时使用的图像占位 token，默认适用于 Qwen2-VL / Qwen3-VL
- `HOST`、`PORT`：服务监听地址和端口
- `API_KEY`：当前版本中仅作为配置项保留，代码尚未在路由层强制校验鉴权
- `MAX_CONTEXT_LEN`、`MAX_NEW_TOKENS`、`TOP_K`、`TOP_P`、`TEMPERATURE`：生成参数默认值
- `REPEAT_PENALTY`、`FREQUENCY_PENALTY`、`PRESENCE_PENALTY`：底层推理时使用的惩罚参数
- `LOG_LEVEL`、`LOG_DIR`、`LOG_MAX_BYTES`、`LOG_BACKUP_COUNT`：日志输出配置

## 服务启动

使用默认 `config.yaml` 启动：

```bash
uv run python main.py
```

开发环境也可以直接通过 `uvicorn` 启动：

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```

启动后可以先检查健康状态：

```bash
curl http://localhost:8080/health
```

查看当前暴露的模型列表：

```bash
curl http://localhost:8080/v1/models
```

## 使用示例

### cURL 示例

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
    api_key="rkllm"  # API key is optional/ignored by default
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

说明：

- 当前实现只支持 `data:image/...;base64,...` 这种内联 Base64 图片
- 外部图片 URL 目前不会被下载处理
- 启用多模态前，请确保 `VISION_MODEL_PATH` 和 `RKNN_LIB_PATH` 已正确配置

### Thinking / Reasoning 示例

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

补充说明：

- 当请求体包含 `tools` 字段时，服务会将工具定义传递给底层模型
- 流式模式下，`reasoning_content` 和普通 `content` 会分开输出
- 服务同一时间只处理一个推理请求；并发请求会返回 `503 Server is busy`

## Docker 部署

仓库提供了面向 Rockchip 设备的 Docker 部署配置，相关文件位于 `docker/` 目录。

### Docker 前置要求

- 目标设备已安装 Docker
- 宿主机已正确加载 NPU 驱动，并可访问 `/dev/dri`
- 宿主机上已准备好模型文件和运行时库

### 使用 Docker Compose 启动

1. **准备文件**

   确保已经准备好：

   - `librkllmrt.so`
   - 多模态场景所需的 `librknnrt.so`
   - `.rkllm` 模型文件
   - 你的 `config.yaml`

2. **检查或修改挂载路径**

   `docker/docker-compose.yml` 默认挂载以下路径：

   ```yaml
   volumes:
     - ./lib:/app/lib:ro
     - ./models:/app/models:ro
     - ./config.yaml:/app/config/config.yaml:ro
     - ./logs:/app/logs
   ```

   这意味着当你在 `docker/` 目录下执行 `docker compose up` 时，Compose 会从 `docker/` 目录下读取这些相对路径。你可以：

   - 按照默认结构把文件放到 `docker/lib`、`docker/models`、`docker/config.yaml`
   - 或者直接把左侧路径改成宿主机上的绝对路径

3. **启动服务**

   ```bash
   cd docker
   docker compose up -d
   ```

`docker/docker-compose.yml` 已设置：

- `privileged: true`
- `network_mode: "host"`
- `devices: /dev/dri:/dev/dri`
- `CONFIG_FILE=/app/config/config.yaml`

这些配置用于让容器访问 Rockchip NPU 与挂载的配置文件。

## License

本项目基于 MIT License 开源，详见 [LICENSE](LICENSE)。

## 致谢

- [Rockchip RKLLM SDK](https://github.com/airockchip/rknn-llm)
