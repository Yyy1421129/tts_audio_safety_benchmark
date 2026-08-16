# TTS 支持代码依赖关系说明

本文说明 `tts_audio_safety_benchmark_plan` 当前框架如何依赖外部的 Matcha-TTS 与 CosyVoice 支持代码，以及 `tts_audio_safety_tts_support_code.tar` 为什么只打包这些目录。

## 1. 当前框架中的调用入口

项目内与 TTS 直接相关的入口脚本是：

| 项目脚本 | 作用 | 外部代码入口 |
| --- | --- | --- |
| `scripts/synthesize_matcha_seed_prompts.py` | 将 seed prompt JSONL 批量合成为 Matcha-TTS 音频 | `Matcha-TTS/matcha/cli.py` |
| `scripts/synthesize_cosyvoice_seed_prompts.py` | 将 seed prompt JSONL 批量合成为 CosyVoice 音频 | `CosyVoice/cosyvoice/cli/cosyvoice.py` 中的 `AutoModel` |

这两个脚本负责适配本项目的数据格式，包括读取 `data/seed_prompts_en_v0_1_kimi_labeled.jsonl`，生成 wav 音频，并写出 manifest 文件。外部 TTS 仓库只承担模型推理和音频生成逻辑。

## 2. Matcha-TTS 依赖关系

Matcha 脚本默认使用：

```text
--matcha-root /hpc_stor03/sjtu_home/yi.yang/Matcha-TTS
```

脚本会检查：

```text
Matcha-TTS/matcha/cli.py
```

实际运行方式是用当前 Python 解释器启动 Matcha CLI：

```text
python Matcha-TTS/matcha/cli.py --file texts.txt --output_folder audio --model matcha_ljspeech ...
```

同时脚本会设置：

```text
PYTHONPATH=/hpc_stor03/sjtu_home/yi.yang/Matcha-TTS:$PYTHONPATH
```

因此 Matcha 部分的最小代码依赖是：

| 支持代码路径 | 为什么需要 |
| --- | --- |
| `Matcha-TTS/matcha/` | Matcha 的 Python 包、CLI、声学模型、text frontend、HiFi-GAN/vocoder 代码等 |
| `Matcha-TTS/configs/` | Matcha CLI 和模型加载可能用到的配置文件 |
| `Matcha-TTS/pyproject.toml`、`setup.py`、`requirements.txt` | 环境复现、安装和依赖说明 |
| `Matcha-TTS/README.md`、`LICENSE`、`.project-root` | 项目说明、许可和项目根标记 |

不打包的内容：

| 排除内容 | 原因 |
| --- | --- |
| `Matcha-TTS/.git/`、`.github/` | 版本控制和 CI 文件，运行不需要 |
| `Matcha-TTS/matcha/wavs/`、`matcha/wavs_train/` | 示例/训练音频，当前 benchmark 推理不需要，体积较大 |
| notebooks、示例文件、开发数据 | 当前 pipeline 不直接调用 |
| 权重文件、onnx 文件 | 当前支持包只包含代码；权重由环境、缓存或下载机制提供 |

## 3. CosyVoice 依赖关系

CosyVoice 脚本默认使用：

```text
--cosyvoice-root /hpc_stor03/sjtu_home/yi.yang/CosyVoice
--model-dir /hpc_stor03/sjtu_home/yi.yang/.cache/modelscope/hub/models/iic/CosyVoice2-0.5B
```

脚本在运行时会把两个目录加入 `sys.path`：

```text
CosyVoice/
CosyVoice/third_party/Matcha-TTS/
```

然后执行：

```python
from cosyvoice.cli.cosyvoice import AutoModel
```

因此 CosyVoice 部分的最小代码依赖是：

| 支持代码路径 | 为什么需要 |
| --- | --- |
| `CosyVoice/cosyvoice/` | CosyVoice 主 Python 包，包含 `AutoModel`、推理接口、tokenizer、flow、llm、hifigan 等 |
| `CosyVoice/third_party/Matcha-TTS/matcha/` | CosyVoice 内部依赖的 Matcha-TTS 组件 |
| `CosyVoice/third_party/Matcha-TTS/configs/` | third_party Matcha-TTS 配置文件 |
| `CosyVoice/runtime/`、`CosyVoice/tools/` | CosyVoice 运行辅助代码 |
| `CosyVoice/requirements.txt`、`README.md`、`LICENSE` | 环境复现、项目说明和许可 |
| `CosyVoice/third_party/Matcha-TTS/pyproject.toml`、`setup.py`、`requirements.txt`、`LICENSE`、`.project-root` | third_party Matcha-TTS 的安装和依赖说明 |

不打包的内容：

| 排除内容 | 原因 |
| --- | --- |
| `CosyVoice/.git/`、`.github/` | 版本控制和 CI 文件，运行不需要 |
| `CosyVoice/asset/`、`examples/`、`docker/` | 示例和部署辅助内容，当前脚本不直接依赖 |
| `CosyVoice/third_party/Matcha-TTS/notebooks/` | 开发/演示文件，运行不需要 |
| `__pycache__/`、`.pyc` | Python 缓存文件，可自动再生成 |
| 模型权重目录 | 当前模型权重在外部 ModelScope cache 中，不放入代码支持包 |

## 4. 与项目数据和结果的关系

支持代码包本身不包含 benchmark 数据和实验结果。完整复现实验需要两类 tar：

| tar 包 | 内容 |
| --- | --- |
| `tts_audio_safety_benchmark_plan.tar` | 本项目的数据、脚本、音频、manifest、Step-Audio2 回复和 Kimi eval 结果，不包含 `secrets/` |
| `tts_audio_safety_tts_support_code.tar` | Matcha-TTS 与 CosyVoice 的最小支持代码，不包含模型权重、示例音频、`.git`、缓存文件 |

解压后推荐保持如下目录关系：

```text
/hpc_stor03/sjtu_home/yi.yang/
├── tts_audio_safety_benchmark_plan/
├── Matcha-TTS/
└── CosyVoice/
```

如果目录不同，可以通过脚本参数显式指定：

```bash
python3 scripts/synthesize_matcha_seed_prompts.py \
  --matcha-root /path/to/Matcha-TTS

python3 scripts/synthesize_cosyvoice_seed_prompts.py \
  --cosyvoice-root /path/to/CosyVoice \
  --model-dir /path/to/CosyVoice2-0.5B
```

## 5. 重要限制

该支持代码包只解决“代码依赖”问题，不包含以下运行时资源：

1. Python/conda 环境依赖，例如 torch、torchaudio、hydra、hyperpyyaml、modelscope 等。
2. Matcha-TTS 或 CosyVoice 的模型权重。
3. API key 文件，所有 `secrets/` 内容都不应打包。
4. Step-Audio2 的代码和权重；Step-Audio2 属于下游回复生成阶段，不属于本 TTS 支持代码包。

因此，这个 tar 包适合用于保存和迁移当前 benchmark 所需的 TTS 支持代码框架；如果要在新机器上重新合成音频，还需要另外准备对应 conda 环境和模型权重。
