# VideoSubX

VideoSubX 是一个以 [VideoLingo](https://github.com/Huanshere/VideoLingo) 部分功能为基础，并增加了部分新功能和优化的自动化视频字幕翻译工具。

> [!IMPORTANT]
> **模型选择结论（多轮实测）**
> - 当前项目在字幕任务（断句/翻译/重断句）上的综合稳定性与效果排序为：`gpt-4.1 > gpt-5.2 > claude-sonnet-4-5`。
> - 请勿盲目选择 SOTA 模型，建议先以项目实际链路做小样本 A/B 测试后再定型。

部署教程：https://www.bilibili.com/video/BV1FnFhzgERG

本项目遵循 [Apache 2.0](LICENSE) 许可证。
#### 重要提示：install.py是基于50系以下设备使用的安装脚本，50系请自行安装最新cuda cudnn，torch，理论上已经去除了强制要求老版本的decumus，AMD 9000系也曾经正常跑通过，自行尝试
50系已有用户安装最新cuda和torch后成功运行
## ⚠️ 首次使用必读

> [!IMPORTANT]
> **网络代理要求**
> - 首次使用时会自动下载 ASR 模型（Whisper）和人声分离模型，**请开启系统代理**
> - 首次切换到新语言的视频时，会自动下载对应的 NLP 模型（如 `ja_core_news_md`），**也请确保系统代理开启**
> - 如遇视频下载卡住，请检查代理设置或手动配置 `config.yaml` 中的 `proxy` 字段

> [!WARNING]
> **启动脚本配置**
> - `run_webui.bat` 中写死了 Conda 环境名称为 `videosubx`
> - 请确保 **Conda 已添加到系统 PATH**，且环境名称与脚本中一致
> - 如需修改环境名称，请编辑 `run_webui.bat` 文件

> [!CAUTION]
> **NumPy 版本问题**
> - 存在未知 Bug 可能导致 `install.py` 中的 NumPy 版本覆盖失效
> - 如遇到 NumPy 相关报错，请手动在对应conda环境里确认 NumPy 版本：`pip show numpy`
> - 推荐版本：`numpy=1.26.4`

## 项目核心变更与重构说明

本项目为了解决原项目在生产环境中的稳定性与精度问题，进行了一部分重写：

1. **架构迁移：FastAPI + 任务队列** 彻底弃用了原有的 Streamlit 架构。Streamlit 强依赖于浏览器进程，一旦网页关闭，后台的长耗时任务（如转写、翻译）就会中断。VideoSubX 采用了 FastAPI 作为后端核心，引入了独立的任务管理系统，并重写了前端页面。现在，可以放心地关闭浏览器，后台任务依然会稳定运行。
2. **音频处理与时间轴重构**
   - **人声分离**：弃用了 Decumus，替换为维护更频繁的 `python-audio-separator`。
   - **时间轴对齐**：弃用了 WhisperX。虽然 WhisperX 速度快，但在某些场景下时间轴漂移严重。我们替换为 `stable-ts`，并引入了 VAD 相关参数，换取了更高的时间轴精确度。
3. **切割与翻译逻辑优化**
   - **硬性分割**：弃用了基于连接词等的软分割逻辑，新增基于停顿时长的硬性分割。
   - **提示词优化**：内置了经过测试优化的 Prompt，并移除了原项目中的 TTS 模块和远程 Whisper API 依赖，专注于本地化的字幕生成质量。
   - **断句 Prompt 重写（新增）**：重写了 LLM 断句提示词的优先级（语义完整性/禁悬挂尾词优先，长度均衡降级），减少模型擅自改写原句的风险。
   - **英文 ASR 词级校正（新增）**：在 `cleaned_chunks.xlsx` 生成后新增独立校正步骤，修正产品名称/人物名称等错误，按 `start_key` 返回并应用替换建议，避免在断句阶段混入纠错任务。

## 🧪 实验性功能：MFA 强制对齐

> [!WARNING]
> **实验性功能** - 此功能需要手动安装，可能会影响现有环境依赖。

### 功能介绍

MFA https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner 是一个声学强制对齐工具。启用后可以**修复部分词语起始时间戳晚于说话人一个音节才开始的问题**，让字幕时间轴更加精准。

启用前：<img width="356" height="191" alt="图片" src="https://github.com/user-attachments/assets/ec93ee5f-455b-4ccf-b099-66305bb80a08" />

启用后：<img width="387" height="194" alt="图片" src="https://github.com/user-attachments/assets/83ee5946-0416-45cb-acdd-2a5c706f8a06" />
单词起始时间明显修复
### 安装方法

```bash
# 在已激活的 conda 环境中运行
conda activate videosubx
python install_mfa.py
```

### 启用方式

> 升级提示：新增 `alignment.mode` 数字选择器，`config.yaml` 需要同步升级；旧的 `mfa.enabled` / `disable_all_align` 逻辑已移除。

编辑 `config.yaml`：

```yaml
alignment:
  mode: 4  # 1=raw, 2=stable-ts align only, 3=mfa only, 4=stable-ts + mfa

mfa:
  acoustic_model: 'english_mfa'  # 或 mandarin_mfa, japanese_mfa 等
  dictionary: 'english_mfa'
```

对齐模式说明：
- `1`：`raw`，无对齐（仅 ASR 原始结果）
- `2`：`stable-ts align only`
- `3`：`mfa only`
- `4`：`stable-ts + mfa`（先 `stable-ts align_words`，再 MFA）

### ⚠️ 常见问题

安装 MFA 可能会导致以下依赖冲突，`install_mfa.py` 已内置修复逻辑，如遇报错请优先检查：

| 问题 | 症状 | 修复方式 |
|------|------|----------|
| **NumPy 版本冲突** | `ModuleNotFoundError: No module named '_kalpy'` | 脚本会自动替换 pip 版本为 conda 版本 |
| **FFmpeg DLL 冲突** | `无法定位程序输入点 DllMain` | 脚本会自动删除冲突的 DLL 文件 |

如自动修复失败，可手动执行：

```bash
# 修复 NumPy
pip uninstall numpy -y
conda install -c conda-forge numpy=1.26.4 -y

# 修复 FFmpeg（删除 conda 环境中的冲突文件）
Remove-Item "$env:CONDA_PREFIX\Library\bin\avcodec*.dll" -Force
Remove-Item "$env:CONDA_PREFIX\Library\bin\ffmpeg.exe" -Force
```

## **界面展示**

<img width="1402" height="939" alt="图片" src="https://github.com/user-attachments/assets/f0546d3f-45dd-47ed-adbf-b83f960e430b" />


## 环境要求 (必读)

> ⚠️ **兼容性警告**： 目前本项目 **仅测试 Windows 系统** 且搭载 **NVIDIA 显卡**。 Linux、macOS 或 AMD 显卡环境尚未测试，可能会有意想不到的错误，请自行尝试。AMD 显卡用户请自行安装 ROCm 版本torch测试，曾在9060XT中运行成功过老版本

### 核心依赖

- **Python**: 3.10 (强烈建议使用 Conda 环境)
- **CUDA**: **使用 CUDA 11.8**，并安装对应的 cuDNN。
  - *注：虽然理论上支持更高版本，但在依赖库兼容性上 11.8 最为稳定。*
- **FFmpeg**: 必需的视频处理后端，需添加至系统环境变量。

## 安装指南

### 1. 环境准备

```bash
conda create -n videosubx python=3.10 -y
conda activate videosubx
```

### 2. 获取代码

```bash
git clone https://github.com/assassinliujie/VideoSubX.git
cd VideoSubX
```

### 3. 一键安装

我们重写了 `install.py` 以适应新的依赖关系。

> **提示**：安装过程中涉及从 HuggingFace 和 Github 下载模型，请务必开启系统代理模式，否则极大概率失败。

```bash
python install.py
```

## 配置说明

在启动前，请复制 `config.example.yaml` 为 `config.yaml`，并编辑其中的配置项。除了常规的 API Key，以下几项设置至关重要：

### 1. 网络代理 (Proxy)

由于 `yt-dlp`（下载视频）和部分模型加载需要访问特定网络，如果您的网络环境受限，**必须**配置代理地址。
注意，此处的代理端口号7890并不是固定的，取决于您的代理软件的设置，具体请询问AI，如果你选择开启系统代理模式，那么该项可以不配置

```yaml
# 示例：http://127.0.0.1:7890
proxy: 'http://127.0.0.1:7890'
```

### 2. 字体设置 (Fonts)

项目涉及视频硬字幕烧录，配置文件中的字体名称必须与您系统中实际安装的字体名称完全一致，否则会导致字幕乱码或方框。

```yaml
style:
  chinese_font: 'SimHei'
  english_font: 'Arial'
```

### Base URL 填写规则（重要）

正确写法：`https://api.openai.com/v1`。

### 3. LLM 请求超时与重试（新增）

当上游接口偶发慢请求时，建议使用较短超时 + 多次重试，以减少“长时间无日志反馈”的卡顿感。

```yaml
api:
  request_timeout_sec: 30      # 单次请求超时时间（秒）
  request_retries: 8           # 失败后重试次数（不含首次）
  request_retry_delay_sec: 1   # 每次重试间隔（秒）
```

### 4. 润色翻译开关（注释更新）

```yaml
# 是否启用二段式翻译（直译 + 润色）
# true: 先直译再润色（两次调用，质量更高，速度更慢）
# false: 单次融合翻译（一次调用，更快），并在翻译后自动执行一次“全文输入/全文输出”润色（实验性）
reflect_translate: true
```

### 5. 英文 ASR 词级校正（新增，默认关闭）

该步骤会在 `cleaned_chunks.xlsx` 生成后执行，向 LLM 发送词级英文识别结果，返回基于 `start_key` 的替换建议并应用到原文件。  
校正规则是“仅允许改单词，不允许增删词、不允许语法改写”。
每次执行会追加写入 `output/log/english_correction_changelog.csv`，即使 LLM 返回空建议也会记录一条 no-op 日志；同时会写入完成标记，供“继续任务”正确跳过该步骤。

```yaml
english_correction:
  enabled: false
  only_when_detected_language: true
  api:
    # 留空则回退到全局 api.*
    key: ''
    base_url: ''
    model: ''
    llm_support_json: true
    request_timeout_sec: 30
    request_retries: 5
    request_retry_delay_sec: 1
```

> [!IMPORTANT]
> 推荐先在小样本上开启验证。该步骤会影响后续分句与时间轴对齐输入文本。

### 6. rough_split 专有名词断裂回并（新增，可选）

该步骤在 `rough_split` 之后执行，向 LLM 发送相邻行边界信息，识别“跨行被断开的多词专有名词”（如产品型号、公司名、人名、地名等）。  
LLM 只负责识别边界断裂，真正拼接由脚本执行，并根据行长自动判断：
- 把下一句句首并回上一句句末
- 或把上一句句末并到下一句句首

每次执行会追加写入 `output/log/rough_split_entity_repair_changelog.csv`，即使 LLM 返回空建议也会记录一条 no-op 日志，并在首次运行时备份 `output/log/rough_split_before_entity_repair.txt`。

```yaml
rough_split_entity_repair:
  enabled: false
  only_when_space_joiner: true
  max_pairs_per_request: 120
  boundary_window_words: 8
  max_fragment_words: 4
  api:
    # 留空则回退到全局 api.*
    key: ''
    base_url: ''
    model: ''
    llm_support_json: true
    request_timeout_sec: 30
    request_retries: 5
    request_retry_delay_sec: 1
```

### 7. single-pass 全文润色（新增，实验性）

当 `reflect_translate: false` 时，流程会在单次融合翻译完成后，自动追加一次“全文输入 / 全文输出”的统一润色。
该步骤会严格校验行号与行数，要求输出与输入逐行对应；若失败会自动回退到原始 single-pass 翻译结果，不中断后续流程。

```yaml
single_pass_full_polish:
  api:
    # 留空则回退到全局 api.*
    key: ''
    base_url: ''
    model: ''
    llm_support_json: true
    request_timeout_sec: 120
    # 失败后重试次数（不含首次），默认 2
    request_retries: 2
    request_retry_delay_sec: 1
```

## 启动服务

安装完成并修改配置后，通过以下命令启动后端服务：

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8501
```

服务启动后，请在浏览器访问 `http://localhost:8501` 进入新的管理界面。
## 致谢

感谢原项目https://github.com/Huanshere/VideoLingo 提供的魔改基础，非常好的项目
