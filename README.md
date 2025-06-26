## 项目概述

目前开源最快最小精度最高本地 ASR 模型：Sensevoice-Small。

本项目实现了一个完整的实时自动语音识别(ASR)系统，从最初的Python原型成功迁移到高性能的C++实现。系统集成了SenseVoice模型，支持中文、英文、日文、韩文和粤语等多语言识别，在保持完整功能的同时实现了显著的性能提升。

## 核心数据流程

整个ASR系统的数据处理流程可以分为以下几个关键步骤：

### 1. 音频采集阶段
```
麦克风 → PortAudio → 16kHz单声道PCM → VAD检测 → 音频片段
```

### 2. 特征提取阶段
```
音频片段 → 预加重 → 分帧(25ms/10ms) → Hamming窗 → FFT(512点) → 
功率谱 → Mel滤波器组(80维) → 对数变换 → LFR(7x6) → CMVN → 560维特征
```

### 3. 模型推理阶段
```
560维特征 → SenseVoice模型 → logits(25055维) → CTC解码 → token序列
```

### 4. 后处理阶段
```
token序列 → 词汇表解码 → 特殊符号过滤 → 最终文本
```

---

## 部署和使用

### 1. 环境准备

**依赖安装** (macOS):
```bash
# 安装基础工具
brew install cmake portaudio libsndfile curl

# 安装ONNX Runtime
# 从 https://github.com/microsoft/onnxruntime/releases 下载
# 或使用包管理器：
brew install onnxruntime

# 安装FFTW
brew install fftw
```

**依赖安装** (Ubuntu):
```bash
sudo apt update
sudo apt install -y cmake build-essential pkg-config \
    libportaudio2 libportaudio-dev \
    libsndfile1 libsndfile1-dev \
    libcurl4-openssl-dev \
    libfftw3-dev

# ONNX Runtime需要手动下载安装
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
```

### 2. 编译构建

```bash
# 克隆代码库
git clone <your-repo-url>
cd asr_cpp_project

# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 或使用提供的脚本
./build.sh
```

### 3. 使用方法

**基本使用**：
```bash
# 使用默认设置
./bin/asr_cpp

# 指定音频设备和参数
./bin/asr_cpp --device_index 1 --sample_rate 48000

# 使用Silero VAD提高精度
./bin/asr_cpp --vad_type silero --trigger_threshold 0.3

# 完整参数示例
./bin/asr_cpp \
  --device_index 1 \
  --sample_rate 48000 \
  --vad_type silero \
  --trigger_threshold 0.4 \
  --stop_threshold 0.2 \
  --max_record_time 10.0 \
  --silence_duration 1.5
```

**参数说明**：
- `--device_index`: 音频输入设备索引
- `--sample_rate`: 音频采样率 (支持自动重采样到16kHz)
- `--vad_type`: VAD类型 (`energy` 或 `silero`)
- `--trigger_threshold`: VAD触发阈值 (0.0-1.0)
- `--stop_threshold`: VAD停止阈值 (0.0-1.0)
- `--max_record_time`: 最大录制时间 (秒)
- `--silence_duration`: 静音停止时间 (秒)