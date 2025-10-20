<div align="center">
# EATD-Corpus 抑郁症倾向音频分类实验
</div>

本仓库提供了一个统一的实验框架，旨在使用**线性探测 (Linear Probing)** 策略，对多种预训练音频模型在`EATD-Corpus`数据集上进行抑郁症倾向的二分类微调。

飞书小记链接：https://rcn55w9pzlac.feishu.cn/wiki/GfoOwLnltibm8ukXk63cCdBqnWg?from=from_copylink

---

## 🚀 快速开始

### 1. 环境配置

首先，创建并激活Conda环境，然后安装所需的核心依赖库。
**注意**: 每个实验子目录 (`CLAP/`, `Qwen2_Audio/` 等) 可能包含其特定的 `requirements.txt` 依赖文件，仅需根据backbone模型环境需要进行安装。

### 2. 数据准备

**这是一个关键步骤。** `EATD-Corpus` 数据集需要被分别放置在 **每一个** 实验文件夹 (`CLAP/`, `Qwen2_Audio/` 等) 内部的一个名为 `data` 的目录中（因为其`dataset.py`对数据集的处理不尽相同）。


### 3. 运行实验

使用位于项目根目录的 `run.bat` 脚本，可以方便地启动所有实验。

1.  直接双击 `run.bat` 文件，或在您的终端中运行它。
2.  脚本将提示您选择要运行的实验。
3.  输入您想运行的实验所对应的编号 (`1`, `2`, 或 `3`)，然后按回车键。
## 📊 性能表现

本项目采用**线性探测 (Linear Probing)** 微调策略。所有预训练骨干网络 (Backbone) 的参数均被冻结，仅训练一个在其之上新增的线性分类头。

各模型在 `EATD-Corpus` 验证集上的表现总结如下：

| 骨干模型 (Backbone) | 验证集准确率 (Accuracy) | 验证集F1分数 (F1-Score) |
| :------------------ |:-----------------:|:------------------:|
| **CLAP**            |     *0.5508*      |      *0.5990*      |
| **Qwen2-Audio**     |     *0.8093*      |      *0.7240*      |
| **Audio-Flamingo-2**|     *0.6398*      |      *0.6778*      |

## 📦 模型库 (Model Zoo)

本项目中所使用的模型及微调策略详情。

| 实验文件夹         | 基础检查点 (Base Checkpoint)          | 微调策略 (Fine-tuning Strategy) |
| :------------------- |:---------------------------------|:----------------------------|
| `CLAP/`              | `laion/clap-htsat-unfused`       | 线性探测 (Linear Probing)       |
| `Qwen2_Audio/`       | `Qwen/Qwen2-Audio-7B`            | 线性探测 (Linear Probing)       |
| `Audio_Flamingo_2/`  | `nvidia/audio-flamingo-2-0.5b`   | 线性探测  (Linear Probing)      |