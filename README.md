# RealMan π0.5：全参数微调与真机抓取（进行中）

基于 **LeRobot + π0.5** 的睿尔曼机械臂视觉语言动作（VLA）实践项目，覆盖已有示范数据检查、四卡 FSDP 全参数微调、检查点导出与跨设备真机测试。

**45000步检查点已完成真机夹取演示。** 下方为实拍演示的4倍速版本。本项目记录可复现的训练配置、数据处理和部署经验，目前尚未统计任务成功率。

[English](README_EN.md) · [训练原理](docs/notes/06_数据与训练原理/README.md) · [数据检查](docs/notes/02_数据盘点与清洗建议.md) · [Apache-2.0](LICENSE)

## 实机演示（4倍速）

![π0.5 45000步检查点真机夹取演示，4倍速](docs/assets/pi05_step45000_4x.gif)

45000步检查点 · RealMan RM65 · 手机实拍，完整片段4倍速播放（约32秒），无音频。播放速度不代表机械臂实际运行速度。

## 项目工作

- **数据检查与划分**：检查已有 RealMan 示范的状态/动作维度、数值有效性、夹爪变化与轨迹重复；隔离待复核数据，按 episode 划分训练/验证集，仅用训练集计算归一化统计。
- **π0.5全参数微调**：使用4张48GB GPU，以 FSDP FULL_SHARD、BF16 混合精度和 gradient checkpointing 运行完整模型训练；视觉、语言与动作网络均参与更新，不使用 LoRA、不冻结视觉编码器。
- **训练恢复与模型导出**：保存 FSDP 模型、优化器、调度器与随机状态，同时导出包含处理器和归一化参数的完整 BF16 推理模型；支持相同卡数、相同 batch 的断点恢复。
- **跨设备验证**：将45000步检查点迁移至 RTX 4080 现场工作站，完成真机抓取演示；整理数据、训练、模型交付与部署接口说明。

## 实机结果

| 项目 | 当前结果 |
|---|---|
| 任务 | 抓取物体并放入盒中；数据指令为 `pick up the plug and place it in the box` |
| 已测试模型 | π0.5全参数微调，45000步检查点 |
| 真机演示 | 本页4倍速GIF |
| 定量评估 | 尚未统计固定测试条件下的成功数/总次数 |

本次训练针对**单臂7维动作**。设备平台有三台机械臂，但当前结果不代表三臂协同或折盒技能已完成。训练 loss 用于检查拟合与数值稳定性，不能替代真机成功率。

## 数据规模与表示

数据来自同学已采集的 RealMan 示范，本项目在其基础上进行数据检查与微调实验。

| 数据项 | 数量/说明 |
|---|---|
| 原始合并数据 | 326条 episode，178362帧 |
| 暂时隔离 | 5条待回看轨迹；未删除原始数据，不将候选直接判为失败 |
| 训练集 | **289条，158355帧** |
| 验证集 | **32条，17624帧**；不参与训练及归一化统计 |
| 可训练起点 | 144194个完整50步动作块起点 |
| 图像 | 全局/腕部两路RGB，640×480，数据标称30 FPS |
| 状态与动作 | 7维：6个关节绝对角度（度）＋夹爪开合量 |
| 动作标签 | 下一帧实际状态；不是关节速度、力矩或角度增量 |

原数据采用 **LeRobotDataset v3**：Parquet 保存数值和索引，MP4 保存图像，元信息定位 episode 与视频片段。相同数值轨迹按组划分，避免直接重复轨迹跨越训练/验证集；近重复和任务完成质量仍需人工复核。当前入口保留验证集清单，**尚未自动计算验证 loss**。

## 全参数微调配置

| 配置 | 本次实验 |
|---|---|
| 基座策略 | π0.5，PaliGemma / Gemma 2B视觉语言侧＋Gemma 300M动作expert |
| 参数更新 | 全参数；`freeze_vision_encoder=false`、`train_expert_only=false`、`use_peft=false` |
| 并行与精度 | 单机4卡 FSDP FULL_SHARD；FP32训练主参数、BF16前向/反向 |
| 显存管理 | Gradient checkpointing；按实际调用的独立Linear设置FSDP单元 |
| Batch | 每卡2，全局8，无梯度累积 |
| 优化器 | AdamW，峰值学习率 `1e-5`，weight decay `0.01`，梯度裁剪 `1.0` |
| 调度 | 500步warmup，余弦衰减至 `1e-6` |
| 训练预算 | 约3轮完整动作块起点曝光，共54075次更新；45000步模型已实机测试 |
| 观测/动作块 | 当前1帧观测；预测未来50步动作，模型内部补齐到32维，loss仅计算实际7维 |
| 推理 | 10次flow积分；标准动作队列每次使用前10个动作后重新观测 |
| 归一化 | 状态/动作使用训练集q01/q99分位数；部署时使用同一组统计反归一化 |
| 保存策略 | 每5000步及训练末尾保存；新检查点完整写入后替换上一份 |

训练以随机噪声时间上的 **flow matching速度场MSE** 为目标，训练时不执行完整的多步采样；推理时才从噪声积分得到动作块。详细张量形状和公式见[矩阵视角说明](docs/notes/06_数据与训练原理/02_pi05训练与推理矩阵.md)。

## 设备分工

| 设备 | 配置 | 用途 |
|---|---|---|
| 训练服务器 | 8×NVIDIA RTX 5880 Ada，48GB/卡；本实验使用其中4卡 | π0.5全参数微调 |
| 数据/前期实验工作站 | NVIDIA RTX 4090 | 已有数据整理与前期微调实验 |
| 现场工作站 | NVIDIA RTX 4080，16GB | 现场采集/控制与模型部署测试 |
| 机械臂平台 | 睿尔曼 RM65，6轴＋夹爪；平台共三臂 | 本实验使用单臂完成抓取任务 |
| 视觉 | 全局＋腕部双RGB相机 | 提供 `camera_global`、`camera_left` 两路观测 |

## 复现入口

### 1. 独立训练环境

```bash
git clone https://github.com/AaoCheen/lerobot-realman-vla.git
cd lerobot-realman-vla
uv venv --python 3.11 .venv
source .venv/bin/activate
bash scripts/install_pi05_env.sh
```

训练依赖固定为 LeRobot `0.4.3`、PyTorch `2.7.1`、Torchvision `0.22.1`、Accelerate `1.10.1`。Transformers固定到支持π0.5的适配提交，安装脚本先准备该提交再本地安装，避免普通pip VCS安装拉取完整Git历史。

### 2. 数据与基座

将具有上述字段的 RealMan 数据放在 `data/datasets/realman_merged/`；准备本地π0.5基座 `models/pi05_base/` 和配套tokenizer `models/paligemma-3b-pt-224/`。本仓库不包含数据、基座、微调权重或虚拟环境。

筛选规则见 [pi05_data_selection.yaml](configs/pi05_data_selection.yaml)，训练参数见 [pi05_realman.yaml](configs/pi05_realman.yaml)。先按自己的数据和GPU修改配置。

### 3. 准备、短跑与正式训练

在tmux内执行训练。下面的卡号对应本次服务器，应按实际空闲卡修改：

```bash
export PI05_GPUS=0,1,2,7
export PI05_OUTPUT_DIR="$PWD/outputs/pi05_smoke01"
bash scripts/train.sh pi05 prepare
PI05_STEPS=10 bash scripts/train.sh pi05

# 确认短跑保存、模型重载和恢复后，使用新目录开始正式实验
export PI05_OUTPUT_DIR="$PWD/outputs/pi05_full_run01"
bash scripts/train.sh pi05
```

恢复同一实验：设置相同输出目录后运行 `bash scripts/train.sh pi05 resume`。当前只支持同4卡、同每卡batch恢复；改变batch需要重新规划数据进度和学习率调度。

### 4. 模型交付

```text
checkpoints/last/
├── pretrained_model/   # 完整权重、策略配置、前后处理器与归一化参数
├── training_state/     # FSDP模型/优化器等分片，仅用于训练恢复
└── complete.json       # 完整保存标记与步数
```

推理需传递整个 `pretrained_model/` 与配套tokenizer，更新目标机的本地路径，无需传递 `training_state/`。将候选阶段模型单独留存，避免被后续自动保留策略替换。

**部署需对齐双相机键、关节顺序、角度单位、夹爪方向和处理器。** 上游 `scripts/inference.py` 的 `cam_high/cam_wrist` 与本次训练的 `camera_global/camera_left` 不同，应确认真实视角后适配；此处不提供未经目标机验证的一键真机命令。

## 文档与验证

- [从采集到保存](docs/notes/01_从采集到保存.md)：上游采集链、动作定义与LeRobot存储。
- [数据盘点](docs/notes/02_数据盘点与清洗建议.md)：各版数据、异常候选和重叠关系。
- [数据与π0.5原理](docs/notes/06_数据与训练原理/README.md)：矩阵形状、flow matching、效果判断与耗时。
- [上游工程指南](docs/pipeline_guide.md)：采集/硬件等原有实现参考，其多策略结果不属于本次实验。

轻量测试不加载模型、不连接机器人：

```bash
python -m unittest discover -s tests -p 'test_pi05_preparation.py'
```

已完成本机四卡训练与检查点导出，45000步模型已进行实机夹取测试。当前没有声称相对基线提升、泛化成功率或三臂协同结果。

## 来源与贡献

本仓库基于 [Humble2Full/lerobot-realman-vla](https://github.com/Humble2Full/lerobot-realman-vla)，继承其 RealMan 采集、数据转换、硬件接口与多策略工程。模型和处理器使用 [Hugging Face LeRobot](https://github.com/huggingface/lerobot) 的π0.5实现，基座策略来自 Physical Intelligence 的π0.5工作。

本分支新增四卡全参数训练入口、数据筛选/划分与训练集统计、固定依赖安装脚本、训练恢复/导出及实验笔记；已有示范数据由同学采集，真机测试由项目参与者协作完成。硬件采集平台与预训练模型分别归属原作者，不作为本分支独创成果。

上游原有的ACT成功率、多策略对比与演示不作为本项目结果。保留 [Apache-2.0许可证](LICENSE) 及 [第三方声明](THIRD_PARTY_NOTICES.md)。
