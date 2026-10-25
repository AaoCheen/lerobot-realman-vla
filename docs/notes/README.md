# 数据研究笔记

建议先读 [01_从采集到保存.md](01_从采集到保存.md)，再读 [02_数据盘点与清洗建议.md](02_数据盘点与清洗建议.md)。

准备训练请读 [03_你来操作_四卡全量微调准备.md](03_你来操作_四卡全量微调准备.md)。

理解数据形状、π0.5 flow matching训练/推理、训练效果和耗时，请读 [06_数据与训练原理](06_数据与训练原理/README.md)，按矩阵视角从数据到模型展开。

本次已将 `lerobot-realman-vla-history/data` **原地移动**到当前仓库 `data/`，同一文件系统，目录 inode 未变；没有复制、删集或修改数据文件。history 源码中的相对 `data/` 路径不会自动跟随移动，研究时按本文提供的新路径找数据，暂不在 history 运行采集或清洗命令。

| 文件 | 用途 |
|---|---|
| `01_从采集到保存.md` | 实际 call chain、状态/动作含义、HDF5 与 LeRobot v3 存储结构、新旧代码差异 |
| `02_数据盘点与清洗建议.md` | 本地实读计数、复核候选、数据版本重叠和训练前处理建议 |
| `episode_qc.csv` | 每个版本、每条 episode 的帧数、夹爪统计、异常及视频路径 |
| `data_audit.json` | 数据集汇总、5条原始 HDF5 检查和完全相同状态/动作轨迹分组 |
| `lineage_check.json` | 95条处理数据与47＋52条源版本的逐轨迹前缀匹配 |
| `audit_data.py` / `audit_lineage.py` | 只读检查脚本；仅覆盖本目录的小型检查结果，不修改数据 |
| `数据检查.ipynb` | 包含上述检查代码的可读副本，可使用有 numpy/pyarrow/h5py 的 kernel 复查 |

本次复核使用机器上已有的 `/home/ca/miniconda/envs/swift/bin/python` 和系统 `ffprobe`，未安装训练环境。命令从仓库根目录执行：

```bash
/home/ca/miniconda/envs/swift/bin/python docs/notes/audit_data.py
/home/ca/miniconda/envs/swift/bin/python docs/notes/audit_lineage.py
```

范围：读取全部5个训练数据集的 Parquet、5条 HDF5 的数值/结构，以及1042个 MP4 的视频流头信息。**未逐帧完整解码视频、未逐条人工判定任务成功、未启动机器人或训练。**
