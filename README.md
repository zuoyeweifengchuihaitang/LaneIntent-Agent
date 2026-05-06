# highD 车辆换道意图识别科研 Agent 

这是一个可直接运行的 **多 Agent 协作式科研实验流水线模板**，用于展示“使用 Agent / AI 驱动构建的具体成果”。它面向 highD 车辆轨迹数据，覆盖从数据检查、样本构建、类别均衡、F1 单帧物理特征生成、模型训练到报告生成的完整流程。

## 1. 项目解决的核心痛点

1. highD 数据处理链路长，人工检查 tracks / tracksMeta / recordingMeta 文件容易漏项。
2. 换道、直行样本构建规则复杂，人工维护脚本容易导致标签和样本索引不一致。
3. 机器学习、深度学习实验结果分散，人工整理 accuracy、macro-F1、混淆矩阵和报告效率低。
4. 论文实验需要可复现的流水线，而不是零散脚本。

## 2. Agent 核心逻辑流

本项目使用 5 个 Agent 协作：

| Agent | 作用 | 输出 |
|---|---|---|
| `PlannerAgent` | 根据配置生成实验计划 | `outputs/00_plan.json` |
| `DataAgent` | 扫描 highD 文件、构建换道/直行样本、类别均衡 | `sample_index.csv`, `sample_index_balanced.csv` |
| `FeatureAgent` | 生成单帧 F1 物理特征表 | `f1_frame_table.csv` |
| `ModelAgent` | 训练 SVM / RandomForest / LogisticRegression / 可选 XGBoost | `metrics_summary.csv`, 混淆矩阵 |
| `ReportAgent` | 汇总实验设置、样本分布、模型指标，生成 Markdown 报告 | `final_report.md` |

## 3. 项目结构

```text
highd_research_agent_mvp/
├─ main.py
├─ make_demo_data.py
├─ app.py
├─ config.example.json
├─ requirements.txt
├─ README.md
└─ src/
   ├─ agents.py
   ├─ config.py
   ├─ highd_labeling.py
   ├─ features_f1.py
   ├─ modeling.py
   ├─ reporting.py
   └─ utils.py
```

## 4. 快速运行

进入项目根目录后运行：

```bash
python main.py --config config.example.json
```

## 5. 使用你自己的 highD 数据

把 highD 文件放到类似下面的目录：

```text
your_data/raw_data/
├─ 01_tracks.csv
├─ 01_tracksMeta.csv
├─ 01_recordingMeta.csv
├─ 02_tracks.csv
├─ 02_tracksMeta.csv
└─ 02_recordingMeta.csv
```

然后修改 `config.example.json`：

```json
{
  "raw_data_dir": "your_data/raw_data",
  "output_dir": "outputs",
  "fps": 25,
  "left_when_lane_decreases": true
}
```

## 6. 关键输出

运行完成后，主要文件在 `outputs/` 中：

| 文件 | 用途 |
|---|---|
| `00_plan.json` | Agent 生成的实验计划 |
| `data_check_report.csv` | 数据完整性检查结果 |
| `sample_index.csv` | 原始样本索引 |
| `sample_index_balanced.csv` | 类别均衡后的样本索引 |
| `label_distribution_before.csv` | 均衡前类别统计 |
| `label_distribution_after.csv` | 均衡后类别统计 |
| `f1_frame_table.csv` | 单帧 F1 物理特征 |
| `feature_alignment_report.csv` | 样本索引与特征表对齐检查 |
| `metrics_summary.csv` | 各模型评估结果 |
| `final_report.md` | 自动生成的实验报告 |

## 7. 成果描述

我构建了一个面向车辆换道意图识别实验的 AI 辅助科研 Agent，主要解决 highD 数据处理链路长、实验脚本版本多、特征对齐和结果复现成本高的问题。该 Agent 会根据研究目标自动拆解任务，包括轨迹数据完整性检查、换道/直行样本标注、类别均衡处理、F1 单帧物理特征生成、特征对齐检查，以及 SVM、RandomForest、LogisticRegression 等模型的实验对比。

核心逻辑流包括：规划 Agent 生成实验流程，数据 Agent 构建样本索引，特征 Agent 生成模型输入，检查 Agent 验证样本数量和标签分布，模型 Agent 汇总 accuracy、macro-F1 和混淆矩阵，报告 Agent 自动生成实验总结。通过该 Agent，我将原本分散的手动实验流程整理成可复现的自动化流水线，减少了重复修改脚本和人工检查结果的时间，也提升了论文实验整理效率。

## 8. 注意事项

- 当前项目默认处理单帧 F1 特征，适合你的“单帧输入”实验设定。
- F2 鸟瞰图和 F3 拓扑图可在此框架上继续添加为新的 `FeatureAgent` 子模块。
- 若安装了 `xgboost`，程序会自动额外训练 XGBoost；没有安装也不影响主流程。
