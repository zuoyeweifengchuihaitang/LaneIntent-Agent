from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import Config


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False)


def write_final_report(
    cfg: Config,
    plan: dict,
    data_result: dict,
    feature_result: dict,
    model_result: dict,
) -> str:
    out_dir = Path(cfg.output_dir)
    report_path = out_dir / "final_report.md"

    before = pd.read_csv(data_result["label_distribution_before_path"])
    after = pd.read_csv(data_result["label_distribution_after_path"])
    metrics = pd.read_csv(model_result["metrics_path"])

    test_metrics = metrics[metrics["split"] == "test"].copy()
    if not test_metrics.empty:
        test_metrics = test_metrics.sort_values("macro_f1", ascending=False)

    text = f"""# highD 车辆换道意图识别科研 Agent 实验报告

## 1. 任务概述

本次流水线由多 Agent 协作完成，目标是将 highD 风格车辆轨迹数据转换为可用于机器学习实验的单帧样本表，并自动完成基础模型训练、指标汇总和报告生成。

## 2. 核心配置

| 配置项 | 数值 |
|---|---:|
| raw_data_dir | `{cfg.raw_data_dir}` |
| output_dir | `{cfg.output_dir}` |
| fps | {cfg.fps} |
| left_when_lane_decreases | {cfg.left_when_lane_decreases} |
| vy_threshold | {cfg.vy_threshold} |
| future_lateral_disp_threshold | {cfg.future_lateral_disp_threshold} |
| lane_keep_stride_frames | {cfg.lane_keep_stride_frames} |
| test_size | {cfg.test_size} |

## 3. Agent 工作流

1. `PlannerAgent`：生成实验计划，明确数据检查、样本构建、特征生成、模型训练和报告输出顺序。
2. `DataAgent`：检查 highD 文件完整性，识别换道样本和直行样本，并完成类别均衡。
3. `FeatureAgent`：基于单帧数据生成 26 维 F1 物理特征，并执行样本 ID 对齐检查。
4. `ModelAgent`：训练多个传统机器学习模型，输出 accuracy、balanced accuracy、macro-F1 等指标。
5. `ReportAgent`：自动汇总关键表格，生成本报告。

## 4. 标签分布

### 4.1 均衡前

{_md_table(before)}

### 4.2 均衡后

{_md_table(after)}

## 5. 特征对齐检查

特征维度：{model_result.get('n_features')}  
样本数量：{model_result.get('n_samples')}

对齐报告文件：`{feature_result['alignment_report_path']}`

## 6. 模型测试集结果

{_md_table(test_metrics)}

## 7. 可用于材料填写的成果总结

我构建了一个面向车辆换道意图识别实验的 AI 辅助科研 Agent，主要解决 highD 数据处理链路长、实验脚本版本多、特征对齐和结果复现成本高的问题。该 Agent 会根据研究目标自动拆解任务，包括轨迹数据完整性检查、换道/直行样本标注、类别均衡处理、F1 单帧物理特征生成、特征对齐检查，以及多种机器学习模型的实验对比。

核心逻辑流包括：规划 Agent 生成实验流程，数据 Agent 构建样本索引，特征 Agent 生成模型输入，检查 Agent 验证样本数量和标签分布，模型 Agent 汇总 accuracy、macro-F1 和混淆矩阵，报告 Agent 自动生成实验总结。通过该 Agent，我将原本分散的手动实验流程整理成可复现的自动化流水线，减少了重复修改脚本和人工检查结果的时间，也提升了论文实验整理效率。

## 8. 输出文件索引

| 文件 | 说明 |
|---|---|
| `{data_result['data_check_path']}` | 数据完整性检查 |
| `{data_result['index_path']}` | 原始样本索引 |
| `{data_result['balanced_index_path']}` | 均衡后样本索引 |
| `{feature_result['f1_path']}` | F1 单帧特征表 |
| `{model_result['metrics_path']}` | 模型指标汇总 |
| `{model_result['confusion_matrix_dir']}` | 混淆矩阵目录 |
"""
    report_path.write_text(text, encoding="utf-8")
    return str(report_path)
