from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import Config
from .utils import ensure_dir, write_json
from .highd_labeling import (
    check_recording_files,
    build_sample_index,
    balance_index,
    save_label_distribution,
    plot_label_distribution,
)
from .features_f1 import build_f1_features, check_feature_alignment
from .modeling import train_and_evaluate
from .reporting import write_final_report


class PlannerAgent:
    """Experiment planning Agent.

    In a real LLM-Agent system, this module can be replaced by a model call.
    Here it uses deterministic planning so that the whole project can run locally.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self) -> dict:
        plan = {
            "project": "highD 车辆换道意图识别科研 Agent MVP",
            "pain_points": [
                "highD 数据处理链路长，文件和路径检查容易出错",
                "换道/直行标签规则复杂，人工维护样本索引成本高",
                "实验结果分散，论文表格和报告整理耗时",
            ],
            "agents": [
                {"name": "DataAgent", "task": "数据完整性检查、样本构建、类别均衡"},
                {"name": "FeatureAgent", "task": "生成单帧 F1 物理特征、检查样本对齐"},
                {"name": "ModelAgent", "task": "训练并评估基础机器学习模型"},
                {"name": "ReportAgent", "task": "生成实验报告与成果描述"},
            ],
            "logic_flow": [
                "scan_raw_highd_files",
                "build_lane_change_and_lane_keep_samples",
                "balance_classes",
                "generate_26d_f1_single_frame_features",
                "train_baseline_models",
                "write_final_report",
            ],
        }
        write_json(plan, Path(self.cfg.output_dir) / "00_plan.json")
        print("[PlannerAgent] 实验计划已写入 outputs/00_plan.json")
        return plan


class DataAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self) -> dict:
        out_dir = ensure_dir(self.cfg.output_dir)

        check_df = check_recording_files(self.cfg)
        data_check_path = out_dir / "data_check_report.csv"
        check_df.to_csv(data_check_path, index=False, encoding="utf-8-sig")
        print(f"[DataAgent] 数据检查完成: {data_check_path}")

        if (check_df["status"] == "error").any():
            print("[DataAgent] 检测到 error，仍尝试继续处理；如失败请先修复数据字段。")

        index = build_sample_index(self.cfg)
        index_path = out_dir / "sample_index.csv"
        index.to_csv(index_path, index=False, encoding="utf-8-sig")

        before_path = out_dir / "label_distribution_before.csv"
        before_dist = save_label_distribution(index, before_path)
        if self.cfg.save_plots:
            plot_label_distribution(before_dist, out_dir / "label_distribution_before.png", "Label Distribution Before Balance")

        balanced = balance_index(index, self.cfg)
        balanced_path = out_dir / "sample_index_balanced.csv"
        balanced.to_csv(balanced_path, index=False, encoding="utf-8-sig")

        after_path = out_dir / "label_distribution_after.csv"
        after_dist = save_label_distribution(balanced, after_path)
        if self.cfg.save_plots:
            plot_label_distribution(after_dist, out_dir / "label_distribution_after.png", "Label Distribution After Balance")

        print(f"[DataAgent] 原始样本数: {len(index)}，均衡后样本数: {len(balanced)}")
        print(f"[DataAgent] 均衡后分布: {after_dist[['label_name', 'count']].to_dict(orient='records')}")

        return {
            "data_check_path": str(data_check_path),
            "index_path": str(index_path),
            "balanced_index_path": str(balanced_path),
            "label_distribution_before_path": str(before_path),
            "label_distribution_after_path": str(after_path),
            "n_samples_before": int(len(index)),
            "n_samples_after": int(len(balanced)),
        }


class FeatureAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, balanced_index_path: str | Path) -> dict:
        out_dir = ensure_dir(self.cfg.output_dir)
        index = pd.read_csv(balanced_index_path)
        f1 = build_f1_features(self.cfg, index)
        f1_path = out_dir / "f1_frame_table.csv"
        f1.to_csv(f1_path, index=False, encoding="utf-8-sig")

        report = check_feature_alignment(index, f1)
        alignment_path = out_dir / "feature_alignment_report.csv"
        report.to_csv(alignment_path, index=False, encoding="utf-8-sig")

        print(f"[FeatureAgent] F1 特征表: {f1_path}")
        print(f"[FeatureAgent] 对齐检查: {alignment_path}")
        return {
            "f1_path": str(f1_path),
            "alignment_report_path": str(alignment_path),
            "n_features": 26,
            "n_samples": int(len(f1)),
        }


class ModelAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, f1_path: str | Path) -> dict:
        result = train_and_evaluate(self.cfg, f1_path)
        print(f"[ModelAgent] 指标汇总: {result['metrics_path']}")
        return result


class ReportAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, plan: dict, data_result: dict, feature_result: dict, model_result: dict) -> str:
        report_path = write_final_report(self.cfg, plan, data_result, feature_result, model_result)
        print(f"[ReportAgent] 最终报告: {report_path}")
        return report_path
