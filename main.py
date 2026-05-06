from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.agents import PlannerAgent, DataAgent, FeatureAgent, ModelAgent, ReportAgent
from src.utils import ensure_dir, print_banner


def main() -> None:
    parser = argparse.ArgumentParser(description="highD vehicle intention research Agent MVP")
    parser.add_argument("--config", default="config.example.json", help="Path to config json file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg.output_dir)

    raw_dir = Path(cfg.raw_data_dir)
    if not raw_dir.exists() or not list(raw_dir.glob("*_tracks.csv")):
        print_banner("未发现 highD 数据，自动生成演示数据")
        from make_demo_data import create_demo_data
        create_demo_data(raw_dir)

    print_banner("1/5 PlannerAgent: 生成实验计划")
    plan = PlannerAgent(cfg).run()

    print_banner("2/5 DataAgent: 检查数据并构建样本索引")
    data_result = DataAgent(cfg).run()

    print_banner("3/5 FeatureAgent: 生成单帧 F1 特征并检查对齐")
    feature_result = FeatureAgent(cfg).run(data_result["balanced_index_path"])

    print_banner("4/5 ModelAgent: 训练模型并汇总指标")
    model_result = ModelAgent(cfg).run(feature_result["f1_path"])

    print_banner("5/5 ReportAgent: 生成最终报告")
    report_path = ReportAgent(cfg).run(plan, data_result, feature_result, model_result)

    print_banner("运行完成")
    print(f"输出目录: {Path(cfg.output_dir).resolve()}")
    print(f"最终报告: {Path(report_path).resolve()}")


if __name__ == "__main__":
    main()
