from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="highD 科研 Agent Dashboard", layout="wide")
st.title("highD 车辆换道意图识别科研 Agent Dashboard")

output_dir = Path(st.sidebar.text_input("输出目录", "outputs"))

st.markdown("该页面用于查看 Agent 流水线生成的样本分布、模型指标和最终报告。")

metrics_path = output_dir / "metrics_summary.csv"
label_before = output_dir / "label_distribution_before.csv"
label_after = output_dir / "label_distribution_after.csv"
report_path = output_dir / "final_report.md"

col1, col2 = st.columns(2)
with col1:
    st.subheader("均衡前标签分布")
    if label_before.exists():
        df = pd.read_csv(label_before)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("尚未找到 label_distribution_before.csv")

with col2:
    st.subheader("均衡后标签分布")
    if label_after.exists():
        df = pd.read_csv(label_after)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("尚未找到 label_distribution_after.csv")

st.subheader("模型指标")
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("尚未找到 metrics_summary.csv，请先运行 python main.py --config config.example.json")

st.subheader("最终报告")
if report_path.exists():
    st.markdown(report_path.read_text(encoding="utf-8"))
else:
    st.info("尚未生成 final_report.md")
