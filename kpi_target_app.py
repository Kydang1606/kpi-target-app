import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# ======================
# Config
# ======================
EXCEL_FILE = "Comparison_Report.xlsx"

# ======================
# Hàm xử lý dữ liệu
# ======================
def load_data(file_path):
    """Đọc dữ liệu từ Excel"""
    if not os.path.exists(file_path):
        st.error(f"Không tìm thấy file {file_path}")
        return None
    df = pd.read_excel(file_path)
    return df


def aggregate_data(df, group_cols, target_factor):
    """Tính toán KPI target"""
    if "workhour" not in df.columns:
        st.error("Thiếu cột 'workhour' trong dữ liệu")
        return None

    grouped = (
        df.groupby(group_cols)["workhour"]
        .sum()
        .reset_index()
        .rename(columns={"workhour": "total_hours"})
    )
    grouped["suggested_target"] = grouped["total_hours"] * target_factor
    return grouped


def plot_bar_chart(df, x_col, y_col, title="Bar Chart"):
    """Biểu đồ cột cơ bản"""
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", x=x_col, y=y_col, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_ylabel(y_col)
    st.pyplot(fig)


def plot_hierarchical_chart(df: pd.DataFrame, group_cols: list, value_col: str = "suggested_target"):
    """Biểu đồ phân tầng (treemap)"""
    if not all(col in df.columns for col in group_cols):
        st.warning("Thiếu cột cần thiết để vẽ biểu đồ phân tầng.")
        return None

    fig = px.treemap(
        df,
        path=[px.Constant("All")] + group_cols,
        values=value_col,
        color=value_col,
        color_continuous_scale="Blues",
        title=f"Biểu đồ phân tầng theo {group_cols} ({value_col})"
    )
    return fig

# ======================
# UI chính (Streamlit)
# ======================
def main():
    st.set_page_config(page_title="KPI Target Generator", layout="wide")

    # Logo
    if os.path.exists("triac_logo.png"):
        logo = Image.open("triac_logo.png")
        st.sidebar.image(logo, use_container_width=True)
        st.image(logo, width=120)
    else:
        st.sidebar.warning("⚠️ Không tìm thấy file triac_logo.png")

    st.title("KPI / Target Generator from Timesheet History")

    # Load data
    df = load_data(EXCEL_FILE)
    if df is None:
        return

    st.subheader("Dữ liệu gốc")
    st.dataframe(df.head())

    # Chọn cột group
    all_cols = list(df.columns)
    group_cols = st.multiselect("Chọn các cột để phân tích (group by):", options=all_cols, default=["project", "team"])
    target_factor = st.slider("Target factor (tỉ lệ so với giờ công)", 0.5, 2.0, 1.0, 0.1)

    if group_cols:
        agg = aggregate_data(df, group_cols, target_factor)
        if agg is not None:
            st.subheader("Kết quả tổng hợp")
            st.dataframe(agg)

            # Bar chart
            st.header("Biểu đồ cột")
            plot_bar_chart(agg, x_col=group_cols[0], y_col="suggested_target", title="Target theo nhóm")

            # Hierarchical chart
            st.header("Biểu đồ phân tầng (Hierarchical)")
            hier_fig = plot_hierarchical_chart(agg, group_cols, value_col="suggested_target")
            if hier_fig:
                st.plotly_chart(hier_fig, use_container_width=True)


if __name__ == "__main__":
    main()
