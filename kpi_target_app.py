import os
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

# ======================
# Config
# ======================
EXCEL_FILE = "Comparison_Report.xlsx"
DEFAULT_SHEET = "Comparison Report"

# ======================
# Helpers
# ======================
def try_read_excel(path, sheet_name=None):
    """Đọc Excel, ưu tiên sheet_name nếu có"""
    if sheet_name:
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except Exception:
            pass
    return pd.read_excel(path)

def find_col(df, names):
    """Tìm cột theo nhiều tên khả dĩ (case-insensitive)"""
    lc_map = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lc_map:
            return lc_map[n.lower()]
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for n in names:
        k = n.lower().replace(" ", "").replace("_", "")
        if k in normalized:
            return normalized[k]
    return None

# ======================
# Load data
# ======================
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ Không tìm thấy file {file_path}")
        return None
    try:
        df = try_read_excel(file_path, sheet_name=DEFAULT_SHEET)
    except Exception as e:
        st.error(f"Lỗi khi đọc file Excel: {e}")
        return None
    return df

# ======================
# KPI calculation
# ======================
def calc_kpi(df, baseline_projects, level, col_hours, target_factor=1.0):
    """Tính KPI từ baseline projects theo cấp độ được chọn"""
    base_df = df[df["Project Name"].isin(baseline_projects)].copy()
    if base_df.empty:
        return pd.DataFrame()

    grouped = base_df.groupby(level)[col_hours].sum().reset_index()
    grouped.rename(columns={col_hours: "Baseline Hours"}, inplace=True)
    grouped["KPI Target"] = grouped["Baseline Hours"] * target_factor
    return grouped

# ======================
# Main Streamlit app
# ======================
def main():
    st.set_page_config(page_title="KPI Target Generator", layout="wide")

    # Logo
    if os.path.exists("triac_logo.png"):
        logo = Image.open("triac_logo.png")
        st.sidebar.image(logo, use_container_width=True)
    else:
        st.sidebar.info("Không tìm thấy triac_logo.png (tuỳ chọn).")

    st.title("📊 KPI / Target Generator từ Timesheet History")

    # Load data
    df = load_data(EXCEL_FILE)
    if df is None:
        return

    st.subheader("📄 Dữ liệu gốc (5 dòng đầu)")
    st.dataframe(df.head())

    # Tìm cột quan trọng
    col_hours = find_col(df, ["Hours", "workhour"])
    col_total = find_col(df, ["Total Hours", "TotalHours"])
    col_project = find_col(df, ["Project Name", "Project"])
    col_team = find_col(df, ["Team"])
    col_workcentre = find_col(df, ["Workcentre", "Workcenter"])
    col_task = find_col(df, ["Task"])
    col_job = find_col(df, ["Job"])

    st.markdown("**🔎 Cột được phát hiện:**")
    st.write({
        "Hours": col_hours,
        "Total Hours": col_total,
        "Project": col_project,
        "Team": col_team,
        "Workcentre": col_workcentre,
        "Task": col_task,
        "Job": col_job
    })

    if col_hours is None or col_project is None:
        st.error("❌ File Excel phải có ít nhất 2 cột: 'Hours' và 'Project Name'")
        return

    # Convert Hours numeric
    df[col_hours] = pd.to_numeric(df[col_hours], errors="coerce").fillna(0)

    # --- Chọn baseline ---
    all_projects = df[col_project].dropna().unique().tolist()
    baseline_projects = st.multiselect("👉 Chọn baseline project(s):", all_projects)

    # --- Drill-down level ---
    level = st.selectbox("👉 Chọn cấp độ KPI", [
        col_project, col_team, col_workcentre, col_task, col_job
    ])

    # --- Target factor ---
    target_factor = st.slider("🎯 Target factor (hệ số điều chỉnh)", 0.5, 2.0, 1.0, 0.1)

    # --- Tính KPI ---
    if baseline_projects:
        kpi_df = calc_kpi(df, baseline_projects, [level], col_hours, target_factor)
        if not kpi_df.empty:
            st.subheader(f"📊 KPI Target theo {level}")
            st.dataframe(kpi_df)

            # Biểu đồ
            fig = px.bar(kpi_df, x=level, y="KPI Target", text_auto=True,
                         title=f"KPI Target theo {level}")
            st.plotly_chart(fig, use_container_width=True)

            # Export Excel
            if st.button("📥 Xuất KPI ra Excel"):
                out_file = "KPI_Target.xlsx"
                kpi_df.to_excel(out_file, index=False)
                st.success(f"✅ Đã lưu file {out_file}")
    else:
        st.info("Hãy chọn ít nhất một baseline project để tính KPI.")

# ======================
# Run
# ======================
if __name__ == "__main__":
    main()
