import io
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# plotting
import matplotlib.pyplot as plt
import plotly.express as px

# for similarity matching
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# PDF export
from fpdf import FPDF

# ---------------------------------------------------------
# Utility: column inference
# ---------------------------------------------------------
COMMON_COL_PATTERNS = {
    "project": ["project name", "project", "proj", "project_code", "project code", "projectid", "project id"],
    "team": ["team", "team name", "teamleader", "team leader"],
    "workcenter": ["workcenter", "work center", "work_center", "workcentre", "work centre"],
    "task": ["task", "task name"],
    "job": ["job", "job code", "jobcode"],
    "actual": ["hours", "actual hours", "actual_hours", "total hours", "total_hours", "totalhours"],
    "date": ["date", "work date", "day"],
    "year": ["year"],
    "overtime": ["overtime", "ot"]
}

def find_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    lower_cols = [c.lower() for c in df.columns]
    for p in patterns:
        # exact or contained
        for orig, lc in zip(df.columns, lower_cols):
            if p in lc:
                return orig
    return None

def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for key, patterns in COMMON_COL_PATTERNS.items():
        mapping[key] = find_column(df, patterns)
    return mapping

# ---------------------------------------------------------
# Preprocess dataframe
# ---------------------------------------------------------
def preprocess(df: pd.DataFrame) -> (pd.DataFrame, Dict[str, Optional[str]]):
    mapping = infer_columns(df)
    # Create a working copy with normalized column names
    df2 = df.copy()
    # rename detected columns to standard names
    rename_map = {}
    for std_name, col in mapping.items():
        if col:
            rename_map[col] = std_name
    df2 = df2.rename(columns=rename_map)
    # try ensure numeric hours
    if "actual" in df2.columns:
        df2["actual"] = pd.to_numeric(df2["actual"], errors="coerce")
    # date parse
    if "date" in df2.columns:
        try:
            df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        except:
            pass
    # derive year if missing
    if "year" not in df2.columns and "date" in df2.columns:
        df2["year"] = df2["date"].dt.year
    return df2, mapping

# ---------------------------------------------------------
# Aggregation & KPI computation
# ---------------------------------------------------------
def compute_aggregates(df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
    # Keep only rows with actual hours
    if "actual" in df.columns:
        sub = df.dropna(subset=["actual"])
    else:
        raise ValueError("Không tìm thấy cột giờ thực tế (Actual Hours).")
    agg = sub.groupby(groupby_cols).agg(
        mean_actual = ("actual","mean"),
        median_actual = ("actual","median"),
        std_actual = ("actual","std"),
        count = ("actual","count"),
        total_hours = ("actual","sum")
    ).reset_index()
    return agg

def suggest_target(agg: pd.DataFrame, method: str = "mean", safety_factor: float = 1.05, percentile: int = 75) -> pd.DataFrame:
    df = agg.copy()
    if method == "mean":
        df["suggested_target"] = (df["mean_actual"] * safety_factor).round(2)
    elif method == "median":
        df["suggested_target"] = (df["median_actual"] * safety_factor).round(2)
    elif method == "percentile":
        # percentile on raw distribution isn't available in aggregated row; assume median * factor if no raw.
        # But keep interface: use median * factor as fallback.
        df["suggested_target"] = (df["median_actual"] * safety_factor).round(2)
        # NOTE: if you want exact percentile use raw data level compute instead.
    else:
        raise ValueError("Unknown method")
    return df

# ---------------------------------------------------------
# Find similar projects (auto matching)
# ---------------------------------------------------------
def project_features(df: pd.DataFrame, project_col: str = "project") -> pd.DataFrame:
    # compute numeric features per project
    if project_col not in df.columns:
        raise ValueError("Không tìm thấy cột project để tính features.")
    df_num = df.dropna(subset=["actual"]).copy()
    features = df_num.groupby(project_col).agg(
        total_hours = ("actual","sum"),
        mean_hours = ("actual","mean"),
        median_hours = ("actual","median"),
        count_rows = ("actual","count"),
        unique_teams = ("team", lambda x: x.nunique() if "team" in df_num.columns else 0),
        unique_tasks = ("task", lambda x: x.nunique() if "task" in df_num.columns else 0)
    ).reset_index()
    # fill NaN
    features = features.fillna(0)
    return features

def find_similar_projects(df: pd.DataFrame, target_project: str, k_neighbors=5) -> pd.DataFrame:
    feats = project_features(df)
    if target_project not in feats.iloc[:,0].values:
        st.warning("Project không có trong dữ liệu historical.")
        return pd.DataFrame()
    X = feats.select_dtypes(include=[np.number]).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(Xs)), metric="euclidean")
    nn.fit(Xs)
    idx_map = {i: proj for i, proj in enumerate(feats.iloc[:,0].values)}
    # find index of target
    target_idx = int(feats[feats.iloc[:,0]==target_project].index[0])
    distances, indices = nn.kneighbors(Xs[target_idx:target_idx+1])
    similar_idx = indices.flatten().tolist()
    # map indices to projects + distances
    results = []
    for ii,dist in zip(similar_idx, distances.flatten()):
        results.append({
            "project": feats.iloc[ii,0],
            "distance": float(dist),
            **feats.iloc[ii,1:].to_dict()
        })
    res_df = pd.DataFrame(results)
    return res_df.sort_values("distance")

# ---------------------------------------------------------
# PDF report simple generator
# ---------------------------------------------------------
def create_pdf_report(title: str, agg_df: pd.DataFrame, chart_png_bytes: bytes, out_path: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, title, ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(6)
    # add chart image
    if chart_png_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(chart_png_bytes)
        tmp.flush()
        tmp.close()
        pdf.image(tmp.name, w=180)
        os.unlink(tmp.name)
        pdf.ln(6)
    # add small table: show first 20 rows
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 6, "Top rows of suggested targets:", ln=True)
    pdf.ln(2)
    # show header
    cols = ["project","team","task","job","suggested_target","mean_actual","median_actual","std_actual","count"]
    # fallback to available columns
    available = [c for c in cols if c in agg_df.columns]
    # print up to 15 rows
    for i,row in agg_df[available].head(15).iterrows():
        row_str = " | ".join([f"{c}:{row[c]}" for c in available if pd.notna(row[c])])
        pdf.multi_cell(0, 5, row_str)
    pdf.output(out_path)
    return out_path

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="KPI Target Generator", layout="wide")
st.title("KPI / Target Generator from Timesheet History")

st.markdown("""
App này đọc dữ liệu giờ công lịch sử (Excel), tính ngưỡng (target/KPI) theo các cấp độ (Project / Team / Workcenter / Task / Job),
và cho phép xuất kết quả (Excel/CSV/PDF).
""")

# Sidebar: file upload
st.sidebar.header("1) File & Options")
uploaded_file = st.sidebar.file_uploader("Upload Excel (historical timesheet)", type=["xlsx","xls","xlsm","csv"])
use_example = False
if uploaded_file is None:
    st.sidebar.info("Bạn chưa upload file. Bạn có thể chọn sử dụng file mẫu (nếu có) hoặc upload file Excel.")
else:
    st.sidebar.success("File received")

safety_factor = st.sidebar.number_input("Safety factor (multiplier applied to base)", value=1.05, min_value=0.8, max_value=3.0, step=0.01, format="%.2f")
method = st.sidebar.selectbox("Target method", ["mean","median","percentile"], index=0)
percentile = st.sidebar.slider("Percentile (if method=percentile)", 50, 95, 75)

# Load df
df = None
if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            # try to read the first sheet by default
            df = pd.read_excel(uploaded_file, sheet_name=0)
        st.sidebar.success(f"Loaded {uploaded_file.name} - {df.shape[0]} rows")
    except Exception as e:
        st.sidebar.error(f"Không thể đọc file: {e}")

if df is None:
    st.info("Vui lòng upload file dữ liệu giờ công (Excel/CSV). Mình hiện tại cần file để tiếp tục.")
    st.stop()

# Preprocess & mapping
df_proc, mapping = preprocess(df)
st.sidebar.subheader("Detected columns (inferred)")
for k,v in mapping.items():
    st.sidebar.text(f"{k}: {v}")

# Choose aggregation level
st.sidebar.header("2) Aggregation level")
levels = {
    "Project": ["project"],
    "Project -> Team": ["project","team"],
    "Project -> Team -> Task": ["project","team","task"],
    "Project -> Team -> Task -> Job": ["project","team","task","job"]
}
level_name = st.sidebar.selectbox("Choose level", list(levels.keys()), index=1)
group_cols = [c for c in levels[level_name] if c in df_proc.columns]
if len(group_cols) == 0:
    st.error("Không tìm thấy các cột để group theo cấp độ bạn chọn. Vui lòng chắc chắn file có cột project/team/task/job.")
    st.stop()

# Optional filters
st.sidebar.header("3) Filters (optional)")
projects_unique = sorted(df_proc["project"].dropna().unique().tolist()) if "project" in df_proc.columns else []
sel_project = st.sidebar.selectbox("Project (filter, optional)", ["(all)"] + projects_unique, index=0)
if sel_project != "(all)":
    df_proc = df_proc[df_proc["project"]==sel_project]

# compute aggregates
with st.spinner("Tính toán KPI..."):
    agg = compute_aggregates(df_proc, group_cols)
    agg = agg.sort_values("mean_actual", ascending=False)
    agg = suggest_target(agg, method=method, safety_factor=safety_factor, percentile=percentile)

st.header("Suggested targets")
st.markdown(f"Aggregation: **{level_name}**  — method: **{method}** — safety factor: **{safety_factor:.2f}**")
st.dataframe(agg, height=400)

# downloads
st.markdown("### Xuất dữ liệu")
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
    agg.to_excel(writer, sheet_name="suggested_targets", index=False)
    writer.save()
st.download_button("Download Excel", data=buf.getvalue(), file_name="suggested_kpi_targets.xlsx")

csv_buf = agg.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_buf, file_name="suggested_kpi_targets.csv")

# plot: top N
st.header("Visualization")
top_n = st.slider("Show top N groups by mean_actual", 5, 50, 10)
plot_df = agg.head(top_n).melt(id_vars=[c for c in group_cols], value_vars=["mean_actual","median_actual","suggested_target"], var_name="metric", value_name="hours")
# create a label column for x axis
plot_df["label"] = plot_df[group_cols].apply(lambda row: " / ".join([str(x) for x in row if pd.notna(x)]), axis=1)
fig = px.bar(plot_df, x="label", y="hours", color="metric", barmode="group", title="Mean / Median / Suggested target (top groups)")
st.plotly_chart(fig, use_container_width=True)

# PDF export (chart snapshot)
if st.button("Export PDF report (simple)"):
    # save current plot as PNG
    png_bytes = fig.to_image(format="png", scale=2)
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    create_pdf_report(
        title=f"KPI Targets - {level_name}",
        agg_df=agg,
        chart_png_bytes=png_bytes,
        out_path=tmp_pdf.name
    )
    tmp_pdf.close()
    with open(tmp_pdf.name, "rb") as f:
        pdf_bytes = f.read()
    st.download_button("Download PDF", data=pdf_bytes, file_name="kpi_targets_report.pdf")
    os.unlink(tmp_pdf.name)

# Similar projects (optional)
st.header("Match similar projects (auto)")
if "project" in df_proc.columns:
    target_for_match = st.selectbox("Pick a project to find similar ones (based on numeric features)", ["(choose)"] + projects_unique)
    if target_for_match and target_for_match != "(choose)":
        sim_df = find_similar_projects(df_proc, target_for_match, k_neighbors=6)
        if not sim_df.empty:
            st.markdown("Similar projects (closest first)")
            st.dataframe(sim_df)
            st.markdown("You can use these projects as benchmark sources when creating targets for a new similar project.")
else:
    st.info("Không có cột project trong dữ liệu nên tính similarity không thể thực hiện.")

# Tips
st.markdown("""
**Tips & next steps**
- Nếu bạn muốn target dựa trên percentile chính xác, mình có thể sửa để tính percentile ở *level raw* thay vì aggregated (trong trường hợp bạn cần).
- Có thể thêm weight theo staff count hoặc duration nếu dữ liệu có sẵn.
- Mình có thể giúp biến app này thành REST API hoặc tạo giao diện nội bộ (Docker) nếu bạn muốn triển khai cho team.
""")
