# app.py — Student Feedback Dashboard (all-native visuals, dark theme, polished)
# Run: streamlit run app.py

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional: for live word cloud rendering
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Feedback Dashboard", layout="wide")

# ---------- Paths (base + assets) ----------
base_dir = os.path.dirname(os.path.abspath(__file__))
ASSETS   = os.path.join(base_dir, "assets")
DATA_DIR = os.path.join(base_dir, "data")

DATA_FILE      = os.path.join(DATA_DIR, "student_feedback_clean.csv")
SEGMENTED_FILE = os.path.join(DATA_DIR, "student_feedback_segmented.csv")
SENTIMENT_FILE = os.path.join(DATA_DIR, "student_feedback_with_sentiment.csv")
COMMENTS_FILE  = os.path.join(DATA_DIR, "student_feedback_with_comments_synthetic.csv")

# Optional PNG references (fallbacks only)
PCA_IMG = os.path.join(ASSETS, "fig_clusters_pca.png")
WORDCLOUD_IMG = os.path.join(ASSETS, "fig_wordcloud_synthetic.png")

# == Dark theme + CSS (safe KPI spacing) ==
st.markdown("""
    <style>
      :root { --bg:#0f172a; --panel:#111827; --text:#e5e7eb;
              --muted:#9ca3af; --accent:#34d399; --border:#1f2937; }
      html, body, [class*="css"] { font-family: "Segoe UI", Roboto, Arial, sans-serif; }
      body { background: var(--bg); color: var(--text); }
      .block-container { padding-top: 2.4rem; padding-bottom: 1.0rem; max-width: 1500px; }
      h1,h2,h3,h4 { color: var(--text); }
      .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
             padding: 16px 16px; text-align: left; min-height: 92px; overflow: visible; }
      .kpi .label { color: var(--muted); font-size:12px; margin-bottom:6px;}
      .kpi .value { color: var(--text); font-size:28px; font-weight:650; line-height:1.15; }
      .kpi .sub   { color: var(--accent); font-size:12px; }
      .modebar{ opacity:0.12; transition:opacity .2s;} .modebar:hover { opacity:1; }
      section[data-testid="stSidebar"] { background-color: #0b1220; }
    </style>
""", unsafe_allow_html=True)

def style_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=13, color="#e5e7eb"),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        legend=dict(font=dict(color="#e5e7eb"))
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1f2937",
                     zeroline=False, linecolor="#374151",
                     tickfont=dict(color="#e5e7eb"), titlefont=dict(color="#e5e7eb"))
    fig.update_yaxes(showgrid=True, gridcolor="#1f2937",
                     zeroline=False, linecolor="#374151",
                     tickfont=dict(color="#e5e7eb"), titlefont=dict(color="#e5e7eb"))
    return fig

# -------------------------
# Load data
# -------------------------
assert os.path.exists(DATA_FILE), f"Data file not found: {DATA_FILE}"
df = pd.read_csv(DATA_FILE)
rating_cols = [c for c in df.columns if c not in ["student_id","overall_satisfaction"]]

# -------------------------
# Sidebar (placeholder)
# -------------------------
st.sidebar.header("Filters")

# -------------------------
# KPIs
# -------------------------
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

overall_mean = df["overall_satisfaction"].mean()
overall_median = df["overall_satisfaction"].median()
overall_std = df["overall_satisfaction"].std()
cat_means = df[rating_cols].mean().sort_values(ascending=False)

c1, c2, c3, c4 = st.columns([1,1,1,1], gap="large")
c1.markdown(
    f'<div class="kpi"><div class="label">Overall mean</div>'
    f'<div class="value">{overall_mean:.2f}</div><div class="sub">Scale 1–10</div></div>',
    unsafe_allow_html=True
)
c2.markdown(
    f'<div class="kpi"><div class="label">Median</div>'
    f'<div class="value">{overall_median:.2f}</div><div class="sub">&nbsp;</div></div>',
    unsafe_allow_html=True
)
c3.markdown(
    f'<div class="kpi"><div class="label">Std dev</div>'
    f'<div class="value">{overall_std:.2f}</div><div class="sub">&nbsp;</div></div>',
    unsafe_allow_html=True
)
c4.markdown(
    f'<div class="kpi"><div class="label">Top category</div>'
    f'<div class="value">{cat_means.index[0].replace("_"," ")}</div>'
    f'<div class="sub">{cat_means.iloc[0]:.2f}</div></div>',
    unsafe_allow_html=True
)

# -------------------------
# Satisfaction bands (for donut)
# -------------------------
bands = pd.cut(
    df["overall_satisfaction"], bins=[0,4,6,8,10],
    labels=["Poor (1-4)","Fair (4-6)","Good (6-8)","Excellent (8-10)"],
    include_lowest=True
)
band_counts = bands.value_counts().reindex(
    ["Poor (1-4)","Fair (4-6)","Good (6-8)","Excellent (8-10)"]
).fillna(0)
band_df = band_counts.rename("count").to_frame()
band_df["pct"] = (band_counts/len(df)*100).round(1)

# -------------------------
# Friendly category labels
# -------------------------
pretty = {
    'well_versed_with_the_subject':'Subject knowledge',
    'explains_concepts_in_an_understandable_way':'Clarity',
    'use_of_presentations':'Presentations',
    'degree_of_difficulty_of_assignments':'Assignment difficulty',
    'solves_doubts_willingly':'Doubt support',
    'structuring_of_the_course':'Course structure',
    'provides_support_for_students_going_above_and_beyond':'Beyond support',
    'course_recommendation_based_on_relevance':'Relevance'
}
cat_series = cat_means.rename(index=lambda x: pretty.get(x, x))

# -------------------------
# Row: Category Averages + Satisfaction Donut
# -------------------------
a1, a2 = st.columns([1,1])

# Category averages bar
fig_bar = px.bar(
    cat_series.sort_values(ascending=True),
    orientation="h",
    labels={"value":"Average (1–10)", "index":"Category"},
    color_discrete_sequence=["#064e3b","#0d6d57","#148a70","#22a37f","#34d399"][:max(1,len(cat_series))]
)
fig_bar.update_traces(
    marker_line_color="#0f172a", marker_line_width=1.0,
    texttemplate='%{x:.2f}', textposition='outside',
    textfont=dict(color="#e5e7eb")
)
fig_bar.update_layout(showlegend=False)
fig_bar = style_fig(fig_bar, 460)
a1.subheader("Where Students Rate Instructors Highest")
a1.plotly_chart(fig_bar, use_container_width=True)

# Satisfaction levels donut
fig_donut = go.Figure(data=[go.Pie(
    labels=band_df.index,
    values=band_df["count"],
    hole=0.55,
    marker=dict(colors=["#ef4444","#22c55e","#f59e0b","#374151"]),
    textinfo="none"
)])
fig_donut.update_traces(hovertemplate="<b>%{label}</b><br>%{percent} (%{value})<extra></extra>")
fig_donut = style_fig(fig_donut, 460)
fig_donut.update_layout(
    annotations=[dict(text=f"N={len(df)}", x=0.5, y=0.5,
                      font=dict(color="#e5e7eb", size=14), showarrow=False)]
)
a2.subheader("Overall Satisfaction Mix (Poor → Excellent)")
a2.plotly_chart(fig_donut, use_container_width=True)

# -------------------------
# Correlation heatmap
# -------------------------
corr = df[rating_cols].corr().round(2)
corr_pretty = corr.rename(index=pretty, columns=pretty)

colorscale = ["#3b0a0a","#b2182b","#ef8a62","#f7f7f7","#67a9cf","#2166ac","#0a2b4a"]
fig_heat = px.imshow(corr_pretty, color_continuous_scale=colorscale, zmin=-0.35, zmax=0.35, aspect="auto")
fig_heat = style_fig(fig_heat, 600)
fig_heat.update_traces(text=None)
fig_heat.update_layout(xaxis_side="top")

annotations, shapes = [], []
z = corr_pretty.values
rows, cols = z.shape
pad_x, pad_y = 0.45, 0.45
for i in range(rows):
    for j in range(cols):
        val = z[i,j]
        if abs(val)<0.12:
            bg="rgba(11,18,32,0.65)"; fg="#e5e7eb"
        else:
            bg="rgba(229,231,235,0.80)"; fg="#0b1220"
        shapes.append(dict(type="rect", xref="x", yref="y",
                           x0=j-pad_x, x1=j+pad_x, y0=i-pad_y, y1=i+pad_y,
                           line=dict(width=0), fillcolor=bg, layer="above"))
        annotations.append(dict(x=j, y=i, text=f"{val:+.2f}" if i!=j else "1.00",
                                xref="x", yref="y", showarrow=False,
                                font=dict(color=fg, size=16, family="Segoe UI Semibold"),
                                align="center"))
fig_heat.update_layout(annotations=annotations, shapes=shapes,
                       coloraxis_colorbar=dict(tickfont=dict(color="#e5e7eb", size=12), title=None))
fig_heat.update_xaxes(tickvals=list(range(cols)), ticktext=corr_pretty.columns)
fig_heat.update_yaxes(tickvals=list(range(rows)), ticktext=corr_pretty.index)
fig_heat.update_traces(hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r=%{z:.2f}<extra></extra>")

st.subheader("Which Aspects Move Together (Correlation)")
st.plotly_chart(fig_heat, use_container_width=True)

# -------------------------
# Distribution of Ratings by Category (boxplots)
# -------------------------
melted = df.melt(value_vars=rating_cols, var_name="Category", value_name="Score")
melted["Category"] = melted["Category"].map(lambda x: pretty.get(x, x))

fig_box = px.box(melted, y="Category", x="Score", points="outliers",
                 color_discrete_sequence=["#34d399"])
fig_box.update_layout(xaxis_title="Rating (1–10)", yaxis_title=None)
fig_box = style_fig(fig_box, 520)

st.subheader("How Ratings Vary Across Course Aspects (Range, Median, Outliers)")
st.plotly_chart(fig_box, use_container_width=True)

# -------------------------
# PCA clusters (interactive scatter) — improved readability
# -------------------------
if os.path.exists(SEGMENTED_FILE):
    seg = pd.read_csv(SEGMENTED_FILE)

    # If the file doesn’t already have PC1/PC2, compute them (fallback)
    need_pca = not all(c in seg.columns for c in ["PC1","PC2"])
    if need_pca:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        rating_cols_seg = [c for c in seg.columns if c not in ["student_id","overall_satisfaction","cluster","PC1","PC2"]]
        X = seg[rating_cols_seg].values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(X_scaled)
        seg["PC1"], seg["PC2"] = pcs[:,0], pcs[:,1]

    if "cluster" not in seg.columns:
        # graceful fallback—assign a single cluster if missing
        seg["cluster"] = 0

    st.subheader("Student Cohorts in 2D (PCA Projection)")

    # High-contrast palette for dark backgrounds
    palette = ["#22c55e", "#f59e0b", "#60a5fa", "#ef4444", "#a78bfa", "#34d399", "#fb7185"]
    # Ensure enough colors
    n_unique = seg["cluster"].nunique()
    colors = palette[:max(n_unique, 3)]

    # Slight transparency and small white outline improve separation on dark
    fig_pca = px.scatter(
        seg, x="PC1", y="PC2", color="cluster",
        color_discrete_sequence=colors,
        opacity=0.85,
        hover_data={c: False for c in seg.columns},  # reduce noisy hover; keep only key fields below
    )

    # Marker size and border
    fig_pca.update_traces(
        marker=dict(size=8, line=dict(width=0.6, color="rgba(255,255,255,0.25)"))
    )

    # Clean axes, subtle grids for orientation
    fig_pca.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title_text="Cluster",
        xaxis=dict(showgrid=True, gridcolor="#1f2937", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1f2937", zeroline=False),
    )

    # Optional: add a light jitter to reduce exact-overplotting (toggle True/False)
    add_jitter = False
    if add_jitter:
        import numpy as np
        jitter = 0.02
        seg["_PC1j"] = seg["PC1"] + np.random.uniform(-jitter, jitter, size=len(seg))
        seg["_PC2j"] = seg["PC2"] + np.random.uniform(-jitter, jitter, size=len(seg))
        fig_pca = px.scatter(
            seg, x="_PC1j", y="_PC2j", color="cluster",
            color_discrete_sequence=colors, opacity=0.85,
        )
        fig_pca.update_traces(marker=dict(size=8, line=dict(width=0.6, color="rgba(255,255,255,0.25)")))
        fig_pca.update_layout(xaxis_title="PC1", yaxis_title="PC2", legend_title_text="Cluster")

    fig_pca = style_fig(fig_pca, 560)
    st.plotly_chart(fig_pca, use_container_width=True)
elif os.path.exists(PCA_IMG):
    st.subheader("Student Cohorts in 2D (PCA Projection)")
    st.image(PCA_IMG, use_column_width=True)


# -------------------------
# Sentiment Analysis (Synthetic)
# -------------------------
if os.path.exists(SENTIMENT_FILE):
    sf = pd.read_csv(SENTIMENT_FILE)
    if "sentiment_label" in sf.columns:
        st.subheader("What Students Felt (Synthetic Labeling)")
        order = ["Positive","Neutral","Negative"]
        counts = sf["sentiment_label"].value_counts().reindex(order).fillna(0).astype(int)

        tbl_sent = pd.DataFrame({
            "Sentiment": order,
            "Count": [int(counts[o]) for o in order],
            "Percent": [round(counts[o]/len(sf)*100,1) for o in order]
        })
        st.dataframe(tbl_sent, use_container_width=True)

        fig_sent_donut = go.Figure(data=[go.Pie(
            labels=order, values=[counts[o] for o in order], hole=0.55,
            marker=dict(colors=["#22c55e","#9ca3af","#ef4444"]),
            textinfo="none"
        )])
        fig_sent_donut = style_fig(fig_sent_donut, 420)
        fig_sent_donut.update_layout(
            annotations=[dict(text=f"N={len(sf)}", x=0.5, y=0.5,
                              font_size=14, font_color="#e5e7eb", showarrow=False)]
        )
        fig_sent_donut.update_traces(
            hovertemplate="<b>%{label}</b><br>%{percent} (%{value})<extra></extra>"
        )
        st.subheader("Sentiment Distribution")
        st.plotly_chart(fig_sent_donut, use_container_width=True)

        bar_df = pd.DataFrame({"Sentiment": order, "Count": [int(counts[o]) for o in order]})
        fig_sent_bar = px.bar(
            bar_df, y="Sentiment", x="Count", orientation="h",
            color="Sentiment",
            color_discrete_map={"Positive":"#22c55e","Neutral":"#9ca3af","Negative":"#ef4444"}
        )
        fig_sent_bar.update_layout(showlegend=False)
        fig_sent_bar = style_fig(fig_sent_bar, 380)
        st.subheader("Sentiment Counts by Class")
        st.plotly_chart(fig_sent_bar, use_container_width=True)

# -------------------------
# Word Cloud (live Matplotlib render on dark)
# -------------------------
if os.path.exists(COMMENTS_FILE):
    sfc = pd.read_csv(COMMENTS_FILE)
    if "comments" in sfc.columns:
        text_wc = " ".join(str(t) for t in sfc["comments"].dropna().astype(str))
        wc = WordCloud(
            width=1200, height=800, background_color="#0f172a", colormap="viridis",
            prefer_horizontal=0.95, collocations=False
        ).generate(text_wc)
        fig_wc, ax_wc = plt.subplots(figsize=(10,6), facecolor="#0f172a")
        ax_wc.imshow(wc, interpolation="bilinear"); ax_wc.axis("off")
        st.subheader("Top Themes in Open Feedback (Word Cloud)")
        st.pyplot(fig_wc, use_container_width=True)
        plt.close(fig_wc)
    elif os.path.exists(WORDCLOUD_IMG):
        st.subheader("Top Themes in Open Feedback (Word Cloud)")
        st.image(WORDCLOUD_IMG, use_column_width=True)
