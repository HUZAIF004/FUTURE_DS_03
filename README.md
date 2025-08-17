# 🎓 College Event Feedback Analysis

Analyze student feedback end-to-end: clean and prepare data, explore insights, run ANOVA and clustering, and present results in a polished Streamlit dashboard.

## 🔍 Overview
- Goal: Convert raw feedback into actionable insights and an interactive dashboard.
- Key questions:
  - Overall satisfaction and its distribution
  - Top and bottom categories
  - Correlations among aspects
  - Student segments from rating patterns
  - Indicative sentiment from synthetic comments
- Deliverables:
  - Clean CSVs in data/
  - Visual assets in assets/
  - Documented notebook
  - Streamlit app (app.py)

## 📁 Repository Structure (bullet list)
- app.py
- data/
  - student_feedback_clean.csv
  - student_feedback_segmented.csv
  - student_feedback_with_comments_synthetic.csv
  - student_feedback_with_sentiment.csv (optional)
  - kpi_summary.csv
- assets/
  - viz_overall_bands.png
  - viz_category_averages_enhanced.png
  - viz_corr_clean.png
  - viz_boxplots_clean.png
  - fig_wordcloud_synthetic.png
  - fig_clusters_pca.png
- notebooks/
  - College_feedback.ipynb
- requirements.txt

## ⚙️ Setup
- Python 3.9+ recommended
- Install dependencies:
  - pip install -r requirements.txt
- Example requirements:
  - pandas, numpy, seaborn, matplotlib, plotly, streamlit, scikit-learn, scipy, textblob, wordcloud
- Optional (TextBlob corpora):
  - python -m textblob.download_corpora

## 🧱 Data Pipeline (Phases)
- Phase 1: 📂 Load Dataset
- Phase 2: 🧹 Data Cleaning
- Phase 3: 🛠️ Data Preparation (tidy formats, standardized names)
- Phase 4: 🔎 Exploratory Data Analysis (EDA)
  - Satisfaction bands, category means, correlations, boxplots
- Phase 5: 🎯 Key Performance Indicators (KPIs)
- Phase 6: 🧪 Statistical Testing & Segmentation
  - One-way ANOVA across categories
  - KMeans clustering (k≈3) with PCA projection (2D)
- Extras:
  - Synthetic comments generator + sentiment labeling demo
  - Word cloud visualization

## ▶️ Run the Notebook
- Open notebooks/College_feedback.ipynb
- Execute cells in order
- Ensure outputs save to:
  - data/ for CSV artifacts
  - assets/ for figures

## 🖥️ Run the Streamlit App
- Required in data/:
  - student_feedback_clean.csv
- Optional (to enable features):
  - student_feedback_segmented.csv (PCA scatter)
  - student_feedback_with_comments_synthetic.csv (word cloud)
  - student_feedback_with_sentiment.csv (sentiment donut/table)
- Launch:
  - streamlit run app.py
- If a “Data file not found” message appears:
  - Verify files in data/
  - Adjust DATA_DIR/ASSETS paths at the top of app.py

## 📦 Outputs
- Data artifacts:
  - student_feedback_clean.csv (cleaned ratings + overall_satisfaction)
  - student_feedback_segmented.csv (cluster labels, optional PC1/PC2)
  - kpi_summary.csv (single-row KPI table)
  - student_feedback_with_comments_synthetic.csv (synthetic comments + optional sentiment)
- Figures:
  - viz_overall_bands.png, viz_category_averages_enhanced.png
  - viz_corr_clean.png, viz_boxplots_clean.png
  - fig_wordcloud_synthetic.png, fig_clusters_pca.png

## ⚠️ Notes & Limitations
- Synthetic comments are for demonstration only; do not treat as real feedback.
- ANOVA relies on assumptions (independence, similar variances).
- Clustering results depend on standardization, chosen k, and initialization; validate with silhouette and domain review.

## 🚀 Next Steps
- Replace synthetic comments with real text; build a robust NLP pipeline (cleaning, tokenization, topic modeling, transformer sentiment).
- Add dashboard filters (course, instructor, batch) and drilldowns.
- Add confidence intervals/bootstraps for KPIs; add silhouette scoring for clusters.
- Automate data refresh and deploy the dashboard.

## ✅ Quick Start
- Install: pip install -r requirements.txt
- Verify data files in data/
- Notebook: open notebooks/College_feedback.ipynb and run
- Dashboard: streamlit run app.py
