# 🎓 College Event Feedback Analysis

A compact, end-to-end project to analyze student feedback, quantify satisfaction, explore correlations, generate synthetic comments for demo sentiment, segment students via clustering, and deliver an interactive Streamlit dashboard.

---

## Contents

- Project Overview  
- Repository Structure  
- Setup and Installation  
- Data Pipeline (Phases)  
- Running the Analysis  
- Running the Streamlit App  
- Outputs and Artifacts  
- Notes, Limitations, and Next Steps  

---

## Project Overview

**Objective:** Turn raw student feedback into actionable insights and an interactive dashboard.

**Key Questions:**

- What is the overall satisfaction level and distribution?  
- Which categories are top/bottom rated?  
- Which aspects move together?  
- What are the indicative sentiments in comments? (synthetic demo)  
- Are there distinct student cohorts based on rating patterns?

**Deliverables:**

- Cleaned datasets in `data/`  
- Visual assets in `assets/`  
- Jupyter notebook with documented phases  
- Streamlit app (`app.py`)  

---
# 🎓 College Event Feedback Analysis

A compact, end-to-end project to analyze student feedback, quantify satisfaction, explore correlations, generate synthetic comments for demo sentiment, segment students via clustering, and deliver an interactive Streamlit dashboard.

---

## Contents

- Project Overview  
- Repository Structure  
- Setup and Installation  
- Data Pipeline (Phases)  
- Running the Analysis  
- Running the Streamlit App  
- Outputs and Artifacts  
- Notes, Limitations, and Next Steps  

---

## Project Overview

**Objective:** Turn raw student feedback into actionable insights and an interactive dashboard.

**Key Questions:**

- What is the overall satisfaction level and distribution?  
- Which categories are top/bottom rated?  
- Which aspects move together?  
- What are the indicative sentiments in comments? (synthetic demo)  
- Are there distinct student cohorts based on rating patterns?

**Deliverables:**

- Cleaned datasets in `data/`  
- Visual assets in `assets/`  
- Jupyter notebook with documented phases  
- Streamlit app (`app.py`)  

---

## Repository Structure

app.py # Streamlit dashboard (dark theme, Plotly-native visuals)
data/
student_feedback.csv # raw, optional
student_feedback_clean.csv
student_feedback_tidy.csv
student_feedback_with_comments_synthetic.csv
student_feedback_segmented.csv
kpi_summary.csv
assets/
viz_overall_bands.png
viz_category_averages_enhanced.png
viz_corr_clean.png
viz_boxplots_clean.png
fig_wordcloud_synthetic.png
fig_clusters_pca.png
notebooks/
College_feedback.ipynb # documented with Markdown sections
requirements.txt

## Repository Structure

