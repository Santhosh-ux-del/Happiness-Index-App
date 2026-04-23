# 😊 World Happiness Score Dashboard

An interactive Streamlit app to explore and predict World Happiness Scores using key socio-economic indicators.

## 📊 Features

- **Overview** — KPI cards, top/bottom 10 countries, full dataset table
- **Exploratory Analysis** — Distribution plots, correlation heatmap, scatter plots, pairwise matrix
- **Prediction Model** — Linear Regression with live slider-based prediction
- **Country Explorer** — Radar charts and side-by-side country comparisons

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run happiness_app.py
```

## 📁 Dataset

`df_cleaned.csv` — 129 countries with the following columns:

| Column | Role |
|---|---|
| Happiness Score | **Dependent variable** |
| GDP per capita | Independent |
| Social support | Independent |
| Healthy life expectancy | Independent |
| Freedom to make life choices | Independent |
| Generosity | Independent |
| Perceptions of corruption | Independent |

## 🌐 Live App

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud)
