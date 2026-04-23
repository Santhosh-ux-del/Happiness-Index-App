import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Happiness Dashboard",
    page_icon="😊",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("df_cleaned.csv")
    df.columns = df.columns.str.strip()          # remove leading/trailing spaces
    return df

df = load_data()

DEPENDENT   = "Happiness Score"
INDEPENDENT = [c for c in df.columns if c not in ["Country", DEPENDENT]]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Exploratory Analysis", "🤖 Prediction Model", "🌍 Country Explorer"],
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("😊 World Happiness Score Dashboard")
    st.markdown("Explore the factors that drive happiness across **129 countries**.")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Countries",  len(df))
    c2.metric("Avg Happiness",    f"{df[DEPENDENT].mean():.3f}")
    c3.metric("Happiest Country", df.loc[df[DEPENDENT].idxmax(), 'Country'])
    c4.metric("Least Happy",      df.loc[df[DEPENDENT].idxmin(), 'Country'])

    st.divider()

    # Top / Bottom 10
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 10 Happiest Countries")
        top10 = df.nlargest(10, DEPENDENT)[["Country", DEPENDENT]]
        fig = px.bar(top10, x=DEPENDENT, y="Country", orientation="h",
                     color=DEPENDENT, color_continuous_scale="Greens",
                     text_auto=".3f")
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                          height=400, margin=dict(l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 Bottom 10 Countries")
        bot10 = df.nsmallest(10, DEPENDENT)[["Country", DEPENDENT]]
        fig = px.bar(bot10, x=DEPENDENT, y="Country", orientation="h",
                     color=DEPENDENT, color_continuous_scale="Reds_r",
                     text_auto=".3f")
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                          height=400, margin=dict(l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("📋 Full Dataset")
    st.dataframe(df.sort_values(DEPENDENT, ascending=False).reset_index(drop=True),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")

    # Distribution of Happiness Score
    st.subheader("Distribution of Happiness Score")
    fig = px.histogram(df, x=DEPENDENT, nbins=25, marginal="box",
                       color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df[INDEPENDENT + [DEPENDENT]]
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, annot_kws={"size": 10})
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    st.pyplot(fig)

    st.divider()

    # Scatter: choose any independent var
    st.subheader("Scatter Plot vs Happiness Score")
    chosen = st.selectbox("Select independent variable", INDEPENDENT)
    fig = px.scatter(df, x=chosen, y=DEPENDENT, hover_name="Country",
                     trendline="ols", color=DEPENDENT,
                     color_continuous_scale="Viridis",
                     labels={chosen: chosen, DEPENDENT: DEPENDENT})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Pair plot (static)
    st.subheader("Pairwise Relationships (sample features)")
    pair_cols = st.multiselect(
        "Choose features for pair plot",
        INDEPENDENT,
        default=INDEPENDENT[:3],
    )
    if len(pair_cols) >= 2:
        pair_df = df[pair_cols + [DEPENDENT]]
        fig = px.scatter_matrix(pair_df, dimensions=pair_cols, color=DEPENDENT,
                                color_continuous_scale="Viridis",
                                title="Scatter Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least 2 features.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Prediction Model":
    st.title("🤖 Linear Regression Model")

    # Feature selection
    selected_features = st.multiselect(
        "Select independent variables to include in the model",
        INDEPENDENT,
        default=INDEPENDENT,
    )

    test_size = st.slider("Test set size (%)", 10, 40, 20, step=5)

    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
    else:
        X = df[selected_features].values
        y = df[DEPENDENT].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score",  f"{r2:.4f}")
        m2.metric("RMSE",      f"{rmse:.4f}")
        m3.metric("Train size", f"{len(X_train)} rows")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actual vs Predicted")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                     marker=dict(color="#636EFA", size=8, opacity=0.7),
                                     name="Predictions"))
            lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                     line=dict(color="red", dash="dash"), name="Perfect fit"))
            fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Feature Coefficients")
            coef_df = pd.DataFrame({
                "Feature":     selected_features,
                "Coefficient": model.coef_,
            }).sort_values("Coefficient", key=abs, ascending=False)
            fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                         color="Coefficient", color_continuous_scale="RdBu",
                         text_auto=".3f")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Live prediction form
        st.subheader("🔮 Predict Happiness Score")
        st.markdown("Adjust the sliders to predict a custom happiness score.")
        user_vals = []
        cols = st.columns(min(len(selected_features), 3))
        for i, feat in enumerate(selected_features):
            col = cols[i % len(cols)]
            lo_v = float(df[feat].min())
            hi_v = float(df[feat].max())
            default = float(df[feat].mean())
            val = col.slider(feat, lo_v, hi_v, default, step=(hi_v - lo_v) / 100)
            user_vals.append(val)

        user_arr = np.array(user_vals).reshape(1, -1)
        user_arr_sc = scaler.transform(user_arr)
        prediction = model.predict(user_arr_sc)[0]
        st.success(f"### 🎯 Predicted Happiness Score: **{prediction:.3f}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COUNTRY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Country Explorer":
    st.title("🌍 Country Explorer")

    country = st.selectbox("Select a country", sorted(df["Country"].unique()))
    row = df[df["Country"] == country].iloc[0]

    st.subheader(f"**{country}** — Happiness Score: {row[DEPENDENT]:.3f}")

    # Radar chart
    categories = INDEPENDENT
    vals = [row[c] for c in categories]

    # Normalize to 0-1 for radar
    norm_vals = [(row[c] - df[c].min()) / (df[c].max() - df[c].min()) for c in categories]

    fig = go.Figure(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line_color="#636EFA",
        fillcolor="rgba(99,110,250,0.3)",
        name=country,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"{country} — Normalised Feature Profile",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Feature Values")
    display = pd.DataFrame({
        "Feature": categories,
        "Value":   [round(row[c], 4) for c in categories],
        "Min (all countries)": [round(df[c].min(), 4) for c in categories],
        "Max (all countries)": [round(df[c].max(), 4) for c in categories],
        "Mean (all countries)": [round(df[c].mean(), 4) for c in categories],
    })
    st.dataframe(display, use_container_width=True)

    # Compare two countries
    st.divider()
    st.subheader("Compare with another country")
    other = st.selectbox("Select another country",
                         [c for c in sorted(df["Country"].unique()) if c != country])
    other_row = df[df["Country"] == other].iloc[0]

    norm_other = [(other_row[c] - df[c].min()) / (df[c].max() - df[c].min()) for c in categories]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]], theta=categories + [categories[0]],
        fill="toself", name=country, line_color="#636EFA",
        fillcolor="rgba(99,110,250,0.25)",
    ))
    fig2.add_trace(go.Scatterpolar(
        r=norm_other + [norm_other[0]], theta=categories + [categories[0]],
        fill="toself", name=other, line_color="#EF553B",
        fillcolor="rgba(239,85,59,0.25)",
    ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"{country} vs {other}",
        height=450,
    )
    st.plotly_chart(fig2, use_container_width=True)
