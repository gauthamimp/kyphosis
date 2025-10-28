
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE

st.title("Kyphosis Disease Prediction & Severity Analysis")

# 1. Load Dataset & Expand with SMOTE
data = pd.read_csv("kyphosis.csv")
data['Kyphosis'] = data['Kyphosis'].map({'present':1, 'absent':0})
X = data.drop('Kyphosis', axis=1)
y = data['Kyphosis']

# Apply SMOTE to balance dataset
sm = SMOTE(sampling_strategy={0: 64, 1: 136}, random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Load saved models
rf = joblib.load("rf_model.pkl")
gb = joblib.load("gb_model.pkl")
stacking_model = joblib.load("stacking_model.pkl")

# 2. Dataset Info
st.write("### Dataset Info")
st.write("Original shape:", X.shape)
st.write("Class distribution:")
st.table(y.value_counts())

st.write("### Expanded Dataset Info (after SMOTE)")
st.write("Expanded shape:", X_res.shape)
st.write("Class distribution after SMOTE:")
st.table(y_res.value_counts())

# 3. User Input
st.write("### Enter Patient Details")
age = st.number_input("Age:", min_value=0, max_value=200, value=10)
number = st.number_input("Number of vertebrae involved:", min_value=1, max_value=10, value=3)
start = st.number_input("Start vertebra:", min_value=0, max_value=20, value=5)

input_df = pd.DataFrame([[age, number, start]], columns=['Age', 'Number', 'Start'])

# 4. Prediction, Severity & Feedback

if st.button("Predict Kyphosis"):
    #  Prediction 
    pred = stacking_model.predict(input_df)[0]
    prob = stacking_model.predict_proba(input_df)[0][pred]
    label = "Present" if pred==1 else "Absent"
    st.subheader(f"Kyphosis Prediction: {label} (Probability: {prob:.2f})")

    #SHAP Feature Contributions
    def shap_percentage(model, input_df, pred_class):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            values = np.array(shap_values[1][0] if pred_class==1 else shap_values[0][0]).flatten()
        else:
            values = np.array(shap_values).flatten()
        values = values[:len(input_df.columns)]
        abs_values = np.abs(values)
        percentages = 100 * abs_values / abs_values.sum()
        return percentages

    rf_pct = shap_percentage(rf, input_df, pred)
    gb_pct = shap_percentage(gb, input_df, pred)

    # Stacking Ensemble Contributions
    final_weights = stacking_model.final_estimator_.coef_[0]
    stacking_raw = rf_pct*final_weights[0] + gb_pct*final_weights[1]
    stacking_abs = np.abs(stacking_raw)
    stacking_pct = 100 * stacking_abs / stacking_abs.sum()

    merged_table = pd.DataFrame({
        "Feature": input_df.columns,
        "Value": input_df.iloc[0].values,
        "RF Contribution (%)": np.round(rf_pct,2),
        "GB Contribution (%)": np.round(gb_pct,2),
        "Stacking Contribution (%)": np.round(stacking_pct,2)
    })

    #  Top feature
    top_feature = merged_table.iloc[stacking_pct.argmax()]['Feature']

    def highlight_top(s):
        return ['background-color: gold; font-weight: bold' if v==top_feature else '' for v in s]

    st.subheader("Feature Contributions Comparison (%)")
    st.dataframe(merged_table.style.apply(highlight_top, subset=['Feature']))

    #  Horizontal Bar Chart 
    st.subheader("Feature Contributions Visualization (%)")
    fig = go.Figure()
    for model_name, col, color in zip(
        ['RF', 'GB', 'Stacking'],
        ['RF Contribution (%)', 'GB Contribution (%)', 'Stacking Contribution (%)'],
        ['skyblue', 'lightgreen', 'salmon']
    ):
        bar_colors = ['gold' if f == top_feature else c for f, c in zip(merged_table['Feature'], [color]*len(merged_table))]
        fig.add_trace(go.Bar(
            y=merged_table['Feature'],
            x=merged_table[col],
            name=model_name,
            orientation='h',
            marker_color=bar_colors,
            hovertemplate='%{y}: %{x:.2f}% (' + model_name + ')<extra></extra>'
        ))

    fig.update_layout(
        barmode='group',
        xaxis_title="Contribution (%)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=100, r=20, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

          #  Cobb Angle & Severity Scale 
    if pred == 0:
        severity = 0
        severity_label = "Normal"
        cob_angle = 0
    else:
        # Cobb angle calculation
        base_cobb = 5
        start_factor = 0.5
        age_factor = 0.1
        cob_angle = number * base_cobb + start * start_factor + age * age_factor

        # Map Cobb angle to 0–10 severity and label
        if cob_angle <= 20:
            severity = cob_angle / 20 * 3
            severity_label = "Normal"
        elif cob_angle <= 40:
            severity = 3 + (cob_angle - 20) / 20 * 4
            severity_label = "Medium"
        else:
            severity = min(7 + (cob_angle - 40) / 20 * 3, 10)
            severity_label = "High"

    #  Display severity
    st.subheader("Severity Scale (0–10)")
    st.progress(severity / 10)
    st.write(f"**Severity:** {severity_label} ({severity:.1f} / 10)")
    if pred != 0:
        st.write(f"Approximate Cobb angle: {cob_angle:.1f}°")


    #  XAI-informed Feedback 
    st.subheader("Feedback & Recommendations (XAI-driven)")
    if pred == 0:
        st.write("No Kyphosis detected. Maintain good posture and regular exercise.")
    else:
        if top_feature == "Age":
            st.write("- Age is significant; monitor posture and consider age-appropriate back exercises.")
        elif top_feature == "Number":
            st.write("- Number of affected vertebrae is significant; focus on targeted back-strengthening exercises.")
        elif top_feature == "Start":
            st.write("- Start vertebra is significant; maintain proper posture and avoid slouching.")

        st.write("- Maintain proper posture while sitting and standing.")
        st.write("- Perform back-strengthening exercises daily.")
        st.write("- Avoid prolonged slouching or heavy lifting.")
        st.write("- Consider physiotherapy if symptoms persist.")
        st.write("💡 *Summary Recommendation:* Focus on the highlighted feature to reduce Kyphosis risk.")
