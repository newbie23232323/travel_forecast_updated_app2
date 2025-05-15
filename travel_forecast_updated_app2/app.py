import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import glob
import re

# ---------------------------
# Simulate & Load Data
# ---------------------------
@st.cache_data()

def load_data():
    np.random.seed(42)
    employees = [f"Employee {i+1}" for i in range(100)]
    employee_numbers = np.random.randint(1000000, 9999999, size=100)
    employee_map = dict(zip(employees, employee_numbers))

    quarters = pd.date_range(start='2020-01-01', end='2025-03-31', freq='QS')
    categories = ['Airfare', 'Lodging', 'Meals', 'Snacks', 'Ground Transport']
    expense_caps = {
        'Airfare': (300, 800),
        'Lodging': (150, 300),
        'Meals': (20, 30),
        'Snacks': (10, 30),
        'Ground Transport': (50, 150)
    }

    records = []
    for name in employees:
        emp_number = employee_map[name]
        for date in quarters:
            records.extend([
                {'Employee': name, 'EmployeeNumber': emp_number, 'Date': date, 'Category': cat, 'Location': 'Washington DC',
                 'Expense': round(np.random.uniform(*expense_caps[cat]) * (3 if cat in ['Lodging', 'Meals', 'Snacks'] else 2), 2)}
                for cat in categories
            ])

    violators = [f"Violation_User_{i+1}" for i in range(5)]
    for name in violators:
        emp_number = np.random.randint(1000000, 9999999)
        for date in quarters:
            for cat in categories:
                if cat in ['Meals', 'Snacks']:
                    over_limit = expense_caps[cat][1] + np.random.uniform(20, 50)
                else:
                    over_limit = np.random.uniform(*expense_caps[cat]) * (3 if cat in ['Lodging', 'Meals', 'Snacks'] else 2)
                records.append({
                    'Employee': name,
                    'EmployeeNumber': emp_number,
                    'Date': date,
                    'Category': cat,
                    'Location': 'Washington DC',
                    'Expense': round(over_limit, 2)
                })

    return pd.DataFrame(records)

# ---------------------------
# Detect Violations
# ---------------------------
def detect_violations(df):
    max_limits = {
        'Meals': 30,
        'Snacks': 30
    }
    violations = df[df['Category'].isin(max_limits.keys()) & (df['Expense'] > df['Category'].map(max_limits))]
    return violations[['Employee', 'EmployeeNumber', 'Date', 'Category', 'Expense']]

# ---------------------------
# Suggest Cost Saving Tips (Trend-Based)
# ---------------------------
def suggest_cost_saving_tips(df):
    st.subheader("Suggested Ways to Reduce Travel Expenses")
    suggestions = []
    df['Quarter'] = df['Date'].dt.to_period('Q')
    grouped = df.groupby(['Quarter', 'Category']).agg({"Expense": "mean"}).reset_index()
    trend_summary = grouped.groupby('Category').apply(lambda x: x.sort_values('Quarter')['Expense'].pct_change().mean()).sort_values(ascending=False)

    for category, avg_trend in trend_summary.items():
        if avg_trend > 0.05:
            suggestions.append(f"{category}: Rising trend. Consider reviewing policies or negotiating rates.")
        else:
            suggestions.append(f"{category}: Stable trend. No immediate changes needed.")

    for tip in suggestions:
        st.markdown(f"- {tip}")

# ---------------------------
# Simple NumPy-based Linear Regression Forecast
# ---------------------------
def run_numpy_linear_regression(df, category):
    df = df[df['Category'] == category].copy()
    df['Quarter'] = df['Date'].dt.to_period('Q').apply(lambda x: x.start_time.toordinal())
    df = df.groupby('Quarter')['Expense'].mean().reset_index()
    X = df['Quarter'].values.reshape(-1, 1)
    y = df['Expense'].values

    m, b = np.polyfit(X.flatten(), y, 1)
    future_quarters = pd.date_range(start='2025-04-01', periods=8, freq='QS')
    future_ordinals = np.array([d.toordinal() for d in future_quarters]).reshape(-1, 1)
    predictions = m * future_ordinals.flatten() + b

    forecast_df = pd.DataFrame({
        'Quarter': future_quarters,
        'Predicted Expense': predictions.round(2)
    })
    return forecast_df

# ---------------------------
# Parse Natural Language Query
# ---------------------------
def parse_query(text):
    adjustments = {
        "airfare": 0,
        "lodging": 0,
        "meals": 0,
        "snacks": 0,
        "ground": 0
    }
    matches = re.findall(r'(airfare|lodging|meals|snacks|ground transport)[^\d]*(\d+)%?', text.lower())
    for category, percent in matches:
        key = category.replace(" ", "") if category != "ground transport" else "ground"
        adjustments[key] = int(percent)
    return adjustments

# ---------------------------
# Generate Summary Report with Anomalies and Filters
# ---------------------------
def generate_expense_summary(df):
    df['Quarter'] = df['Date'].dt.to_period('Q')
    selected_category = st.selectbox("Filter by Category:", options=["All"] + sorted(df['Category'].unique()))
    selected_quarters = st.multiselect("Select Quarters:", options=sorted(df['Quarter'].unique().astype(str)))

    filtered_df = df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    if selected_quarters:
        filtered_df = filtered_df[filtered_df['Quarter'].astype(str).isin(selected_quarters)]

    pivot = filtered_df.pivot_table(index='Quarter', columns='Category', values='Expense', aggfunc='mean').fillna(0)
    st.subheader("Visual Expense Trend Analysis")
    st.line_chart(pivot)

    ### Anomaly detection: flag quarters with sudden >10% jump ###
    st.subheader("Detected Anomalies")
    anomalies = []
    for cat in pivot.columns:
        diffs = pivot[cat].pct_change()
        for i, pct in enumerate(diffs):
            if pct is not None and abs(pct) > 0.10:
                anomalies.append(f"{cat} had a {'rise' if pct > 0 else 'drop'} of {pct:.1%} in {pivot.index[i]}")
    for alert in anomalies:
        st.warning(alert)

    st.subheader("Download Summary Report (CSV)")
    summary_csv = pivot.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Summary Report",
        data=summary_csv,
        file_name="expense_summary_report.csv",
        mime="text/csv"
    )

# ---------------------------
# Inflation Preset UI
# ---------------------------
def inflation_controls():
    st.sidebar.header("Inflation Presets")
    presets = {
        "Baseline (0%)": {"airfare": 0, "lodging": 0, "meals": 0, "snacks": 0, "ground": 0},
        "Moderate (10%)": {"airfare": 10, "lodging": 10, "meals": 10, "snacks": 10, "ground": 10},
        "Aggressive (20%)": {"airfare": 20, "lodging": 20, "meals": 20, "snacks": 20, "ground": 20}
    }
    selected = st.sidebar.selectbox("Choose a preset:", list(presets.keys()))
    values = presets[selected]

    st.sidebar.subheader("Adjust Inflation Rates for 2026")
    factors = {
        "airfare": st.sidebar.slider("Airfare Inflation (%)", 0, 50, values["airfare"]),
        "lodging": st.sidebar.slider("Lodging Inflation (%)", 0, 50, values["lodging"]),
        "meals": st.sidebar.slider("Meals Inflation (%)", 0, 50, values["meals"]),
        "snacks": st.sidebar.slider("Snacks Inflation (%)", 0, 50, values["snacks"]),
        "ground": st.sidebar.slider("Ground Transport Inflation (%)", 0, 50, values["ground"])
    }
    return factors

# ---------------------------
# Streamlit App Main Section
# ---------------------------
def main():
    inflation = inflation_controls()
    st.title("Travel Expense Forecasting Tool")
    df = load_data()

    st.subheader("Forecast for a Selected Category")
    category = st.selectbox("Select a category to forecast:", df['Category'].unique())
    forecast_df = run_numpy_linear_regression(df, category)
    st.dataframe(forecast_df)
    st.line_chart(forecast_df.set_index(pd.to_datetime(forecast_df['Quarter'])))

    suggest_cost_saving_tips(df)
    generate_expense_summary(df)

    st.subheader("Expense Violations Report")
    violations_df = detect_violations(df)
    st.dataframe(violations_df)

# ---------------------------
# Run App
# ---------------------------
if __name__ == '__main__':
    main()
