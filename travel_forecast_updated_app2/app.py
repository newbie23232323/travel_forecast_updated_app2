import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import glob
import re

# ---------------------------
# Simulate or Load Data
# ---------------------------
@st.cache_data

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

    # Add 5 users that intentionally violate meal & snack limits
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

def run_numpy_linear_regression(df, category):
    df = df[df['Category'] == category].copy()
    df['Quarter'] = df['Date'].dt.to_period('Q').apply(lambda x: x.start_time.toordinal())
    df = df.groupby('Quarter')['Expense'].mean().reset_index()
    X = df['Quarter'].values.reshape(-1, 1)
    y = df['Expense'].values

    # Manual linear regression: y = mX + b
    m, b = np.polyfit(X.flatten(), y, 1)
    future_quarters = pd.date_range(start='2025-04-01', periods=8, freq='QS')
    future_ordinals = np.array([d.toordinal() for d in future_quarters]).reshape(-1, 1)
    predictions = m * future_ordinals.flatten() + b

    forecast_df = pd.DataFrame({
        'Quarter': future_quarters.to_period('Q').astype(str),
        'Predicted Expense': predictions.round(2)
    })
    return forecast_df


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
# Train Linear Regression Model
# ---------------------------
def train_model(df):
    df['Quarter'] = df['Date'].dt.to_period('Q')
    grouped = df.groupby(['Quarter', 'Category']).agg({'Expense': 'sum'}).reset_index()
    grouped['QuarterIndex'] = grouped['Quarter'].astype(str).apply(lambda x: int(x[:4]) * 4 + int(x[-1]) - 1)

    X = grouped[['QuarterIndex', 'Category']]
    y = grouped['Expense']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category'])
    ], remainder='passthrough')

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X, y)
    return model, grouped

# ---------------------------
# Forecast Future Quarters
# ---------------------------
def forecast_future(model, inflation_factors):
    future_quarters = pd.period_range(start='2025Q2', end='2026Q4', freq='Q')
    categories = ['Airfare', 'Lodging', 'Meals', 'Snacks', 'Ground Transport']
    future_data = []
    for q in future_quarters:
        quarter_index = int(q.strftime('%Y')) * 4 + int(q.quarter) - 1
        for category in categories:
            future_data.append({'Quarter': q, 'QuarterIndex': quarter_index, 'Category': category})
    future_df = pd.DataFrame(future_data)
    X_future = future_df[['QuarterIndex', 'Category']]
    future_df['Predicted_Expense'] = model.predict(X_future)

    for category, factor in inflation_factors.items():
        future_df.loc[(future_df['Quarter'].astype(str).str.startswith('2026')) &
                      (future_df['Category'] == category), 'Predicted_Expense'] *= factor

    return future_df

# ---------------------------
# Visualization
# ---------------------------
def plot_forecast(actual, forecast):
    actual['Date'] = actual['Quarter'].dt.to_timestamp()
    forecast['Date'] = forecast['Quarter'].dt.to_timestamp()
    plt.figure(figsize=(12, 6))
    for cat in actual['Category'].unique():
        a = actual[actual['Category'] == cat]
        f = forecast[forecast['Category'] == cat]
        plt.plot(a['Date'], a['Expense'], label=f"{cat} Actual", marker='o')
        plt.plot(f['Date'], f['Predicted_Expense'], label=f"{cat} Forecast", linestyle='--')
    plt.title("Quarterly Travel Expenses: Actual vs Forecast")
    plt.xlabel("Quarter")
    plt.ylabel("Total Expense ($)")
    plt.legend()
    st.pyplot(plt)

# ---------------------------
# Save and Load Presets
# ---------------------------
def save_preset(name, data):
    with open(f"{name}_preset.json", "w") as f:
        json.dump(data, f)

def load_preset(name):
    try:
        with open(f"{name}_preset.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def list_presets():
    return [os.path.splitext(os.path.basename(f))[0].replace('_preset', '') for f in glob.glob("*_preset.json")]

# ---------------------------
# Streamlit App UI
# ---------------------------

st.title("Travel Expense Forecasting Tool")

st.sidebar.header("Inflation Presets")
preset = st.sidebar.selectbox("Choose a scenario", ["Custom", "Baseline (0%)", "Conservative (10%)", "High Cost (20%)"])

saved_presets = list_presets()
selected_saved_preset = st.sidebar.selectbox("Load a saved preset", saved_presets)
if st.sidebar.button("Load Selected Preset"):
    loaded = load_preset(selected_saved_preset)
    if loaded:
        for k, v in loaded.items():
            st.session_state[k] = v
        st.sidebar.success(f"Loaded preset: {selected_saved_preset}")
    else:
        st.sidebar.error("Preset not found.")

default_values = {
    "Custom": {"airfare": 10, "lodging": 0, "meals": 0, "snacks": 0, "ground": 0},
    "Baseline (0%)": {"airfare": 0, "lodging": 0, "meals": 0, "snacks": 0, "ground": 0},
    "Conservative (10%)": {"airfare": 10, "lodging": 10, "meals": 10, "snacks": 10, "ground": 10},
    "High Cost (20%)": {"airfare": 20, "lodging": 20, "meals": 20, "snacks": 20, "ground": 20}
}

if preset != "Custom":
    for k, v in default_values[preset].items():
        st.session_state[k] = v

if st.sidebar.button("Reset Inflation Settings"):
    for k, v in default_values["Custom"].items():
        st.session_state[k] = v

st.sidebar.header("Adjust Inflation Rates for 2026")
airfare_factor = st.sidebar.slider("Airfare Inflation (%)", 0, 50, st.session_state.get("airfare", 10)) / 100 + 1
lodging_factor = st.sidebar.slider("Lodging Inflation (%)", 0, 50, st.session_state.get("lodging", 0)) / 100 + 1
meals_factor = st.sidebar.slider("Meals Inflation (%)", 0, 50, st.session_state.get("meals", 0)) / 100 + 1
snacks_factor = st.sidebar.slider("Snacks Inflation (%)", 0, 50, st.session_state.get("snacks", 0)) / 100 + 1
ground_factor = st.sidebar.slider("Ground Transport Inflation (%)", 0, 50, st.session_state.get("ground", 0)) / 100 + 1

st.sidebar.subheader("Manage Custom Preset")
custom_preset_name = st.sidebar.text_input("Preset Name")
if st.sidebar.button("Save Preset") and custom_preset_name:
    save_preset(custom_preset_name, {
        "airfare": int((airfare_factor - 1) * 100),
        "lodging": int((lodging_factor - 1) * 100),
        "meals": int((meals_factor - 1) * 100),
        "snacks": int((snacks_factor - 1) * 100),
        "ground": int((ground_factor - 1) * 100)
    })
    st.sidebar.success("Preset saved!")

st.session_state.airfare = int((airfare_factor - 1) * 100)
st.session_state.lodging = int((lodging_factor - 1) * 100)
st.session_state.meals = int((meals_factor - 1) * 100)
st.session_state.snacks = int((snacks_factor - 1) * 100)
st.session_state.ground = int((ground_factor - 1) * 100)

st.sidebar.subheader("Inflation Summary")
st.sidebar.write(pd.DataFrame({
    "Category": ["Airfare", "Lodging", "Meals", "Snacks", "Ground Transport"],
    "Inflation (%)": [
        st.session_state.airfare,
        st.session_state.lodging,
        st.session_state.meals,
        st.session_state.snacks,
        st.session_state.ground
    ]
}))

df = load_data()
model, actual = train_model(df)

def_inflation = {
    'Airfare': airfare_factor,
    'Lodging': lodging_factor,
    'Meals': meals_factor,
    'Snacks': snacks_factor,
    'Ground Transport': ground_factor
}

forecast = forecast_future(model, def_inflation)

st.subheader("Forecast Table (2025–2026)")
st.dataframe(forecast[['Quarter', 'Category', 'Predicted_Expense']].round(2))

st.subheader("Forecast Visualization")
plot_forecast(actual, forecast)

st.caption("Inflation adjustments for 2026 are configurable in the sidebar. Use the reset button or select a preset to get started.")

# ---------------------------
# Natural Language Query Input
# ---------------------------
st.subheader("Try a Natural Language Query")
query = st.text_input("Describe your inflation changes (e.g., 'Raise airfare by 12% and lodging by 8%')")
if st.button("Apply Query") and query:
    adjustments = parse_query(query)
    for key in adjustments:
        st.session_state[key] = adjustments[key]
    st.success("Applied inflation changes from query.")

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
# Violation Detection Output
# ---------------------------
st.subheader("Expense Violations Report")
df = load_data()
violations_df = detect_violations(df)
st.dataframe(violations_df)

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
