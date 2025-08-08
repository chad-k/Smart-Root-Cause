
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("🔍 Smart Root Cause Ranking Dashboard with Notes & Actionability")

@st.cache_data
def load_data():
    return pd.read_csv("c:/users/ckaln/downloads/smart_root_cause_data.csv")

@st.cache_resource
def train_model(df):
    df_encoded = pd.get_dummies(df, columns=["Operator", "Tool", "MaterialLot", "Machine"], drop_first=True)
    X = df_encoded.drop(columns=["Defect"])
    y = df_encoded["Defect"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X.columns.tolist()

df = load_data()

# Filters
st.sidebar.header("📁 Filters")
selected_operator = st.sidebar.multiselect("Operator", df["Operator"].unique(), df["Operator"].unique())
selected_tool = st.sidebar.multiselect("Tool", df["Tool"].unique(), df["Tool"].unique())
selected_lot = st.sidebar.multiselect("MaterialLot", df["MaterialLot"].unique(), df["MaterialLot"].unique())
selected_machine = st.sidebar.multiselect("Machine", df["Machine"].unique(), df["Machine"].unique())

# Actionable toggle
show_actionable_only = st.sidebar.checkbox("Show Only Actionable Causes", value=False)

filtered_df = df[
    df["Operator"].isin(selected_operator) &
    df["Tool"].isin(selected_tool) &
    df["MaterialLot"].isin(selected_lot) &
    df["Machine"].isin(selected_machine)
]

st.write(f"Filtered dataset contains `{len(filtered_df)}` rows.")
st.dataframe(filtered_df.head())

model, X_test, y_test, feature_names = train_model(filtered_df)

# Classification Report
st.subheader("📋 Model Performance")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose().round(3))

# Feature Importance with Notes
st.subheader("📊 Ranked Root Cause Contributors")

importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance Score": importances
}).sort_values(by="Importance Score", ascending=False)

# Add Notes and Actionability
def get_notes_and_actionability(feature):
    f = feature.lower()
    if "operator" in f:
        return "Frequently appears in failed batches", "Yes"
    elif "tool" in f:
        return "Might need maintenance or replacement", "Yes"
    elif "materiallot" in f:
        return "May indicate supplier/batch issue", "Yes"
    elif "machine" in f:
        return "May require calibration", "Yes"
    elif "pressure" in f:
        return "Adjustable process parameter", "Yes"
    elif "speed" in f:
        return "Adjustable process parameter", "Yes"
    elif "temperature" in f:
        return "Check for consistency in heating/cooling", "Yes"
    else:
        return "Not easily actionable", "No"

importance_df["Feature Clean"] = (
    importance_df["Feature"]
    .str.replace("Operator_", "Operator: ")
    .str.replace("Tool_", "Tool: ")
    .str.replace("MaterialLot_", "MaterialLot: ")
    .str.replace("Machine_", "Machine: ")
)
importance_df[["Notes", "Actionable"]] = importance_df["Feature"].apply(
    lambda f: pd.Series(get_notes_and_actionability(f))
)

if show_actionable_only:
    importance_df = importance_df[importance_df["Actionable"] == "Yes"]

st.dataframe(
    importance_df[["Feature Clean", "Importance Score", "Notes", "Actionable"]]
    .rename(columns={"Feature Clean": "Feature"})
    .reset_index(drop=True)
    .head(10)
)
