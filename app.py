import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector as SFS
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

st.set_page_config(page_title="Loan Approval Classifier", layout="wide")
st.title("🏦 Loan Approval Classifier")
st.markdown("Upload your loan dataset and run the full ML pipeline: preprocessing → dimensionality reduction → SVM training.")

# ── 1. File Upload ──────────────────────────────────────────────────────────
st.header("1. Upload Dataset")

df = pd.read_csv(r"Loan.csv")
df = pd.DataFrame(df)

st.subheader("Data Preview")
st.dataframe(df.head())
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("Dataset Info"):
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    st.text(buf.getvalue())

# ── 2. Train / Test Split ────────────────────────────────────────────────────
st.header("2. Train / Test Split")

X = df.drop(["LoanApproved", "ApplicationDate"], axis=1)
y = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

col1, col2 = st.columns(2)
col1.metric("Train samples", X_train.shape[0])
col2.metric("Test samples", X_test.shape[0])

with st.expander("Training set descriptive statistics"):
    st.dataframe(X_train.describe())

# ── 3. EDA ───────────────────────────────────────────────────────────────────
st.header("3. Exploratory Data Analysis")

# Class distribution
st.subheader("Loan Approval Distribution (Train)")
fig, ax = plt.subplots()
sns.countplot(x=y_train, ax=ax)
ax.set_title("Loan Approval Distribution")
st.pyplot(fig)
plt.close()

# Numeric distributions
st.subheader("Feature Distributions")
X_train.columns = X_train.columns.str.strip()
cols = [c for c in ["Age", "AnnualIncome", "Experience", "LoanAmount", "MonthlyIncome", "CreditScore"] if c in X_train.columns]

if cols:
    n = len(cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(X_train[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution for {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Correlation heatmap
st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(X_train.corr(numeric_only=True), cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix")
st.pyplot(fig)
plt.close()

# Employment vs Approval
if "EmploymentStatus" in X_train.columns:
    st.subheader("Employment Status vs Loan Approval")
    fig, ax = plt.subplots()
    sns.countplot(x=X_train["EmploymentStatus"], hue=y_train, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Employment vs Approval")
    st.pyplot(fig)
    plt.close()

# Missing values
st.subheader("Missing Values")
missing = X_train.isnull().sum()
st.dataframe(missing[missing > 0].rename("Missing Count") if missing.any() else pd.Series({"No missing values": 0}))

# ── 4. Preprocessing ─────────────────────────────────────────────────────────
st.header("4. Preprocessing")

with st.spinner("Removing outliers…"):
    for c in [c for c in ["AnnualIncome", "MonthlyIncome", "LoanAmount"] if c in X_train.columns]:
        Q1 = X_train[c].quantile(0.25)
        Q3 = X_train[c].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_train = X_train[(X_train[c] > lower) & (X_train[c] < upper)]
        y_train = y_train[X_train.index]
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

st.success(f"After outlier removal — Train: {X_train.shape[0]} rows")

with st.spinner("Encoding & scaling…"):
    object_cols = X_train.select_dtypes(include="object").columns
    LE = LabelEncoder()
    for col in object_cols:
        X_train[col] = LE.fit_transform(X_train[col])
        X_test[col] = LE.transform(X_test[col])

    bool_cols = X_train.select_dtypes(include="bool").columns
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols] = X_test[bool_cols].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

with st.spinner("Applying SMOTE…"):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

st.subheader("Class Distribution After SMOTE")
fig, ax = plt.subplots()
sns.countplot(x=y_train_res, ax=ax)
ax.set_title("Loan Approval Distribution (After SMOTE)")
st.pyplot(fig)
plt.close()

# ── 5. Dimensionality Reduction ──────────────────────────────────────────────
st.header("5. Dimensionality Reduction")

with st.spinner("Variance thresholding…"):
    var = VarianceThreshold(threshold=0.9)
    features_before = X_train_res.columns
    X_train_var_array = var.fit_transform(X_train_res)
    X_test_var_array = var.transform(X_test_scaled)
    mask = var.get_support()
    features_after = features_before[mask]
    X_train_res_final = pd.DataFrame(X_train_var_array, columns=features_after)
    X_test_final = pd.DataFrame(X_test_var_array, columns=features_after)

col1, col2 = st.columns(2)
col1.metric("Features before VarianceThreshold", len(features_before))
col2.metric("Features after VarianceThreshold", X_train_res_final.shape[1])

with st.spinner("Applying PCA…"):
    pca = PCA(n_components=0.95)
    X_train_res_pca = pca.fit_transform(X_train_res_final)
    X_test_pca = pca.transform(X_test_final)

st.metric("PCA components chosen (95% variance)", X_train_res_pca.shape[1])

# ── 6. Model Training ─────────────────────────────────────────────────────────
st.header("6. SVM Training (GridSearchCV)")

cv_choice = st.selectbox("Cross-validation folds", [3, 5], index=1,
                          help="CV=5 is more reliable; CV=3 is faster.")

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto"],
    "kernel": ["rbf"],
}

if st.button("🚀 Train Model"):
    with st.spinner(f"Running GridSearchCV with CV={cv_choice} — this may take a few minutes…"):
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=cv_choice)
        grid.fit(X_train_res_pca, y_train_res)

    st.success(f"Best parameters: **{grid.best_params_}**")

    grid_predictions = grid.predict(X_test_pca)

    # ── 7. Evaluation ────────────────────────────────────────────────────────
    st.header("7. Model Evaluation")
    report = classification_report(y_test, grid_predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

    st.markdown("""
    > **Model Evaluation Insight:**  
    > CV=5 with C=100 typically achieves ~90% accuracy. CV=3 may show a slightly higher 91%,  
    > but CV=5 is more reliable as it validates across more data subsets.
    """)
else:
    st.info("Click **Train Model** above to run GridSearchCV and see evaluation results.")
