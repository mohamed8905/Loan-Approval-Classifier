import io
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Loan Approval Classifier", layout="wide")
st.title("🏦 Loan Approval Classifier")
st.markdown("ML pipeline: preprocessing → dimensionality reduction → SVM training.")

# ── Load data ────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("Loan.csv")
except FileNotFoundError:
    st.error("❌ `Loan.csv` not found. Make sure it is in the same directory as `app.py`.")
    st.stop()

# ── Full pipeline function ───────────────────────────────────────────────────
def run_pipeline(df, cv_choice):
    X = df.drop(["LoanApproved", "ApplicationDate"], axis=1)
    y = df["LoanApproved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train.columns = X_train.columns.str.strip()
    X_test.columns  = X_test.columns.str.strip()

    # Outlier removal
    for c in [c for c in ["AnnualIncome", "MonthlyIncome", "LoanAmount"] if c in X_train.columns]:
        Q1, Q3 = X_train[c].quantile(0.25), X_train[c].quantile(0.75)
        IQR = Q3 - Q1
        X_train = X_train[(X_train[c] > Q1 - 1.5 * IQR) & (X_train[c] < Q3 + 1.5 * IQR)]
        y_train = y_train[X_train.index]
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Encode
    label_encoders = {}
    object_cols = X_train.select_dtypes(include="object").columns
    for col in object_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col]  = le.transform(X_test[col])
        label_encoders[col] = le

    bool_cols = X_train.select_dtypes(include="bool").columns
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols]  = X_test[bool_cols].astype(int)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,  index=X_test.index)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Variance threshold
    var = VarianceThreshold(threshold=0.9)
    X_train_var = var.fit_transform(X_train_res)
    X_test_var  = var.transform(X_test_scaled)
    features_after = X_train_res.columns[var.get_support()]
    X_train_final = pd.DataFrame(X_train_var, columns=features_after)
    X_test_final  = pd.DataFrame(X_test_var,  columns=features_after)

    # PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_final)
    X_test_pca  = pca.transform(X_test_final)

    # GridSearchCV
    param_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"], "kernel": ["rbf"]}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=cv_choice)
    grid.fit(X_train_pca, y_train_res)

    return {
        "grid": grid, "scaler": scaler, "label_encoders": label_encoders,
        "var": var, "pca": pca, "features_after": features_after,
        "X_train": X_train, "X_test": X_test,
        "X_train_res": X_train_res, "y_train": y_train,
        "y_train_res": y_train_res, "y_test": y_test,
        "X_test_pca": X_test_pca, "best_params": grid.best_params_,
        "predictions": grid.predict(X_test_pca),
        "n_features_before": len(X_train_res.columns),
        "n_features_after": X_train_final.shape[1],
        "n_pca_components": X_train_pca.shape[1],
    }

# ── Preprocessing preview (runs at startup, no model needed) ─────────────────
@st.cache_data
def run_preprocessing_preview(df):
    X = df.drop(["LoanApproved", "ApplicationDate"], axis=1)
    y = df["LoanApproved"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train.columns = X_train.columns.str.strip()

    # Outlier removal
    for c in [c for c in ["AnnualIncome", "MonthlyIncome", "LoanAmount"] if c in X_train.columns]:
        Q1, Q3 = X_train[c].quantile(0.25), X_train[c].quantile(0.75)
        IQR = Q3 - Q1
        X_train = X_train[(X_train[c] > Q1 - 1.5 * IQR) & (X_train[c] < Q3 + 1.5 * IQR)]
        y_train = y_train[X_train.index]

    # Encode
    object_cols = X_train.select_dtypes(include="object").columns
    for col in object_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
    bool_cols = X_train.select_dtypes(include="bool").columns
    X_train[bool_cols] = X_train[bool_cols].astype(int)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Variance threshold
    var = VarianceThreshold(threshold=0.9)
    X_train_var = var.fit_transform(X_train_res)
    features_after = X_train_res.columns[var.get_support()]
    X_train_final = pd.DataFrame(X_train_var, columns=features_after)

    # PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_final)

    return {
        "rows_after_outlier": X_train.shape[0],
        "y_train_res": y_train_res,
        "n_features_before": len(X_train_res.columns),
        "n_features_after": X_train_final.shape[1],
        "n_pca_components": X_train_pca.shape[1],
    }

prep_preview = run_preprocessing_preview(df)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview",
    "🔍 EDA",
    "⚙️ Preprocessing & Reduction",
    "🤖 Train & Evaluate",
    "🧾 Manual Prediction",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Overview
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    with st.expander("Dataset Info"):
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())

    X_raw = df.drop(["LoanApproved", "ApplicationDate"], axis=1)
    y_raw = df["LoanApproved"]
    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    c1, c2 = st.columns(2)
    c1.metric("Train samples", X_train_raw.shape[0])
    c2.metric("Test samples",  X_test_raw.shape[0])

    with st.expander("Training set descriptive statistics"):
        st.dataframe(X_train_raw.describe())

    with st.expander("📄 View Original Notebook Code — Data Loading & Split"):
        st.code("""\
df = pd.read_csv("C:\\\\Users\\\\aa\\\\OneDrive\\\\Desktop\\\\Data sets\\\\Loan.csv")
df = pd.DataFrame(df)
df.head()

print(f"Shape: {df.shape}")

X = df.drop(["LoanApproved", "ApplicationDate"], axis=1)
y = df["LoanApproved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:",  X_test.shape)

print(df.info())
X_train.describe()
""", language="python")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    X_eda = df.drop(["LoanApproved", "ApplicationDate"], axis=1).copy()
    X_eda.columns = X_eda.columns.str.strip()
    y_eda = df["LoanApproved"]
    X_tr, _, y_tr, _ = train_test_split(X_eda, y_eda, test_size=0.2, random_state=42, stratify=y_eda)

    st.subheader("Loan Approval Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y_tr, ax=ax)
    ax.set_title("Loan Approval Distribution (Train)")
    st.pyplot(fig); plt.close()

    st.subheader("Feature Distributions")
    num_cols = [c for c in ["Age", "AnnualIncome", "Experience", "LoanAmount", "MonthlyIncome", "CreditScore"] if c in X_tr.columns]
    if num_cols:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(X_tr[col], kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution for {col}")
            axes[i].set_xlabel(col); axes[i].set_ylabel("Count")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(X_tr.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig); plt.close()

    if "EmploymentStatus" in X_tr.columns:
        st.subheader("Employment Status vs Loan Approval")
        fig, ax = plt.subplots()
        sns.countplot(x=X_tr["EmploymentStatus"], hue=y_tr, ax=ax)
        plt.xticks(rotation=45); ax.set_title("Employment vs Approval")
        st.pyplot(fig); plt.close()

    st.subheader("Missing Values")
    missing = X_tr.isnull().sum()
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        st.success("✅ No missing values found in the training set.")
    else:
        st.warning(f"⚠️ {len(missing_cols)} column(s) have missing values.")
        st.dataframe(missing_cols.rename("Missing Count"))

    with st.expander("📄 View Original Notebook Code — EDA"):
        st.code("""\
sns.countplot(x=y_train)
plt.title("Loan Approval Distribution")
plt.show()

X_train.columns = X_train.columns.str.strip()
cols = ["Age", "AnnualIncome", "Experience", "LoanAmount", "MonthlyIncome", "CreditScore"]
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.flatten()
for i, col in enumerate(cols):
    sns.histplot(X_train[col], kde=True, ax=ax[i])
    ax[i].set_title(f"Distribution for {col}")
    ax[i].set_xlabel(f"{col}")
    ax[i].set_ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 10))
sns.heatmap(X_train.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

sns.countplot(x=X_train['EmploymentStatus'], hue=y_train)
plt.xticks(rotation=45)
plt.title("Employment vs Approval")
plt.show()

print(X_train.isnull().sum())
""", language="python")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Preprocessing & Reduction
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("After Outlier Removal")
    st.write(f"Training rows after IQR filtering: **{prep_preview['rows_after_outlier']}**")

    st.subheader("Class Distribution After SMOTE")
    fig, ax = plt.subplots()
    sns.countplot(x=prep_preview["y_train_res"], ax=ax)
    ax.set_title("Loan Approval Distribution (After SMOTE)")
    st.pyplot(fig); plt.close()

    st.subheader("Dimensionality Reduction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Features before VarianceThreshold", prep_preview["n_features_before"])
    c2.metric("Features after VarianceThreshold",  prep_preview["n_features_after"])
    c3.metric("PCA components (95% variance)",     prep_preview["n_pca_components"])

    with st.expander("📄 View Original Notebook Code — Preprocessing & Reduction"):
        st.code("""\
# Outlier removal (IQR)
col = ["AnnualIncome", "MonthlyIncome", "LoanAmount"]
for c in col:
    Q1 = X_train[c].quantile(0.25)
    Q3 = X_train[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X_train = X_train[(X_train[c] > lower) & (X_train[c] < upper)]
    y_train = y_train[X_train.index]
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Label encoding
object_cols = X_train.select_dtypes(include='object').columns
LE = LabelEncoder()
for col in object_cols:
    X_train[col] = LE.fit_transform(X_train[col])
    X_test[col]  = LE.transform(X_test[col])

bool_cols = X_train.select_dtypes(include='bool').columns
X_train[bool_cols] = X_train[bool_cols].astype(int)
X_test[bool_cols]  = X_test[bool_cols].astype(int)

# StandardScaler — crucial for SVM: ensures all features contribute equally
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns,  index=X_test.index)

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Variance threshold (0.9 — effective after StandardScaler shifts variance toward 1.0)
var = VarianceThreshold(threshold=0.9)
X_train_var = var.fit_transform(X_train_res)
X_test_var  = var.transform(X_test_scaled)

# PCA (retain 95% of variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_var)
X_test_pca  = pca.transform(X_test_var)
print(f"Number of components chosen: {X_train_pca.shape[1]}")
""", language="python")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Train & Evaluate
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    cv_choice = st.selectbox(
        "Cross-validation folds", [3, 5], index=1,
        help="CV=5 is more reliable; CV=3 is faster."
    )

    if st.button("🚀 Train Model"):
        with st.spinner(f"Running full pipeline + GridSearchCV (CV={cv_choice})…"):
            st.session_state["pipeline"] = run_pipeline(df, cv_choice)
        st.success("Training complete!")

    if "pipeline" in st.session_state:
        p = st.session_state["pipeline"]
        st.success(f"Best parameters: **{p['best_params']}**")

        st.subheader("Classification Report")
        report_df = pd.DataFrame(
            classification_report(p["y_test"], p["predictions"], output_dict=True)
        ).transpose()
        st.dataframe(report_df.style.format({
            "precision": "{:.2f}", "recall": "{:.2f}",
            "f1-score": "{:.2f}", "support": "{:.0f}"
        }))

        st.markdown("""
        > **Model Evaluation Insight:**  
        > CV=5 with C=100 typically achieves ~90% accuracy. CV=3 may show a slightly higher 91%,  
        > but CV=5 is more reliable as it validates across more data subsets.
        """)
    else:
        st.info("Click **Train Model** above to run the pipeline.")

    with st.expander("📄 View Original Notebook Code — SVM Training & Evaluation"):
        st.code("""\
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C':      [0.1, 1, 10, 100],
    'gamma':  ['scale', 'auto'],
    'kernel': ['rbf'],
}

# CV=5 — more reliable (validates across more subsets)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_res_pca, y_train_res)
print("best parameters:", grid.best_params_)

grid_predictions = grid.predict(X_test_pca)
print(classification_report(y_test, grid_predictions))

# CV=3 — slightly faster, marginally higher reported accuracy
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)
grid.fit(X_train_res_pca, y_train_res)
print("best parameters:", grid.best_params_)

grid_predictions = grid.predict(X_test_pca)
print(classification_report(y_test, grid_predictions))

# Model Evaluation Insight:
# CV=5 + C=100 → ~90% accuracy (preferred — more robust validation)
# CV=3          → ~91% accuracy (could be a lucky split)
""", language="python")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Manual Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🧾 Predict Loan Approval for a Single Applicant")

    if "pipeline" not in st.session_state:
        st.warning("Please train the model first in the **Train & Evaluate** tab.")
    else:
        p = st.session_state["pipeline"]
        # Only show the columns that survived VarianceThreshold (the ones actually used by the model)
        feature_cols   = list(p["features_after"])
        label_encoders = p["label_encoders"]

        st.markdown(f"Fill in the **{len(feature_cols)} features** used by the trained model:")
        input_data = {}
        cols_left, cols_right = st.columns(2)

        for i, col in enumerate(feature_cols):
            target_col = cols_left if i % 2 == 0 else cols_right
            with target_col:
                if col in label_encoders:
                    options = list(label_encoders[col].classes_)
                    input_data[col] = st.selectbox(col, options, key=f"pred_{col}")
                elif set(p["X_train"][col].dropna().unique()).issubset({0, 1}):
                    input_data[col] = st.selectbox(
                        col, [0, 1],
                        format_func=lambda x: "Yes" if x else "No",
                        key=f"pred_{col}"
                    )
                else:
                    min_val  = float(p["X_train"][col].min())
                    max_val  = float(p["X_train"][col].max())
                    mean_val = float(p["X_train"][col].mean())
                    input_data[col] = st.number_input(
                        col, min_value=min_val, max_value=max_val,
                        value=round(mean_val, 2), key=f"pred_{col}"
                    )

        if st.button("🔮 Predict"):
            input_df = pd.DataFrame([input_data])

            # Encode categoricals that are in this reduced feature set
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col])

            # Scale using only the features_after columns
            # First build a full-column frame for the scaler, fill unseen cols with 0
            full_input = pd.DataFrame([{c: 0 for c in p["X_train"].columns}])
            for col in feature_cols:
                full_input[col] = input_df[col].values
            full_scaled  = pd.DataFrame(p["scaler"].transform(full_input), columns=p["X_train"].columns)

            # Keep only variance-selected features
            input_var  = full_scaled[p["features_after"]]

            # PCA → predict
            input_pca  = p["pca"].transform(input_var)
            prediction = p["grid"].predict(input_pca)[0]

            st.divider()
            if prediction == 1:
                st.success("✅ **Loan Approved** — The model predicts this applicant is likely to be approved.")
            else:
                st.error("❌ **Loan Denied** — The model predicts this applicant is unlikely to be approved.")

    with st.expander("📄 View Original Notebook Code — This tab has no original code"):
        st.info("The Manual Prediction tab is a Streamlit-only feature — it was not part of the original notebook. It applies the trained pipeline to a single new applicant in real time.")
