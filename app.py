"""
Student Placement Predictor -- Data Science Lifecycle Demo
============================================================
A teaching tool for BCA 1st year students to walk through the complete
Data Science lifecycle using a real-world use case: predicting whether
a student will get placed.

To run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

sns.set_style("whitegrid")
st.set_page_config(page_title="DS Lifecycle - Placement Predictor", layout="wide", page_icon="🎓")

CONTINUOUS_COLS = ['CGPA', 'Attendance', 'Projects_Completed', 'Backlogs', 'Communication_Skill', 'Overall_Score']
BINARY_COLS = ['Internship', 'Has_Experience']
ALL_FEATURE_CANDIDATES = CONTINUOUS_COLS + BINARY_COLS


# ============================================================
# STAGE 1 : RAW (MESSY) DATA -- as it would look in the real world
# ============================================================
@st.cache_data
def generate_raw_data(seed=42):
    np.random.seed(seed)
    n = 300

    cgpa = np.round(np.random.normal(7, 1.2, n), 2)
    attendance = np.round(np.random.normal(75, 12, n), 1)
    projects = np.random.randint(0, 6, n)
    internship = np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65])
    backlogs = np.random.choice([0, 0, 0, 1, 1, 2, 3], n)
    comm = np.random.randint(3, 11, n)

    score = (cgpa * 1.2) + (attendance * 0.05) + (projects * 0.8) + (comm * 0.5) - (backlogs * 1.5)
    score += (internship == 'Yes').astype(int) * 2
    score += np.random.normal(0, 2, n)
    threshold = np.percentile(score, 72)   # ~28% students placed -> class imbalance
    placed = np.where(score > threshold, 'Yes', 'No')

    df = pd.DataFrame({
        'Student_ID': [f"STU{1000 + i}" for i in range(n)],
        'CGPA': cgpa,
        'Attendance': attendance,
        'Projects_Completed': projects,
        'Internship': internship,
        'Backlogs': backlogs,
        'Communication_Skill': comm,
        'Placed': placed
    })

    rng = np.random.RandomState(7)
    missing_idx = rng.choice(df.index, 15, replace=False)
    df.loc[missing_idx[:6], 'CGPA'] = np.nan
    df.loc[missing_idx[6:11], 'Attendance'] = np.nan
    df.loc[missing_idx[11:], 'Communication_Skill'] = np.nan

    dup_rows = df.sample(8, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    outlier_idx = rng.choice(df.index, 4, replace=False)
    df.loc[outlier_idx[0], 'CGPA'] = 15.5
    df.loc[outlier_idx[1], 'Attendance'] = 145.0
    df.loc[outlier_idx[2], 'CGPA'] = -2.0
    df.loc[outlier_idx[3], 'Attendance'] = -10.0

    return df


# ============================================================
# STAGE 2 : DATA CLEANING
# ============================================================
def clean_data(df):
    d = df.copy()
    dup_count = d.duplicated().sum()
    d = d.drop_duplicates().reset_index(drop=True)

    d['CGPA'] = d['CGPA'].clip(0, 10)
    d['Attendance'] = d['Attendance'].clip(0, 100)

    fill_info = {}
    for col in ['CGPA', 'Attendance', 'Communication_Skill']:
        n_missing = int(d[col].isna().sum())
        med = round(float(d[col].median()), 2)
        d[col] = d[col].fillna(med)
        fill_info[col] = (n_missing, med)

    return d, dup_count, fill_info


# ============================================================
# STAGE 4 : ENCODING (text -> number)
# ============================================================
def encode_data(df):
    d = df.copy()
    d['Internship'] = d['Internship'].map({'Yes': 1, 'No': 0})
    d['Placed'] = d['Placed'].map({'Yes': 1, 'No': 0})
    return d


# ============================================================
# STAGE 5 : FEATURE CONSTRUCTION (building new features)
# ============================================================
def construct_features(df):
    d = df.copy()
    d['Overall_Score'] = (d['CGPA'] * 4) + (d['Communication_Skill'] * 2) + \
                          (d['Projects_Completed'] * 3) - (d['Backlogs'] * 5)
    d['Has_Experience'] = ((d['Internship'] == 1) | (d['Projects_Completed'] >= 2)).astype(int)
    return d


# ============================================================
# STAGE 6 : FEATURE SCALING
# ============================================================
def scale_features(df, cols):
    d = df.copy()
    scaler = StandardScaler()
    d[cols] = scaler.fit_transform(d[cols])
    return d, scaler


# ============================================================
# STAGE 7 : FEATURE SELECTION (correlation based)
# ============================================================
def select_features(df, candidates, target='Placed', threshold=0.12, min_features=4):
    corr = df[candidates + [target]].corr()[target].drop(target)
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)
    selected = corr_sorted[corr_sorted.abs() >= threshold].index.tolist()
    if len(selected) < min_features:
        selected = corr_sorted.abs().sort_values(ascending=False).index[:min_features].tolist()
    return selected, corr_sorted


# ============================================================
# FULL PIPELINE (runs once, cached)
# ============================================================
@st.cache_resource
def run_full_pipeline():
    raw = generate_raw_data()
    cleaned, dup_count, fill_info = clean_data(raw)
    encoded = encode_data(cleaned)
    constructed = construct_features(encoded)
    scaled, scaler = scale_features(constructed, CONTINUOUS_COLS)
    selected_features, corr_series = select_features(scaled, ALL_FEATURE_CANDIDATES)

    X = scaled[selected_features]
    y = scaled['Placed']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df['Placed'] = y_train.values
    before_counts = train_df['Placed'].value_counts().to_dict()

    majority = train_df[train_df.Placed == 0]
    minority = train_df[train_df.Placed == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced_train = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    after_counts = balanced_train['Placed'].value_counts().to_dict()

    X_train_bal = balanced_train[selected_features]
    y_train_bal = balanced_train['Placed']

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_bal, y_train_bal)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train_bal, y_train_bal)

    return {
        'raw': raw, 'cleaned': cleaned, 'dup_count': dup_count, 'fill_info': fill_info,
        'encoded': encoded, 'constructed': constructed, 'scaled': scaled, 'scaler': scaler,
        'selected_features': selected_features, 'corr_series': corr_series,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'before_counts': before_counts, 'after_counts': after_counts,
        'balanced_train': balanced_train,
        'log_reg': log_reg, 'rf': rf
    }


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds, zero_division=0),
        'F1 Score': f1_score(y_test, preds, zero_division=0)
    }
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    return metrics, cm, fpr, tpr, roc_auc


def predict_new_student(inputs, pipe, model_name):
    cgpa, attendance = inputs['CGPA'], inputs['Attendance']
    projects, backlogs, comm = inputs['Projects_Completed'], inputs['Backlogs'], inputs['Communication_Skill']
    internship = 1 if inputs['Internship'] == 'Yes' else 0

    overall_score = (cgpa * 4) + (comm * 2) + (projects * 3) - (backlogs * 5)
    has_experience = int((internship == 1) or (projects >= 2))

    row = pd.DataFrame([{
        'CGPA': cgpa, 'Attendance': attendance, 'Projects_Completed': projects,
        'Backlogs': backlogs, 'Communication_Skill': comm, 'Overall_Score': overall_score,
        'Internship': internship, 'Has_Experience': has_experience
    }])
    row[CONTINUOUS_COLS] = pipe['scaler'].transform(row[CONTINUOUS_COLS])
    row_selected = row[pipe['selected_features']]

    model = pipe['log_reg'] if model_name == 'Logistic Regression' else pipe['rf']
    pred = model.predict(row_selected)[0]
    prob = model.predict_proba(row_selected)[0][1]
    return pred, prob


@st.cache_data
def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


def download_button_for(df, filename, key):
    st.download_button(
        label=f"⬇️ Download this dataset ({len(df)} rows) as CSV",
        data=to_csv_bytes(df),
        file_name=filename,
        mime='text/csv',
        key=key
    )


# ============================================================
# APP UI
# ============================================================
pipe = run_full_pipeline()

st.title("🎓 Data Science Lifecycle — Student Placement Predictor")
st.caption("A real-world Data Science project for BCA students — step by step, end to end")

tab1, tab2, tab3 = st.tabs(["📘 Information", "🔄 DS Lifecycle (Step-by-Step)", "🔮 Prediction"])

# ------------------------------------------------------------------
# TAB 1 : INFORMATION
# ------------------------------------------------------------------
with tab1:
    st.header("What Is This Project?")
    st.markdown("""
Imagine you're sitting in a **placement cell**. Every year hundreds of students walk in,
and you need to predict in advance **who is likely to get placed and who isn't** — just by
looking at their academic and skill records. This is exactly what a **Data Scientist** does:
learn from historical data, then predict on new, unseen data.

In this app we'll walk through a **complete Data Science project**, live — exactly the way
it's done in industry. Think of it like **cooking a meal**:

- 🥦 **Raw data** = vegetables straight from the market (some bruised, some muddy)
- 🧼 **Data Cleaning** = washing and chopping the vegetables
- 👀 **EDA** = looking closely to understand freshness and quantities
- 🧂 **Feature Engineering** = spices and seasoning — enhancing the flavour
- 🍳 **Model Building** = actually cooking the dish
- 🍽️ **Evaluation** = tasting it — checking whether it turned out right
    """)

    st.subheader("📊 Dataset Columns")
    col_info = pd.DataFrame({
        'Column': ['Student_ID', 'CGPA', 'Attendance', 'Projects_Completed',
                   'Internship', 'Backlogs', 'Communication_Skill', 'Placed (Target)'],
        'Meaning': [
            'Unique ID for the student',
            'CGPA on a scale of 0 to 10',
            'Attendance percentage (0-100%)',
            'Number of mini/major projects completed',
            'Whether the student did an internship (Yes/No)',
            'Number of pending/failed papers',
            'Communication skill rating (3 to 10)',
            'Whether the student got placed (Yes/No) — this is what we predict'
        ]
    })
    st.table(col_info)

    st.subheader("🔄 Data Science Lifecycle — At a Glance")
    st.markdown("""
| Step | What Happens |
|---|---|
| 1. Data Collection | Gathering raw data from the real world (often messy) |
| 2. Data Cleaning | Fixing missing values, duplicates, and outliers |
| 3. EDA | Understanding the data through charts and statistics |
| 4. Encoding | Converting text categories (Yes/No) into numbers |
| 5. Feature Construction | Building new, more useful columns from existing ones |
| 6. Feature Scaling | Bringing all numeric columns to a common range |
| 7. Feature Selection | Keeping only the useful features, dropping weak ones |
| 8. Imbalance Handling | Balancing classes when one class is under-represented |
| 9. Model Building | Training a machine learning algorithm |
| 10. Evaluation | Checking how accurate the model actually is |

👉 **In Tab 2**, see exactly what the dataset looks like after every single step.
👉 **In Tab 3**, enter a new student's details yourself and try the prediction.
    """)
    st.info("⚠️ Teaching note: for simplicity, the scaler in this demo is fit on the "
            "**entire dataset**. In a real production project, the scaler is fit **only on "
            "the training data**, so the test data stays completely unseen — otherwise you "
            "risk 'data leakage'.")

# ------------------------------------------------------------------
# TAB 2 : DS LIFECYCLE STEP BY STEP
# ------------------------------------------------------------------
with tab2:
    stage = st.selectbox("👉 Choose a stage to explore:", [
        "1️⃣ Raw Data (As Collected)",
        "2️⃣ Data Cleaning",
        "3️⃣ EDA (Exploratory Data Analysis)",
        "4️⃣ Encoding (Text → Number)",
        "5️⃣ Feature Construction",
        "6️⃣ Feature Scaling",
        "7️⃣ Feature Selection",
        "8️⃣ Train-Test Split + Imbalance Handling",
        "9️⃣ Model Training",
        "🔟 Model Evaluation"
    ])

    st.divider()

    if stage.startswith("1️⃣"):
        st.subheader("Raw Data — Straight From the Field")
        raw = pipe['raw']
        st.markdown("This is exactly how the data looks when it first arrives in the real "
                    "world — **messy**: some values are missing, some rows are duplicated, "
                    "and a few are outright outliers.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(raw))
        c2.metric("Missing Values", int(raw.isna().sum().sum()))
        c3.metric("Duplicate Rows", int(raw.duplicated().sum()))
        st.dataframe(raw, width='stretch')
        st.caption("Notice some odd values in CGPA and Attendance (like CGPA = 15.5 or "
                  "Attendance = -10) — these are outliers, and some cells are blank (NaN).")
        download_button_for(raw, "1_raw_data.csv", "dl1")

    elif stage.startswith("2️⃣"):
        st.subheader("After Data Cleaning")
        cleaned, dup_count, fill_info = pipe['cleaned'], pipe['dup_count'], pipe['fill_info']
        st.markdown("""
**What was done:**
- Removed duplicate rows
- Clipped CGPA to the 0-10 range and Attendance to 0-100 (fixing outliers)
- Filled missing values using the column's **median** (more robust than mean when outliers exist)
        """)
        c1, c2 = st.columns(2)
        c1.metric("Duplicates Removed", int(dup_count))
        c2.metric("Rows Now", len(cleaned))
        for col, (n_missing, med) in fill_info.items():
            st.write(f"• **{col}**: {n_missing} missing values filled with **{med}** (median)")
        st.dataframe(cleaned, width='stretch')
        download_button_for(cleaned, "2_cleaned_data.csv", "dl2")

    elif stage.startswith("3️⃣"):
        st.subheader("EDA — Understanding the Data")
        cleaned = pipe['cleaned']
        st.markdown("The data is clean now. Let's use charts and statistics to spot patterns.")

        st.write("**Summary Statistics**")
        st.dataframe(cleaned.describe().round(2), width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            cleaned['Placed'].value_counts().plot(kind='bar', ax=ax, color=['#e76f51', '#2a9d8f'])
            ax.set_title("Placed vs Not Placed (Class Balance)")
            ax.set_xlabel("Placed")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.caption("Notice there are far more 'No' students than 'Yes' — this is an "
                      "**imbalanced dataset**.")

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(data=cleaned, x='CGPA', hue='Placed', kde=True, ax=ax, palette=['#e76f51', '#2a9d8f'])
            ax.set_title("CGPA Distribution by Placement")
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(data=cleaned, x='Placed', y='Attendance', hue='Placed', ax=ax,
                        palette=['#e76f51', '#2a9d8f'], legend=False)
            ax.set_title("Attendance vs Placement")
            st.pyplot(fig)

        with col4:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            numeric_cols = ['CGPA', 'Attendance', 'Projects_Completed', 'Backlogs', 'Communication_Skill']
            corr = cleaned[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            st.caption("This shows which columns move together.")

    elif stage.startswith("4️⃣"):
        st.subheader("Encoding — Converting Text to Numbers")
        st.markdown("""
Machine Learning models only understand **numbers**, not text.
So the 'Yes'/'No' columns were converted to 1/0:
- `Internship`: Yes → 1, No → 0
- `Placed`: Yes → 1, No → 0 (this is our target variable)
        """)
        st.dataframe(pipe['encoded'], width='stretch')
        download_button_for(pipe['encoded'], "3_encoded_data.csv", "dl3")

    elif stage.startswith("5️⃣"):
        st.subheader("Feature Construction — Building New, Useful Columns")
        st.markdown("""
Sometimes the existing columns can be combined into a **new, more powerful column**.
Here we built two new features:

- **Overall_Score** = `(CGPA × 4) + (Communication × 2) + (Projects × 3) − (Backlogs × 5)`
  → A single number that captures a student's overall strength
- **Has_Experience** = 1 if the student did an internship OR completed 2+ projects, else 0
  → Captures whether the student has hands-on practical exposure
        """)
        st.dataframe(pipe['constructed'], width='stretch')
        download_button_for(pipe['constructed'], "4_constructed_data.csv", "dl4")

    elif stage.startswith("6️⃣"):
        st.subheader("Feature Scaling")
        st.markdown("""
Notice — CGPA ranges from 0-10, but Attendance ranges from 0-100.
If we feed these as-is into a model, it might treat Attendance as "more important"
just because its numbers are bigger — which would be misleading.

**Scaling** brings all numeric columns onto a common range (mean = 0, std = 1),
so every feature gets a fair chance to contribute.

*(Note: `Internship` and `Has_Experience` are already 0/1, so they weren't scaled)*
        """)
        st.dataframe(pipe['scaled'], width='stretch')
        download_button_for(pipe['scaled'], "5_scaled_data.csv", "dl5")

    elif stage.startswith("7️⃣"):
        st.subheader("Feature Selection — Keeping Only What Matters")
        st.markdown("""
Not every feature is equally useful. We check how strongly each feature is
**correlated** with `Placed`, and drop the weaker ones. This keeps the model
both simpler and more accurate.
        """)
        corr_df = pipe['corr_series'].reset_index()
        corr_df.columns = ['Feature', 'Correlation with Placed']
        corr_df['Correlation with Placed'] = corr_df['Correlation with Placed'].round(3)
        corr_df['Selected?'] = corr_df['Feature'].apply(lambda x: '✅ Yes' if x in pipe['selected_features'] else '❌ No')
        st.dataframe(corr_df, width='stretch')
        st.success(f"**Selected Features:** {', '.join(pipe['selected_features'])}")

    elif stage.startswith("8️⃣"):
        st.subheader("Train-Test Split + Imbalance Handling")
        st.markdown("""
**Train-Test Split**: The data is split into 2 parts — 75% for **training**
(the model learns from this) and 25% for **testing** (the model never sees this
until the final check).
        """)
        c1, c2 = st.columns(2)
        c1.metric("Training Rows", len(pipe['X_train']))
        c2.metric("Testing Rows", len(pipe['X_test']))

        st.markdown("""
**Imbalance Handling**: Recall from the EDA that "Placed = Yes" students were far
fewer. If trained as-is, the model could just predict "No" every time and still look
"accurate" — but that's useless. So the **training data** (only the training set,
never the test set) was rebalanced by upsampling the minority class (repeating rows)
until both classes were equal in size.
        """)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            labels = ['Not Placed (0)', 'Placed (1)']
            values = [pipe['before_counts'].get(0, 0), pipe['before_counts'].get(1, 0)]
            ax.bar(labels, values, color=['#e76f51', '#2a9d8f'])
            ax.set_title("Training Set — Before")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            values = [pipe['after_counts'].get(0, 0), pipe['after_counts'].get(1, 0)]
            ax.bar(labels, values, color=['#e76f51', '#2a9d8f'])
            ax.set_title("Training Set — After (Balanced)")
            st.pyplot(fig)

    elif stage.startswith("9️⃣"):
        st.subheader("Model Training")
        st.markdown("""
Now we train two different models on the balanced training data, so we can compare:

- **Logistic Regression**: Simple, fast, and easy to interpret — it draws a "line"
  that separates Placed from Not-Placed.
- **Random Forest**: A group of many "decision trees" working together — capable of
  capturing more complex patterns.
        """)
        st.code("""
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_balanced, y_train_balanced)

rf = RandomForestClassifier(n_estimators=200, max_depth=6)
rf.fit(X_train_balanced, y_train_balanced)
        """, language="python")
        st.success("✅ Both models are trained! Head to 'Model Evaluation' to see which one performs better.")

    elif stage.startswith("🔟"):
        st.subheader("Model Evaluation")
        st.markdown("Now we check both models on the **test data** — data they've never seen before.")

        summary_rows = []
        for name, model in [("Logistic Regression", pipe['log_reg']), ("Random Forest", pipe['rf'])]:
            st.markdown(f"### {name}")
            metrics, cm, fpr, tpr, roc_auc = evaluate_model(model, pipe['X_test'], pipe['y_test'])
            summary_rows.append({'Model': name, **metrics, 'ROC-AUC': roc_auc})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['Accuracy']*100:.1f}%")
            c2.metric("Precision", f"{metrics['Precision']*100:.1f}%")
            c3.metric("Recall", f"{metrics['Recall']*100:.1f}%")
            c4.metric("F1 Score", f"{metrics['F1 Score']*100:.1f}%")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Not Placed', 'Placed'], yticklabels=['Not Placed', 'Placed'])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ax.plot(fpr, tpr, color='#2a9d8f', label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)
            st.divider()

        st.markdown("### 📋 Model Comparison")
        summary_df = pd.DataFrame(summary_rows).set_index('Model').round(3)
        st.dataframe(summary_df, width='stretch')
        best_model_name = summary_df['F1 Score'].idxmax()
        st.success(f"Based on F1 Score, **{best_model_name}** performs better on this dataset.")

        st.markdown("### 🔍 Feature Importance (Random Forest)")
        st.markdown("Random Forest can tell us which features it relied on most when making decisions.")
        importances = pd.Series(pipe['rf'].feature_importances_, index=pipe['selected_features'])
        importances = importances.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        importances.plot(kind='barh', ax=ax, color='#2a9d8f')
        ax.set_xlabel("Importance")
        ax.set_title("Which Features Mattered Most?")
        st.pyplot(fig)

        st.markdown("""
**Understanding the metrics:**
- **Accuracy**: Overall, how many predictions were correct
- **Precision**: Of everyone predicted "Placed", how many actually were
- **Recall**: Of everyone who actually got placed, how many did the model catch
- **F1 Score**: A balance between Precision and Recall
        """)

# ------------------------------------------------------------------
# TAB 3 : PREDICTION
# ------------------------------------------------------------------
with tab3:
    st.header("🔮 Predict Placement for a New Student")
    st.markdown("Enter a new student's details and see what the model predicts.")

    model_choice = st.radio("Choose a model:", ["Logistic Regression", "Random Forest"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        cgpa_in = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        attendance_in = st.slider("Attendance (%)", 0.0, 100.0, 78.0, 1.0)
        projects_in = st.slider("Projects Completed", 0, 5, 2)
    with col2:
        internship_in = st.radio("Did an internship?", ["Yes", "No"], horizontal=True)
        backlogs_in = st.slider("Backlogs", 0, 5, 0)
        comm_in = st.slider("Communication Skill (1-10)", 1, 10, 7)

    if st.button("🎯 Predict", type="primary"):
        inputs = {
            'CGPA': cgpa_in, 'Attendance': attendance_in, 'Projects_Completed': projects_in,
            'Internship': internship_in, 'Backlogs': backlogs_in, 'Communication_Skill': comm_in
        }
        pred, prob = predict_new_student(inputs, pipe, model_choice)

        st.divider()
        if pred == 1:
            st.success("### ✅ Placement Chance: HIGH")
            st.markdown(f"According to the model, this student's **probability of being placed is {prob*100:.1f}%**")
            st.progress(min(max(prob, 0.0), 1.0))
            st.balloons()
        else:
            st.error("### ❌ Placement Chance: LOW")
            st.markdown(f"According to the model, this student's **probability of being placed is {prob*100:.1f}%**")
            st.progress(min(max(prob, 0.0), 1.0))
            st.markdown("Tip: improving CGPA, doing more projects, or completing an internship "
                        "tends to have a positive effect on placement chances.")

        st.caption(f"Model used: {model_choice}")
