"""
Student Placement Predictor -- Data Science Lifecycle Demo
============================================================
BCA 1st year students ko poora Data Science lifecycle ek real
use-case (placement prediction) ke through samjhane ke liye bana
hua Streamlit app.

Run karne ke liye:
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
# STAGE 1 : RAW (MESSY) DATA -- jaisa real duniya me milta hai
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
    threshold = np.percentile(score, 72)   # ~28% students placed -> imbalance
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
# STAGE 5 : FEATURE CONSTRUCTION (naye features banana)
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
# FULL PIPELINE (ek baar chalta hai, cache ho jaata hai)
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


# ============================================================
# APP UI
# ============================================================
pipe = run_full_pipeline()

st.title("🎓 Data Science Lifecycle — Student Placement Predictor")
st.caption("BCA students ke liye ek real-world Data Science project — step-by-step, end to end")

tab1, tab2, tab3 = st.tabs(["📘 Information", "🔄 DS Lifecycle (Step-by-Step)", "🔮 Prediction"])

# ------------------------------------------------------------------
# TAB 1 : INFORMATION
# ------------------------------------------------------------------
with tab1:
    st.header("Yeh Project Kya Hai?")
    st.markdown("""
Socho tum ek **placement cell** mein baithe ho. Har saal sainkdo students aate hain,
aur tumhe pehle se andaza lagana hai ki **kaun placed hoga aur kaun nahi** — sirf unke
academic aur skill records dekh kar. Yehi kaam ek **Data Scientist** karta hai:
purane data se seekh kar, naye data ke liye prediction karta hai.

Is app mein hum ek **poora Data Science project** live dekhenge — bilkul waise hi jaise
industry mein hota hai. Socho isse ek **thali banane** jaisa:

- 🥦 **Raw data** = bazaar se laayi hui sabzi (kuch kharab, kuch mitti lagi hui)
- 🧼 **Data Cleaning** = sabzi dhona aur kaatna
- 👀 **EDA** = sabzi ko dekh kar samajhna — kaunsi taazi hai, kitni chahiye
- 🧂 **Feature Engineering** = masale, seasoning — swaad badhaana
- 🍳 **Model Building** = khana pakana
- 🍽️ **Evaluation** = chakh kar dekhna — swaad sahi bana ya nahi
    """)

    st.subheader("📊 Dataset ke Columns")
    col_info = pd.DataFrame({
        'Column': ['Student_ID', 'CGPA', 'Attendance', 'Projects_Completed',
                   'Internship', 'Backlogs', 'Communication_Skill', 'Placed (Target)'],
        'Matlab': [
            'Student ka unique ID',
            'CGPA (0 se 10 ke beech)',
            'Attendance percentage (0-100%)',
            'Kitne mini/major projects banaye',
            'Internship ki thi ya nahi (Yes/No)',
            'Kitne backlog/pending papers hain',
            'Communication skill rating (3 se 10)',
            'Placed hua ya nahi (Yes/No) — yeh predict karna hai'
        ]
    })
    st.table(col_info)

    st.subheader("🔄 Data Science Lifecycle — Ek Nazar Mein")
    st.markdown("""
| Step | Kya Hota Hai |
|---|---|
| 1. Data Collection | Real duniya se raw data uthana (aksar messy hota hai) |
| 2. Data Cleaning | Missing values, duplicates, outliers theek karna |
| 3. EDA | Data ko charts/stats se samajhna — patterns dhoondhna |
| 4. Encoding | Text categories (Yes/No) ko numbers mein badalna |
| 5. Feature Construction | Purane columns se naya useful column banana |
| 6. Feature Scaling | Sab numbers ko ek jaisi range mein laana |
| 7. Feature Selection | Sirf useful features rakhna, kamzor hata dena |
| 8. Imbalance Handling | Agar ek class kam hai, use balance karna |
| 9. Model Building | Machine learning algorithm ko train karna |
| 10. Evaluation | Model kitna accurate hai, yeh check karna |

👉 **Tab 2 mein** har step ke baad dataset kaisa dikhta hai, wo dekho.
👉 **Tab 3 mein** khud ek naye student ka data daal kar prediction try karo.
    """)
    st.info("⚠️ Teaching note: is demo mein scaler poore dataset par fit kiya gaya hai (samajhne ke liye simple rakha). "
            "Real production project mein scaler sirf **training data** par fit hota hai, taaki test data ka gyaan model ko pehle se na mil jaaye (isse 'data leakage' kehte hain).")

# ------------------------------------------------------------------
# TAB 2 : DS LIFECYCLE STEP BY STEP
# ------------------------------------------------------------------
with tab2:
    stage = st.selectbox("👉 Konsa stage dekhna hai, chuno:", [
        "1️⃣ Raw Data (Jaisa Mila)",
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
        st.subheader("Raw Data — Jaisa Field Se Aaya")
        raw = pipe['raw']
        st.markdown("Yeh data bilkul waisa hi hai jaisa real duniya mein milta hai — **messy**: kuch values missing hain, kuch duplicate rows hain, aur kuch outliers (galat values) bhi hain.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(raw))
        c2.metric("Missing Values", int(raw.isna().sum().sum()))
        c3.metric("Duplicate Rows", int(raw.duplicated().sum()))
        st.dataframe(raw, width='stretch')
        st.caption("Dekho CGPA aur Attendance mein kuch ajeeb values hain (jaise CGPA = 15.5 ya Attendance = -10) — yeh outliers hain, aur kuch cells khaali (NaN) hain.")

    elif stage.startswith("2️⃣"):
        st.subheader("Data Cleaning ke Baad")
        cleaned, dup_count, fill_info = pipe['cleaned'], pipe['dup_count'], pipe['fill_info']
        st.markdown("""
**Kya kiya:**
- Duplicate rows hata diye
- CGPA ko 0-10 aur Attendance ko 0-100 range mein clip kiya (outliers fix)
- Missing values ko us column ke **median** se fill kiya (average se better hai jab data mein outliers hon)
        """)
        c1, c2 = st.columns(2)
        c1.metric("Duplicates Hataye Gaye", int(dup_count))
        c2.metric("Rows Ab", len(cleaned))
        for col, (n_missing, med) in fill_info.items():
            st.write(f"• **{col}**: {n_missing} missing values ko **{med}** (median) se fill kiya")
        st.dataframe(cleaned, width='stretch')

    elif stage.startswith("3️⃣"):
        st.subheader("EDA — Data Ko Samajhna")
        cleaned = pipe['cleaned']
        st.markdown("Ab data saaf hai. Charts aur stats se dekhte hain data mein kya patterns hain.")

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
            st.caption("Dekho — 'No' wale students bahut zyada hain 'Yes' se. Yeh hai **imbalanced dataset** ka example.")

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(data=cleaned, x='CGPA', hue='Placed', kde=True, ax=ax, palette=['#e76f51', '#2a9d8f'])
            ax.set_title("CGPA Distribution (Placed ke hisaab se)")
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
            st.caption("Yeh dikhata hai ki kaunse columns ek doosre se juday hue hain.")

    elif stage.startswith("4️⃣"):
        st.subheader("Encoding — Text ko Number Mein Badalna")
        st.markdown("""
Machine Learning models sirf **numbers** samajhte hain, text nahi.
Isliye 'Yes'/'No' wale columns ko 1/0 mein badal diya:
- `Internship`: Yes → 1, No → 0
- `Placed`: Yes → 1, No → 0 (yeh humara target hai)
        """)
        st.dataframe(pipe['encoded'], width='stretch')

    elif stage.startswith("5️⃣"):
        st.subheader("Feature Construction — Naya Useful Column Banana")
        st.markdown("""
Kabhi kabhi jo columns humare paas hain unse ek **naya, zyada powerful column** bana sakte hain.
Yahan humne 2 naye features banaye:

- **Overall_Score** = `(CGPA × 4) + (Communication × 2) + (Projects × 3) − (Backlogs × 5)`
  → Ek single number jo student ki overall strength batata hai
- **Has_Experience** = 1 agar internship ki hai YA 2+ projects banaye hain, warna 0
  → Yeh batata hai ki student ke paas practical exposure hai ya nahi
        """)
        st.dataframe(pipe['constructed'], width='stretch')

    elif stage.startswith("6️⃣"):
        st.subheader("Feature Scaling")
        st.markdown("""
Dekho — CGPA 0-10 ke range mein hai, lekin Attendance 0-100 ke range mein.
Agar hum inhe aise hi model ko denge, toh model Attendance ko "zyada important" samajh sakta hai
sirf isliye ki uske numbers bade hain — yeh galat hoga.

**Scaling** sab numeric columns ko ek jaisi range mein le aati hai (mean = 0, std = 1),
taaki har feature ko barabar mauka mile.

*(Note: `Internship` aur `Has_Experience` already 0/1 hain, unhe scale nahi kiya)*
        """)
        st.dataframe(pipe['scaled'], width='stretch')

    elif stage.startswith("7️⃣"):
        st.subheader("Feature Selection — Sirf Useful Features Rakhna")
        st.markdown("""
Har feature equally useful nahi hota. Hum dekhte hain ki kaunsa feature `Placed` ke saath
sabse zyada **correlated** (juda hua) hai, aur kamzor features ko hata dete hain.
Isse model simple aur accurate dono banta hai.
        """)
        corr_df = pipe['corr_series'].reset_index()
        corr_df.columns = ['Feature', 'Correlation with Placed']
        corr_df['Correlation with Placed'] = corr_df['Correlation with Placed'].round(3)
        corr_df['Selected?'] = corr_df['Feature'].apply(lambda x: '✅ Haan' if x in pipe['selected_features'] else '❌ Nahi')
        st.dataframe(corr_df, width='stretch')
        st.success(f"**Selected Features:** {', '.join(pipe['selected_features'])}")

    elif stage.startswith("8️⃣"):
        st.subheader("Train-Test Split + Imbalance Handling")
        st.markdown("""
**Train-Test Split**: Data ko 2 hisso mein baanta — 75% **training** ke liye (model isse seekhega)
aur 25% **testing** ke liye (model ko isse kabhi nahi dikhaya, sirf final check ke liye).
        """)
        c1, c2 = st.columns(2)
        c1.metric("Training Rows", len(pipe['X_train']))
        c2.metric("Testing Rows", len(pipe['X_test']))

        st.markdown("""
**Imbalance Handling**: Yaad hai EDA mein humne dekha tha ki "Placed = Yes" wale students kam hain?
Agar aise hi model train kiya, toh model hamesha "No" predict karke bhi "accurate" dikhega —
lekin yeh kaam ka nahi. Isliye humne **training data** mein minority class (Placed) ko
upsample kiya (repeat karke barabar kiya) — sirf training data mein, test data ko touch nahi kiya.
        """)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            labels = ['Not Placed (0)', 'Placed (1)']
            values = [pipe['before_counts'].get(0, 0), pipe['before_counts'].get(1, 0)]
            ax.bar(labels, values, color=['#e76f51', '#2a9d8f'])
            ax.set_title("Training Set — Pehle")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            values = [pipe['after_counts'].get(0, 0), pipe['after_counts'].get(1, 0)]
            ax.bar(labels, values, color=['#e76f51', '#2a9d8f'])
            ax.set_title("Training Set — Baad Mein (Balanced)")
            st.pyplot(fig)

    elif stage.startswith("9️⃣"):
        st.subheader("Model Training")
        st.markdown("""
Ab hum 2 alag-alag models train karte hain balanced training data par, taaki compare kar sakein:

- **Logistic Regression**: Simple, tez, aur samajhne mein aasan — ek "line" khींchta hai
  jo Placed/Not-Placed ko separate karti hai.
- **Random Forest**: Bohot saare "decision trees" ka group — zyada complex patterns pakad sakta hai.
        """)
        st.code("""
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_balanced, y_train_balanced)

rf = RandomForestClassifier(n_estimators=200, max_depth=6)
rf.fit(X_train_balanced, y_train_balanced)
        """, language="python")
        st.success("✅ Dono models train ho chuke hain! Ab 'Evaluation' stage mein dekhte hain kaunsa behtar hai.")

    elif stage.startswith("🔟"):
        st.subheader("Model Evaluation")
        st.markdown("Ab dono models ko **test data** (jo unhone kabhi nahi dekha) par check karte hain.")

        for name, model in [("Logistic Regression", pipe['log_reg']), ("Random Forest", pipe['rf'])]:
            st.markdown(f"### {name}")
            metrics, cm, fpr, tpr, roc_auc = evaluate_model(model, pipe['X_test'], pipe['y_test'])
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

        st.markdown("""
**Metrics samjho:**
- **Accuracy**: Kul kitne predictions sahi the
- **Precision**: Jinhe "Placed" predict kiya, unme se kitne sach mein placed the
- **Recall**: Jo sach mein placed the, unme se kitno ko model ne pakda
- **F1 Score**: Precision aur Recall ka balance
        """)

# ------------------------------------------------------------------
# TAB 3 : PREDICTION
# ------------------------------------------------------------------
with tab3:
    st.header("🔮 Naye Student Ka Placement Predict Karo")
    st.markdown("Ek naye student ki details daalo aur dekho model kya predict karta hai.")

    model_choice = st.radio("Model chuno:", ["Logistic Regression", "Random Forest"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        cgpa_in = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        attendance_in = st.slider("Attendance (%)", 0.0, 100.0, 78.0, 1.0)
        projects_in = st.slider("Projects Completed", 0, 5, 2)
    with col2:
        internship_in = st.radio("Internship ki hai?", ["Yes", "No"], horizontal=True)
        backlogs_in = st.slider("Backlogs", 0, 5, 0)
        comm_in = st.slider("Communication Skill (1-10)", 1, 10, 7)

    if st.button("🎯 Predict Karo", type="primary"):
        inputs = {
            'CGPA': cgpa_in, 'Attendance': attendance_in, 'Projects_Completed': projects_in,
            'Internship': internship_in, 'Backlogs': backlogs_in, 'Communication_Skill': comm_in
        }
        pred, prob = predict_new_student(inputs, pipe, model_choice)

        st.divider()
        if pred == 1:
            st.success(f"### ✅ Placement Chance: HIGH")
            st.markdown(f"Model ke hisaab se is student ke **placed hone ka probability: {prob*100:.1f}%**")
            st.balloons()
        else:
            st.error(f"### ❌ Placement Chance: LOW")
            st.markdown(f"Model ke hisaab se is student ke **placed hone ka probability: {prob*100:.1f}%**")
            st.markdown("Tip: CGPA badhao, projects karo, ya internship karo — inka positive asar padta hai.")

        st.caption(f"Model used: {model_choice}")
