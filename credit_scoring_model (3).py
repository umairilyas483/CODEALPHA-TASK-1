"""
╔══════════════════════════════════════════════════════════════════════╗
║           CREDIT SCORING MODEL — Internship Task 1                  ║
║   Predict creditworthiness using classification algorithms           ║
║   Models: Logistic Regression | Decision Tree | Random Forest        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────
#  STEP 1 — GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
print("=" * 65)
print("  CREDIT SCORING MODEL")
print("  Internship Task 1 | Classification Algorithms")
print("=" * 65)

np.random.seed(42)
N = 2000

print("\n[1/6] Generating synthetic financial dataset...")

age             = np.random.randint(21, 70, N)
income          = np.random.normal(55000, 25000, N).clip(10000, 200000).astype(int)
loan_amount     = np.random.normal(15000, 10000, N).clip(1000, 80000).astype(int)
loan_duration   = np.random.choice([12, 24, 36, 48, 60], N)
num_credit_lines= np.random.randint(1, 15, N)
num_late_payments= np.random.poisson(1.2, N).clip(0, 20)
debt_to_income  = np.round(np.random.beta(2, 5, N) * 0.8, 3)
employment_years= np.random.randint(0, 40, N)
savings_balance = np.random.exponential(8000, N).clip(0, 100000).astype(int)
employment_type = np.random.choice(["Salaried", "Self-Employed", "Business", "Unemployed"], N,
                                    p=[0.55, 0.25, 0.15, 0.05])
education       = np.random.choice(["High School", "Bachelor's", "Master's", "PhD"], N,
                                    p=[0.30, 0.45, 0.20, 0.05])

# ── Build credit score based on realistic rules ──
score = (
    (income / 5000) * 0.25
    + (savings_balance / 2000) * 0.15
    + (employment_years * 1.5) * 0.15
    - (num_late_payments * 6) * 0.20
    - (debt_to_income * 40) * 0.15
    + (num_credit_lines * 2) * 0.10
    + np.random.normal(0, 5, N)        # noise
)
creditworthy = (score > score.mean()).astype(int)  # 1 = Good, 0 = Bad

df = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Loan_Amount": loan_amount,
    "Loan_Duration_Months": loan_duration,
    "Num_Credit_Lines": num_credit_lines,
    "Num_Late_Payments": num_late_payments,
    "Debt_to_Income_Ratio": debt_to_income,
    "Employment_Years": employment_years,
    "Savings_Balance": savings_balance,
    "Employment_Type": employment_type,
    "Education": education,
    "Creditworthy": creditworthy
})

print(f"   ✔ Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   ✔ Class balance  — Good: {creditworthy.sum()} ({creditworthy.mean()*100:.1f}%)  "
      f"| Bad: {(1-creditworthy).sum()} ({(1-creditworthy).mean()*100:.1f}%)")


# ─────────────────────────────────────────────
#  STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n[2/6] Exploratory Data Analysis...")

print("\n── Dataset Overview ──")
print(df.describe().T[["mean","std","min","max"]].round(2).to_string())

print("\n── Missing Values ──")
print(df.isnull().sum().to_string())

print("\n── Class Distribution ──")
print(df["Creditworthy"].value_counts().rename({1:"Good (1)", 0:"Bad (0)"}).to_string())


# ─────────────────────────────────────────────
#  STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3/6] Feature Engineering...")

# Encode categoricals
le_emp  = LabelEncoder()
le_edu  = LabelEncoder()
df["Employment_Type_Enc"] = le_emp.fit_transform(df["Employment_Type"])
df["Education_Enc"]       = le_edu.fit_transform(df["Education"])

# New derived features
df["Loan_to_Income"]          = (df["Loan_Amount"] / df["Income"]).round(4)
df["Savings_to_Loan"]         = (df["Savings_Balance"] / (df["Loan_Amount"] + 1)).round(4)
df["Payment_History_Score"]   = (1 / (df["Num_Late_Payments"] + 1)).round(4)
df["Creditworthiness_Index"]  = (
    df["Income"] * 0.3
    + df["Savings_Balance"] * 0.2
    - df["Loan_Amount"] * 0.15
    - df["Num_Late_Payments"] * 5000
    + df["Employment_Years"] * 1000
).round(2)

FEATURE_COLS = [
    "Age", "Income", "Loan_Amount", "Loan_Duration_Months",
    "Num_Credit_Lines", "Num_Late_Payments", "Debt_to_Income_Ratio",
    "Employment_Years", "Savings_Balance",
    "Employment_Type_Enc", "Education_Enc",
    "Loan_to_Income", "Savings_to_Loan",
    "Payment_History_Score", "Creditworthiness_Index"
]

X = df[FEATURE_COLS]
y = df["Creditworthy"]

print(f"   ✔ Features used: {len(FEATURE_COLS)}")
print(f"   ✔ Engineered features: Loan_to_Income, Savings_to_Loan, "
      "Payment_History_Score, Creditworthiness_Index")


# ─────────────────────────────────────────────
#  STEP 4 — TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
print("\n[4/6] Splitting & Scaling data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   ✔ Training set : {X_train.shape[0]} samples")
print(f"   ✔ Test set     : {X_test.shape[0]} samples")


# ─────────────────────────────────────────────
#  STEP 5 — TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[5/6] Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   min_samples_leaf=10, random_state=42, n_jobs=-1),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc  if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    cv_scores = cross_val_score(
        model,
        X_train_sc if name == "Logistic Regression" else X_train,
        y_train, cv=cv, scoring="roc_auc"
    )

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
        "CV-AUC":    cv_scores.mean(),
        "CV-Std":    cv_scores.std(),
    }
    print(f"   ✔ {name:22s} — AUC: {results[name]['ROC-AUC']:.4f} | "
          f"F1: {results[name]['F1-Score']:.4f} | "
          f"CV-AUC: {results[name]['CV-AUC']:.4f} ± {results[name]['CV-Std']:.4f}")


# ─────────────────────────────────────────────
#  STEP 6 — METRICS TABLE
# ─────────────────────────────────────────────
print("\n[6/6] Full Evaluation Metrics\n")
metrics_df = pd.DataFrame({
    name: {k: v for k, v in vals.items()
           if k not in ("model","y_pred","y_proba")}
    for name, vals in results.items()
}).T

print(metrics_df[["Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV-AUC","CV-Std"]]
      .round(4).to_string())

best_model_name = metrics_df["ROC-AUC"].idxmax()
print(f"\n  🏆  Best model by ROC-AUC: {best_model_name} "
      f"({metrics_df.loc[best_model_name,'ROC-AUC']:.4f})")

print("\n── Detailed Classification Report (Best Model) ──")
print(classification_report(
    y_test,
    results[best_model_name]["y_pred"],
    target_names=["Bad Credit (0)", "Good Credit (1)"]
))


# ─────────────────────────────────────────────
#  VISUALIZATIONS (10-panel figure)
# ─────────────────────────────────────────────
print("\nGenerating visualisation dashboard...")

PALETTE = {"bg": "#0a0f1e", "surface": "#111827", "accent": "#00e5ff",
           "green": "#22c55e", "red": "#ef4444", "yellow": "#f59e0b",
           "purple": "#a855f7", "text": "#e2e8f0", "muted": "#64748b"}
MODEL_COLORS = ["#00e5ff", "#22c55e", "#a855f7"]

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    "#1e2d45",
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        "#1e2d45",
    "grid.linestyle":    "--",
    "font.family":       "monospace",
})

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor(PALETTE["bg"])
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

# ── Panel 0 — Class Distribution ──
ax0 = fig.add_subplot(gs[0, 0])
counts = y.value_counts().sort_index()
bars   = ax0.bar(["Bad Credit (0)", "Good Credit (1)"], counts.values,
                 color=[PALETTE["red"], PALETTE["green"]], width=0.5, edgecolor="none")
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f"{val}\n({val/N*100:.1f}%)", ha="center", va="bottom", fontsize=9)
ax0.set_title("Class Distribution", fontweight="bold", pad=10)
ax0.set_ylabel("Count"); ax0.grid(axis="y", alpha=0.3); ax0.set_axisbelow(True)

# ── Panel 1 — Income Distribution by Class ──
ax1 = fig.add_subplot(gs[0, 1])
for cls, color, label in [(0, PALETTE["red"], "Bad Credit"),
                            (1, PALETTE["green"], "Good Credit")]:
    ax1.hist(df.loc[df["Creditworthy"] == cls, "Income"],
             bins=35, alpha=0.65, color=color, label=label, edgecolor="none")
ax1.set_title("Income Distribution by Class", fontweight="bold", pad=10)
ax1.set_xlabel("Annual Income ($)"); ax1.set_ylabel("Frequency")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3); ax1.set_axisbelow(True)

# ── Panel 2 — Late Payments vs Credit ──
ax2 = fig.add_subplot(gs[0, 2])
bp = ax2.boxplot(
    [df.loc[df["Creditworthy"]==0,"Num_Late_Payments"],
     df.loc[df["Creditworthy"]==1,"Num_Late_Payments"]],
    labels=["Bad Credit","Good Credit"],
    patch_artist=True, medianprops=dict(color="#fff", linewidth=2),
    whiskerprops=dict(color=PALETTE["muted"]),
    capprops=dict(color=PALETTE["muted"]),
    flierprops=dict(marker="o", markersize=2, alpha=0.3, color=PALETTE["muted"])
)
for patch, color in zip(bp["boxes"], [PALETTE["red"], PALETTE["green"]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax2.set_title("Late Payments by Class", fontweight="bold", pad=10)
ax2.set_ylabel("# Late Payments"); ax2.grid(alpha=0.3); ax2.set_axisbelow(True)

# ── Panel 3 — Correlation Heatmap ──
ax3 = fig.add_subplot(gs[1, :2])
num_cols = ["Income","Loan_Amount","Savings_Balance","Debt_to_Income_Ratio",
            "Num_Late_Payments","Employment_Years","Loan_to_Income",
            "Payment_History_Score","Creditworthiness_Index","Creditworthy"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, ax=ax3, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 7.5},
            linewidths=0.4, linecolor="#0a0f1e",
            cbar_kws={"shrink": 0.7})
ax3.set_title("Feature Correlation Heatmap", fontweight="bold", pad=10)
ax3.tick_params(labelsize=8); ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha="right")

# ── Panel 4 — Debt-to-Income scatter ──
ax4 = fig.add_subplot(gs[1, 2])
colors_scatter = [PALETTE["red"] if c==0 else PALETTE["green"] for c in df["Creditworthy"]]
ax4.scatter(df["Debt_to_Income_Ratio"], df["Income"],
            c=colors_scatter, alpha=0.3, s=8, edgecolors="none")
ax4.set_title("Debt-to-Income vs Income", fontweight="bold", pad=10)
ax4.set_xlabel("Debt-to-Income Ratio"); ax4.set_ylabel("Income ($)")
from matplotlib.patches import Patch
ax4.legend(handles=[Patch(facecolor=PALETTE["green"], label="Good"),
                    Patch(facecolor=PALETTE["red"],   label="Bad")],
           fontsize=8, loc="upper right")
ax4.grid(alpha=0.3); ax4.set_axisbelow(True)

# ── Panel 5,6,7 — ROC Curves ──
ax5 = fig.add_subplot(gs[2, 0])
for (name, res), color in zip(results.items(), MODEL_COLORS):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax5.plot(fpr, tpr, color=color, lw=2,
             label=f"{name}\n(AUC={res['ROC-AUC']:.3f})")
ax5.plot([0,1],[0,1], "--", color=PALETTE["muted"], lw=1)
ax5.fill_between([0,1],[0,1], alpha=0.05, color=PALETTE["muted"])
ax5.set_title("ROC Curves — All Models", fontweight="bold", pad=10)
ax5.set_xlabel("False Positive Rate"); ax5.set_ylabel("True Positive Rate")
ax5.legend(fontsize=7.5, loc="lower right"); ax5.grid(alpha=0.3)

# ── Panel 6 — Metrics Bar Chart ──
ax6 = fig.add_subplot(gs[2, 1])
metric_names = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
x = np.arange(len(metric_names)); width = 0.25
for i, (name, color) in enumerate(zip(results.keys(), MODEL_COLORS)):
    vals = [results[name][m] for m in metric_names]
    bars = ax6.bar(x + i*width, vals, width, label=name, color=color, alpha=0.85)
ax6.set_title("Model Metrics Comparison", fontweight="bold", pad=10)
ax6.set_xticks(x + width); ax6.set_xticklabels(metric_names, rotation=15, ha="right", fontsize=8)
ax6.set_ylim(0.5, 1.02); ax6.legend(fontsize=7.5); ax6.grid(axis="y", alpha=0.3); ax6.set_axisbelow(True)
ax6.axhline(1.0, color=PALETTE["muted"], lw=0.5, ls="--")

# ── Panel 7 — Confusion Matrix (Best Model) ──
ax7 = fig.add_subplot(gs[2, 2])
cm   = confusion_matrix(y_test, results[best_model_name]["y_pred"])
cmap_cm = sns.light_palette(PALETTE["accent"], as_cmap=True)
sns.heatmap(cm, annot=True, fmt="d", ax=ax7, cmap=cmap_cm,
            xticklabels=["Predicted Bad","Predicted Good"],
            yticklabels=["Actual Bad","Actual Good"],
            linewidths=1, linecolor=PALETTE["bg"],
            annot_kws={"size": 13, "weight": "bold"})
ax7.set_title(f"Confusion Matrix\n({best_model_name})", fontweight="bold", pad=10)
ax7.tick_params(labelsize=8)

# ── Panel 8 — Feature Importances (Random Forest) ──
ax8 = fig.add_subplot(gs[3, :2])
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
colors_fi = [PALETTE["accent"] if v >= importances.quantile(0.75) else PALETTE["muted"]
             for v in importances.values]
bars = ax8.barh(importances.index, importances.values, color=colors_fi, edgecolor="none", height=0.6)
ax8.set_title("Feature Importances — Random Forest", fontweight="bold", pad=10)
ax8.set_xlabel("Importance Score"); ax8.grid(axis="x", alpha=0.3); ax8.set_axisbelow(True)
for bar, val in zip(bars, importances.values):
    ax8.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=7.5, color=PALETTE["text"])

# ── Panel 9 — Cross-Validation AUC ──
ax9 = fig.add_subplot(gs[3, 2])
cv_means = [results[n]["CV-AUC"] for n in results]
cv_stds  = [results[n]["CV-Std"] for n in results]
y_pos    = np.arange(len(results))
ax9.barh(y_pos, cv_means, xerr=cv_stds, color=MODEL_COLORS, alpha=0.85,
         edgecolor="none", capsize=5, ecolor=PALETTE["muted"], height=0.5)
ax9.set_yticks(y_pos); ax9.set_yticklabels(list(results.keys()), fontsize=9)
ax9.set_xlabel("CV ROC-AUC Score"); ax9.set_title("5-Fold Cross-Validation AUC", fontweight="bold", pad=10)
ax9.set_xlim(0.5, 1.05); ax9.grid(axis="x", alpha=0.3); ax9.set_axisbelow(True)
ax9.axvline(1.0, color=PALETTE["muted"], lw=0.5, ls="--")
for y_p, mean, std in zip(y_pos, cv_means, cv_stds):
    ax9.text(mean + 0.005, y_p, f"{mean:.3f}±{std:.3f}", va="center", fontsize=8)

# ── Super Title ──
fig.suptitle("CREDIT SCORING MODEL — Full Analysis Dashboard",
             fontsize=18, fontweight="bold", color=PALETTE["accent"],
             y=0.995, fontfamily="monospace")

plt.savefig("/mnt/user-data/outputs/credit_scoring_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print("   ✔ Dashboard saved → credit_scoring_dashboard.png")


# ─────────────────────────────────────────────
#  PREDICTION FUNCTION (INFERENCE)
# ─────────────────────────────────────────────
def predict_creditworthiness(applicant: dict, verbose: bool = True) -> dict:
    """
    Predict creditworthiness for a new applicant.

    Parameters
    ----------
    applicant : dict with keys matching raw feature columns
    verbose   : print a human-readable report

    Returns
    -------
    dict with prediction, probability, risk level, and model scores
    """
    emp_map = {v: i for i, v in enumerate(le_emp.classes_)}
    edu_map = {v: i for i, v in enumerate(le_edu.classes_)}

    income       = applicant["Income"]
    loan_amount  = applicant["Loan_Amount"]
    savings      = applicant["Savings_Balance"]
    late         = applicant["Num_Late_Payments"]
    emp_years    = applicant["Employment_Years"]

    row = pd.DataFrame([{
        "Age":                    applicant["Age"],
        "Income":                 income,
        "Loan_Amount":            loan_amount,
        "Loan_Duration_Months":   applicant["Loan_Duration_Months"],
        "Num_Credit_Lines":       applicant["Num_Credit_Lines"],
        "Num_Late_Payments":      late,
        "Debt_to_Income_Ratio":   applicant["Debt_to_Income_Ratio"],
        "Employment_Years":       emp_years,
        "Savings_Balance":        savings,
        "Employment_Type_Enc":    emp_map.get(applicant["Employment_Type"], 0),
        "Education_Enc":          edu_map.get(applicant["Education"], 0),
        "Loan_to_Income":         round(loan_amount / income, 4),
        "Savings_to_Loan":        round(savings / (loan_amount + 1), 4),
        "Payment_History_Score":  round(1 / (late + 1), 4),
        "Creditworthiness_Index": round(income*0.3 + savings*0.2
                                        - loan_amount*0.15 - late*5000 + emp_years*1000, 2),
    }])

    model_scores = {}
    for mname, res in results.items():
        m = res["model"]
        X_in = scaler.transform(row) if mname == "Logistic Regression" else row
        prob = m.predict_proba(X_in)[0][1]
        model_scores[mname] = prob

    # Ensemble average
    ensemble_prob = np.mean(list(model_scores.values()))
    final_pred    = int(ensemble_prob >= 0.5)

    if   ensemble_prob >= 0.75: risk = "LOW RISK   ✅"
    elif ensemble_prob >= 0.50: risk = "MODERATE   ⚠️"
    elif ensemble_prob >= 0.30: risk = "HIGH RISK  🔶"
    else:                       risk = "VERY HIGH  ❌"

    if verbose:
        sep = "─" * 50
        print(f"\n{sep}")
        print(f"  CREDIT ASSESSMENT REPORT")
        print(sep)
        print(f"  Applicant Age  : {applicant['Age']}")
        print(f"  Income         : ${income:,}")
        print(f"  Loan Amount    : ${loan_amount:,}")
        print(f"  Savings        : ${savings:,}")
        print(f"  Employment     : {applicant['Employment_Type']} ({emp_years} yrs)")
        print(f"  Late Payments  : {late}")
        print(f"  D/I Ratio      : {applicant['Debt_to_Income_Ratio']:.2%}")
        print(sep)
        print("  Model Probabilities:")
        for mn, prob in model_scores.items():
            bar = "█" * int(prob * 20)
            print(f"    {mn:22s} {bar:<20} {prob:.3f}")
        print(sep)
        print(f"  Ensemble Probability : {ensemble_prob:.3f}")
        print(f"  Decision             : {'APPROVED ✅' if final_pred else 'DENIED ❌'}")
        print(f"  Risk Level           : {risk}")
        print(sep)

    return {
        "prediction": final_pred,
        "probability": round(ensemble_prob, 4),
        "risk_level": risk.strip(),
        "model_scores": model_scores,
    }


# ─────────────────────────────────────────────
#  DEMO PREDICTIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  DEMO PREDICTIONS")
print("=" * 65)

applicants = [
    {   # Should be APPROVED
        "Age": 38, "Income": 85000, "Loan_Amount": 12000,
        "Loan_Duration_Months": 36, "Num_Credit_Lines": 7,
        "Num_Late_Payments": 0, "Debt_to_Income_Ratio": 0.18,
        "Employment_Years": 12, "Savings_Balance": 25000,
        "Employment_Type": "Salaried", "Education": "Master's"
    },
    {   # Should be DENIED
        "Age": 24, "Income": 22000, "Loan_Amount": 18000,
        "Loan_Duration_Months": 60, "Num_Credit_Lines": 2,
        "Num_Late_Payments": 8, "Debt_to_Income_Ratio": 0.72,
        "Employment_Years": 1, "Savings_Balance": 800,
        "Employment_Type": "Self-Employed", "Education": "High School"
    },
    {   # Borderline
        "Age": 31, "Income": 50000, "Loan_Amount": 10000,
        "Loan_Duration_Months": 24, "Num_Credit_Lines": 4,
        "Num_Late_Payments": 2, "Debt_to_Income_Ratio": 0.35,
        "Employment_Years": 5, "Savings_Balance": 5000,
        "Employment_Type": "Salaried", "Education": "Bachelor's"
    },
]

for i, app in enumerate(applicants, 1):
    print(f"\n  ── Applicant #{i} ──")
    predict_creditworthiness(app)

print("\n" + "=" * 65)
print("  ALL DONE!")
print("  Output: credit_scoring_dashboard.png")
print("=" * 65)
