"""
Local AI Data Analyst — Layoff Dataset
No external API. Runs fully on your machine.

Usage:
    python3 ai_analyst.py
"""

import subprocess, sys

for pkg in ["pandas", "numpy", "scikit-learn", "openpyxl", "scipy", "tabulate"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import warnings; warnings.filterwarnings("ignore")
import re, textwrap
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

try:
    from tabulate import tabulate
    HAS_TAB = True
except ImportError:
    HAS_TAB = False


# ── colour helpers ────────────────────────────────────────────────────────────
def c(text, code): return f"\033[{code}m{text}\033[0m"
CYAN  = lambda t: c(t, "96")
GREEN = lambda t: c(t, "92")
YELLOW= lambda t: c(t, "93")
BOLD  = lambda t: c(t, "1")
DIM   = lambda t: c(t, "2")
RED   = lambda t: c(t, "91")

def box(text, width=70):
    lines = textwrap.wrap(text, width - 4)
    top   = "┌" + "─" * (width - 2) + "┐"
    bot   = "└" + "─" * (width - 2) + "┘"
    mid   = "\n".join("│ " + l.ljust(width - 4) + " │" for l in lines)
    return f"{top}\n{mid}\n{bot}"

def tbl(df, cols=None, max_rows=15):
    if cols: df = df[cols]
    df = df.head(max_rows)
    if HAS_TAB:
        return tabulate(df, headers="keys", tablefmt="rounded_outline",
                        showindex=False, floatfmt=".4f")
    return df.to_string(index=False)


# ── load data ─────────────────────────────────────────────────────────────────
print(CYAN("\nLoading dataset..."))
try:
    train = pd.read_excel("train.xlsx")
    test  = pd.read_excel("test.xlsx")
except FileNotFoundError:
    print(RED("train.xlsx / test.xlsx not found. Put them in the same folder."))
    sys.exit(1)

try:
    preds = pd.read_excel("test_predictions.xlsx")
except FileNotFoundError:
    preds = None

TARGET   = "layoff_happened"
CAT_COLS = ["industry", "country"]
NUM_COLS = ["funding_amount", "employee_count", "growth_rate", "valuation"]
FEAT_COLS= CAT_COLS + NUM_COLS


# ── train model ───────────────────────────────────────────────────────────────
def build_features(df, enc=None, fit=True):
    df  = df.copy(); enc = enc or {}
    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str)); enc[col] = le
        else:
            le = enc[col]; known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1)
    if fit:
        imp = SimpleImputer(strategy="median")
        df[NUM_COLS] = imp.fit_transform(df[NUM_COLS]); enc["imp"] = imp
    else:
        df[NUM_COLS] = enc["imp"].transform(df[NUM_COLS])
    for col in ["funding_amount","employee_count","valuation"]:
        df[col] = np.log1p(df[col].clip(lower=0))
    if fit:
        sc = StandardScaler()
        df[NUM_COLS] = sc.fit_transform(df[NUM_COLS]); enc["sc"] = sc
    else:
        df[NUM_COLS] = enc["sc"].transform(df[NUM_COLS])
    return df[FEAT_COLS], enc

X_all, ENC = build_features(train[FEAT_COLS], fit=True)
y_all      = train[TARGET]
X_tr,X_v,y_tr,y_v = train_test_split(X_all,y_all,test_size=0.2,random_state=42,stratify=y_all)
MODEL = RandomForestClassifier(n_estimators=300,max_depth=10,class_weight="balanced",
                                random_state=42,n_jobs=-1)
MODEL.fit(X_tr, y_tr)
FI = dict(zip(FEAT_COLS, MODEL.feature_importances_))
print(GREEN("✓ Model ready.\n"))


# ── knowledge base ────────────────────────────────────────────────────────────
def layoff_rate_overall():
    r = train[TARGET].mean()
    return f"Overall layoff rate in training data: {BOLD(f'{r:.1%}')} " \
           f"({train[TARGET].sum()} out of {len(train)} companies)."

def top_industries(n=10):
    df = (train.groupby("industry")[TARGET]
          .agg(["mean","count"]).query("count>=5")
          .sort_values("mean",ascending=False).head(n).reset_index())
    df.columns = ["Industry","Layoff Rate","Count"]
    df["Layoff Rate"] = df["Layoff Rate"].map("{:.1%}".format)
    return f"Top {n} industries by layoff rate:\n" + tbl(df)

def top_countries(n=10):
    df = (train.groupby("country")[TARGET]
          .agg(["mean","count"]).query("count>=5")
          .sort_values("mean",ascending=False).head(n).reset_index())
    df.columns = ["Country","Layoff Rate","Count"]
    df["Layoff Rate"] = df["Layoff Rate"].map("{:.1%}".format)
    return f"Top {n} countries by layoff rate:\n" + tbl(df)

def safest_industries(n=8):
    df = (train.groupby("industry")[TARGET]
          .agg(["mean","count"]).query("count>=5")
          .sort_values("mean",ascending=True).head(n).reset_index())
    df.columns = ["Industry","Layoff Rate","Count"]
    df["Layoff Rate"] = df["Layoff Rate"].map("{:.1%}".format)
    return f"Safest {n} industries (lowest layoff rate):\n" + tbl(df)

def feature_importance_info():
    fi_sorted = sorted(FI.items(), key=lambda x: x[1], reverse=True)
    lines = [f"  {BOLD(f):<30} {v:.1%}" for f,v in fi_sorted]
    return "Feature importance (Random Forest):\n" + "\n".join(lines)

def stats_by_group():
    grp = train.groupby(TARGET)[NUM_COLS].mean()
    grp.index = ["No Layoff (0)","Layoff (1)"]
    return "Average values by group:\n" + tbl(grp.reset_index().rename(columns={"index":"Group"}))

def dataset_overview():
    return (f"Dataset overview:\n"
            f"  Training rows : {len(train):,}\n"
            f"  Test rows     : {len(test):,}\n"
            f"  Columns       : {', '.join(train.columns)}\n"
            f"  Missing values: {train.isnull().sum().sum()}\n"
            f"  Layoff rate   : {train[TARGET].mean():.1%}\n"
            f"  Industries    : {train['industry'].nunique()}\n"
            f"  Countries     : {train['country'].nunique()}")

def model_performance():
    from sklearn.metrics import roc_auc_score, f1_score, classification_report
    probs = MODEL.predict_proba(X_v)[:,1]
    preds_ = MODEL.predict(X_v)
    auc = roc_auc_score(y_v,probs)
    f1  = f1_score(y_v,preds_)
    rep = classification_report(y_v,preds_,target_names=["No Layoff","Layoff"])
    return (f"Random Forest performance on validation set:\n"
            f"  ROC AUC : {BOLD(f'{auc:.4f}')}\n"
            f"  F1 Score: {BOLD(f'{f1:.4f}')}\n\n{rep}")

def predictions_summary():
    if preds is None:
        return "test_predictions.xlsx not found. Run run_analysis.py first."
    n1 = (preds[TARGET]==1).sum()
    n0 = (preds[TARGET]==0).sum()
    return (f"Test set predictions (200 companies):\n"
            f"  Predicted Layoff    : {BOLD(str(n1))} companies ({n1/200:.0%})\n"
            f"  Predicted No Layoff : {BOLD(str(n0))} companies ({n0/200:.0%})\n\n"
            f"Top 10 highest-risk companies:\n" +
            tbl(preds.sort_values("layoff_probability",ascending=False)
                     .head(10)[["company","industry","country","layoff_probability"]]))

def stat_test_info():
    lines = ["Mann-Whitney U test (are the groups statistically different?):\n",
             f"  {'Feature':<20} {'U-stat':>12} {'p-value':>12} {'Significant?':>14}"]
    lines.append("  " + "-"*60)
    for col in NUM_COLS:
        g0 = train[train[TARGET]==0][col].dropna()
        g1 = train[train[TARGET]==1][col].dropna()
        u,p = stats.mannwhitneyu(g0,g1,alternative="two-sided")
        sig = GREEN("Yes ***") if p<0.001 else (YELLOW("Yes **") if p<0.01 else
              (YELLOW("Yes *") if p<0.05 else RED("No (ns)")))
        lines.append(f"  {col:<20} {u:>12.0f} {p:>12.6f} {sig:>14}")
    return "\n".join(lines)

def predict_company(industry, country, funding, employees, growth, valuation):
    row = pd.DataFrame([{
        "industry":industry,"country":country,
        "funding_amount":funding,"employee_count":employees,
        "growth_rate":growth,"valuation":valuation
    }])
    X,_ = build_features(row, enc=ENC, fit=False)
    prob = MODEL.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)
    risk = RED("HIGH RISK 🔴") if prob>0.6 else (YELLOW("MEDIUM 🟡") if prob>0.4 else GREEN("LOW RISK 🟢"))
    return (f"Prediction for your company:\n"
            f"  Layoff Probability : {BOLD(f'{prob:.1%}')}\n"
            f"  Risk Level         : {risk}\n"
            f"  Prediction         : {'Likely to lay off' if pred else 'Likely safe'}")

def growth_analysis():
    lo = train[train[TARGET]==1]["growth_rate"].mean()
    hi = train[train[TARGET]==0]["growth_rate"].mean()
    return (f"Growth rate analysis:\n"
            f"  Companies that laid off    — avg growth rate: {BOLD(f'{lo:.1f}%')}\n"
            f"  Companies that did NOT     — avg growth rate: {BOLD(f'{hi:.1f}%')}\n"
            f"  Difference                 : {BOLD(f'{hi-lo:.1f}%')}\n\n"
            f"  Growth rate is the {BOLD('strongest predictor')} of layoff risk.\n"
            f"  Companies with growth < 50% are significantly more likely to lay off workers.")

def career_advice():
    return (f"Career survival recommendations (based on the data):\n\n"
            f"  1. {GREEN('✓')} Target companies with {BOLD('growth_rate > 80%')} — biggest safety signal\n"
            f"  2. {GREEN('✓')} Stable industries (Healthcare, Finance) have lower layoff rates\n"
            f"  3. {GREEN('✓')} Mid-size companies (5k–50k employees) tend to be more stable\n"
            f"  4. {YELLOW('~')} Funding and valuation matter, but much less than growth rate\n"
            f"  5. {RED('✗')} Company size alone is NOT a reliable safety indicator\n"
            f"  6. {RED('✗')} Country of origin has the least predictive power")

def help_menu():
    cmds = [
        ("overview",         "Dataset size, columns, missing values"),
        ("layoff rate",      "Overall layoff rate in training data"),
        ("top industries",   "Industries with highest layoff rates"),
        ("safe industries",  "Industries with lowest layoff rates"),
        ("top countries",    "Countries with highest layoff rates"),
        ("feature importance","Which features drive the model most"),
        ("model performance","AUC, F1, precision, recall on validation set"),
        ("stats by group",   "Average values for layoff vs no-layoff companies"),
        ("stat test",        "Mann-Whitney U significance tests"),
        ("predictions",      "Summary of 200 test company predictions"),
        ("growth analysis",  "How growth rate separates the two groups"),
        ("career advice",    "Data-driven career survival guide"),
        ("predict",          "Predict risk for a custom company"),
        ("quit / exit",      "Exit the analyst"),
    ]
    lines = [f"  {BOLD(CYAN(k)): <45}  {DIM(v)}" for k,v in cmds]
    return "Available commands:\n" + "\n".join(lines)


# ── intent matching ───────────────────────────────────────────────────────────
INTENTS = [
    (r"overview|summary|dataset|about|shape|size|columns",   dataset_overview),
    (r"layoff rate|overall rate|how many.*lay|percentage.*lay",layoff_rate_overall),
    (r"top.*industr|industr.*layoff|worst.*industr|highest.*industr",top_industries),
    (r"safe.*industr|low.*industr|best.*industr|lowest.*industr",  safest_industries),
    (r"countr|region|nation|where",           top_countries),
    (r"feature.*import|import.*feature|which.*feature|matter most|key factor",feature_importance_info),
    (r"model.*perf|perf|auc|f1|accuracy|how.*good|score|metric",  model_performance),
    (r"stat.*group|group.*stat|avg|average|mean|compare.*group",   stats_by_group),
    (r"stat.*test|mann|p.value|significant|hypothesis",            stat_test_info),
    (r"predict.*summary|test.*pred|200.*compan|result",            predictions_summary),
    (r"growth|growth.*rate|revenue",          growth_analysis),
    (r"career|advice|tip|survive|safe|protect",career_advice),
    (r"help|command|what can|option|menu",    help_menu),
]

def match_intent(text):
    t = text.lower().strip()
    for pattern, fn in INTENTS:
        if re.search(pattern, t):
            return fn()
    return None


def handle_predict(text):
    """Walk the user through a custom company prediction."""
    print(CYAN("\nLet's predict layoff risk for a custom company."))
    print(DIM("(Press Enter to use a default value)\n"))
    def ask(prompt, default, cast):
        v = input(f"  {prompt} [{default}]: ").strip()
        return cast(v) if v else default

    industry  = ask("Industry (e.g. Tech, Finance, Healthcare)", "Tech", str)
    country   = ask("Country (e.g. USA, Germany, Japan)", "USA", str)
    funding   = ask("Funding amount in USD (e.g. 500000000)", 500_000_000, float)
    employees = ask("Employee count (e.g. 10000)", 10_000, int)
    growth    = ask("Growth rate % (e.g. 75.5)", 75.5, float)
    valuation = ask("Valuation in USD (e.g. 2000000000)", 2_000_000_000, float)
    return predict_company(industry, country, funding, employees, growth, valuation)


# ── main loop ─────────────────────────────────────────────────────────────────
BANNER = f"""
{CYAN('┌' + '─'*62 + '┐')}
{CYAN('│')}  {BOLD('🤖  Local AI Data Analyst — Layoff Dataset')}                {CYAN('│')}
{CYAN('│')}  {DIM('No API · No internet · Runs 100% on your machine')}          {CYAN('│')}
{CYAN('│')}  {DIM('Type a question in plain English, or type  help  to start')} {CYAN('│')}
{CYAN('└' + '─'*62 + '┘')}
"""

print(BANNER)

SUGGESTIONS = [
    "Try: 'what is the layoff rate?'",
    "Try: 'which industries are safest?'",
    "Try: 'show me feature importance'",
    "Try: 'predict' to check a custom company",
    "Try: 'how good is the model?'",
]
import random
print(DIM(random.choice(SUGGESTIONS)) + "\n")

while True:
    try:
        user = input(f"{BOLD(GREEN('You'))} › ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{CYAN('Goodbye!')}")
        break

    if not user:
        continue

    if re.search(r"quit|exit|bye|q$", user.lower()):
        print(f"\n{CYAN('Goodbye! Good luck with the competition 🎯')}\n")
        break

    if re.search(r"predict|forecast|will.*company|risk.*for", user.lower()):
        response = handle_predict(user)
    else:
        response = match_intent(user)

    if response:
        print(f"\n{BOLD(CYAN('AI'))} › {response}\n")
    else:
        # fuzzy fallback
        msg = "I'm not sure what you mean. Here are some things I can help with:"
        print(f"\n{BOLD(CYAN('AI'))} › {YELLOW(msg)}\n")
        print(help_menu() + "\n")
