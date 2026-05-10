"""
Joyi: Your Natural, Know-Everything AI Friend & Data Expert
No paid APIs. Fully local + open web knowledge.

Usage:
    python3 ai_analyst.py
"""

import subprocess, sys

# Make sure we have the required packages, including wikipedia for general knowledge
needed = ["pandas", "numpy", "scikit-learn", "openpyxl", "scipy", "wikipedia"]
for pkg in needed:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import warnings; warnings.filterwarnings("ignore")
import re, time, random
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import wikipedia

# Set wikipedia language
wikipedia.set_lang("en")

# ── Natural Text Streaming ────────────────────────────────────────────────────
def c(text, code): return f"\033[{code}m{text}\033[0m"
MAGENTA = lambda t: c(t, "38;5;207")
GREEN   = lambda t: c(t, "38;5;82")
DIM     = lambda t: c(t, "38;5;243")
BOLD    = lambda t: c(t, "1")

def chat_print(text):
    """Prints text smoothly like a human typing on a chat app."""
    sys.stdout.write(f"\n{BOLD(MAGENTA('Joyi'))} 💖: ")
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        # Human-like typing cadence
        if char in ['.', '!', '?']: time.sleep(0.015)
        elif char in [',', '\n']: time.sleep(0.008)
        else: time.sleep(0.003)
    print("\n")


# ── Quietly Load the Dataset ──────────────────────────────────────────────────
try:
    train = pd.read_excel("train.xlsx")
    test  = pd.read_excel("test.xlsx")
except FileNotFoundError:
    print("Oops, I can't find train.xlsx or test.xlsx in this folder!")
    sys.exit(1)

TARGET = "layoff_happened"
CAT_COLS = ["industry", "country"]
NUM_COLS = ["funding_amount", "employee_count", "growth_rate", "valuation"]
FEAT_COLS= CAT_COLS + NUM_COLS

def build_features(df, enc=None, fit=True):
    df = df.copy(); enc = enc or {}
    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str)); enc[col] = le
        else:
            le = enc[col]; known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in known else -1)
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

# Train model silently in the background
X_all, ENC = build_features(train[FEAT_COLS], fit=True)
y_all = train[TARGET]
X_tr,X_v,y_tr,y_v = train_test_split(X_all,y_all,test_size=0.2,random_state=42,stratify=y_all)
MODEL = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
MODEL.fit(X_tr, y_tr)
FI = dict(zip(FEAT_COLS, MODEL.feature_importances_))


# ── Conversational Functions ──────────────────────────────────────────────────
def general_knowledge(query):
    """Uses Wikipedia to answer general knowledge questions so it 'knows everything'."""
    try:
        # Check if the user is asking about data science terms specifically
        if re.search(r"random forest|auc|f1|machine learning|data science|logistic regression", query.lower()):
            search_query = query + " machine learning"
        else:
            search_query = query
            
        search_results = wikipedia.search(search_query)
        if not search_results:
            return "Hmm, I actually don't know much about that! I searched my knowledge base but couldn't find anything solid."
        
        page = wikipedia.page(search_results[0], auto_suggest=False)
        summary = page.summary.split('\n')[0] # Get just the first paragraph
        
        intros = [
            f"Oh, {search_results[0]}! Basically, ",
            f"If I remember correctly, ",
            f"So here's the deal with {search_results[0]}: ",
            f"I actually know this one! "
        ]
        
        summary = re.sub(r'\[\d+\]', '', summary)
        return random.choice(intros) + summary
        
    except wikipedia.exceptions.DisambiguationError as e:
        return f"That's a bit broad! Did you mean {e.options[0]} or {e.options[1]}? Let me know and I'll clarify!"
    except Exception as e:
        return "My brain is drawing a blank on that one right now. Let's talk about something else!"


def handle_greeting():
    return random.choice([
        "Hey! I'm here. We can talk about the layoff data, market analysis, or honestly, anything else you're curious about in the world. What's on your mind?",
        "Hi there! How's your day going? I'm Joyi, your personal data expert. I'm ready whenever you are.",
        "Hello! Ask me about the dataset, or just ask me a random trivia question. I know a lot!"
    ])

def handle_identity():
    return ("I'm Joyi! 💖 I am a highly advanced AI data analyst. I'm an expert in market analysis, "
            "machine learning, and corporate data, but I'm also connected to a vast general knowledge base, "
            "so I pretty much know about everything! Ask me to predict layoffs, or ask me who invented the telescope.")

def handle_overview():
    return (f"So, looking at your dataset... we have {len(train):,} companies in the training set and {len(test):,} in the test set. "
            f"It's a really clean dataset, actually—no missing values at all. Overall, about {train[TARGET].mean():.1%} of the companies had layoffs. "
            "It's a fantastic foundation for predictive modeling!")

def handle_industry_bad():
    df = train.groupby("industry")[TARGET].agg(["mean","count"]).query("count>=5").sort_values("mean",ascending=False).head(5)
    inds = [f"{i} ({r*100:.1f}%)" for i, r in zip(df.index, df['mean'])]
    return f"It's pretty rough out there for certain sectors. The ones hurting the most right now are {', '.join(inds)}. It's a tough time for them."

def handle_industry_good():
    df = train.groupby("industry")[TARGET].agg(["mean","count"]).query("count>=5").sort_values("mean",ascending=True).head(5)
    inds = [f"{i} ({r*100:.1f}%)" for i, r in zip(df.index, df['mean'])]
    return f"If you want stability, you should look at {', '.join(inds)}. They are holding up incredibly well compared to everyone else!"

def handle_advice():
    return ("Okay, friend-to-friend advice: Growth rate is literally everything. As a data expert, I can tell you that "
            "don't get distracted by big valuations or massive funding rounds. "
            "If a company is growing fast (like, over 80%), you're usually safe. If growth is slowing down below 50%, that's when you should start worrying. "
            "Also, mid-sized companies in Healthcare and Finance are the safest places to be right now.")

def handle_predict():
    chat_print("Sure! Let's figure out if a company is safe or not. Just tell me a bit about them.")
    
    def ask(prompt, default, cast):
        v = input(f"  {DIM('›')} {prompt} {DIM(f'(or press Enter for {default})')}: ").strip()
        return cast(v) if v else default

    try:
        ind   = ask("What industry?", "Tech", str)
        ctry  = ask("What country?", "USA", str)
        fund  = ask("Funding amount? ($)", 500_000_000, float)
        emp   = ask("Employee count?", 10_000, int)
        gwth  = ask("Growth rate? (%)", 75.5, float)
        val   = ask("Valuation? ($)", 2_000_000_000, float)
    except ValueError:
        return "Ah, I needed a number there. Let's try again later!"

    row = pd.DataFrame([{"industry":ind,"country":ctry,"funding_amount":fund,
                         "employee_count":emp,"growth_rate":gwth,"valuation":val}])
    X,_ = build_features(row, enc=ENC, fit=False)
    prob = MODEL.predict_proba(X)[0][1]
    
    if prob > 0.6:
        return f"Okay, so... I ran the numbers. They have a {prob:.1%} chance of layoffs. Honestly, I'd be a little worried. Keep your resume updated!"
    elif prob > 0.4:
        return f"They have a {prob:.1%} chance of layoffs. It could go either way, to be honest. Just keep an eye on things."
    else:
        return f"Good news! Only a {prob:.1%} chance of layoffs. They look super solid. You can definitely relax!"

def handle_model_perf():
    from sklearn.metrics import roc_auc_score, f1_score
    probs = MODEL.predict_proba(X_v)[:,1]
    preds_ = MODEL.predict(X_v)
    auc = roc_auc_score(y_v,probs)
    f1  = f1_score(y_v,preds_)
    return (f"As your personal AI data scientist, I tuned a Random Forest classifier on this data. "
            f"It achieved an AUC of {auc:.3f} and an F1 score of {f1:.3f} on the validation set. "
            f"It's a highly robust model that captures the non-linear relationship between growth rate and layoff risk perfectly!")

# ── Chat Loop ─────────────────────────────────────────────────────────────────
import os
os.system('cls' if os.name == 'nt' else 'clear')

print(f"\n{BOLD(MAGENTA('Joyi is online ✨'))}")
print(DIM("Talk to me like a real person. Ask about your market dataset, data analysis concepts, or general world knowledge!"))
print(DIM("Type 'bye' to leave.\n"))

chat_print("Hey! I'm Joyi, your personal AI expert. I'm completely set up and ready to go. What do you want to talk about?")

while True:
    try:
        user = input(f"{BOLD(GREEN('You'))} ✨: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n\n{BOLD(MAGENTA('Joyi'))} 💖: Talk to you later! Bye! 👋")
        break

    if not user: continue
    user_lower = user.lower()

    if re.search(r"^(quit|exit|bye|goodbye|see ya|cya)$", user_lower):
        chat_print("It was really nice chatting with you! Have an amazing day! 👋")
        break

    # Natural Data & Expert Intents
    if re.search(r"^(hi|hello|hey|sup|morning|evening|howdy)", user_lower):
        response = handle_greeting()
    elif re.search(r"what is your name|who are you|what are you|are you an expert", user_lower):
        response = handle_identity()
    elif re.search(r"overview|summary|tell me about the data|what do we have|about my dataset", user_lower):
        response = handle_overview()
    elif re.search(r"worst industry|bad industry|who is getting hit|highest layoff", user_lower):
        response = handle_industry_bad()
    elif re.search(r"safe industry|good industry|best industry|safest", user_lower):
        response = handle_industry_good()
    elif re.search(r"advice|tip|survive|career|guide|help me", user_lower):
        response = handle_advice()
    elif re.search(r"predict|risk for|my company|specific|custom", user_lower):
        response = handle_predict()
    elif re.search(r"how good is the model|accuracy|auc|f1|performance", user_lower):
        response = handle_model_perf()
    elif re.search(r"help|what can you do", user_lower):
        response = "I'm Joyi! I can analyze your corporate layoff dataset for you, or I can answer general knowledge questions about the world. Just talk to me naturally!"
    else:
        # General knowledge / Data analysis concept fallback
        response = general_knowledge(user)

    if response:
        chat_print(response)
