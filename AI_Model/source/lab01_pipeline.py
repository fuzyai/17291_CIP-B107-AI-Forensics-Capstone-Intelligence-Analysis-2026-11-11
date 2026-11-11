"""CIP-B107-LAB01 - offline IP-evidence triage pipeline (Phases A, B, C, E).
Run:  python lab01_pipeline.py <path_to_folder_with_supplied_files>
Outputs go to ./out/.  Reads originals only; never modifies them."""
import sys, os, json, hashlib, ipaddress, re, datetime as dt, platform
import pandas as pd, numpy as np, sklearn, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = "out"; os.makedirs(OUT, exist_ok=True)
SEED = 42; np.random.seed(SEED)
p = lambda f: os.path.join(SRC, f)

# ---------- Phase A: manifest ----------
tz_wat = dt.timezone(dt.timedelta(hours=1))
rows = []
for f in ["dataset_small.csv","dataset_train.jsonl","dataset_test.jsonl","result.csv","fine_tuning_steps.txt"]:
    b = open(p(f), "rb").read(); now = dt.datetime.now(dt.timezone.utc)
    rows.append(dict(filename=f, sha256=hashlib.sha256(b).hexdigest(), bytes=len(b),
                     timestamp_utc=now.isoformat(timespec="seconds"),
                     timestamp_wat=now.astimezone(tz_wat).isoformat(timespec="seconds"),
                     transfer_source="ICDFA supplied package", examiner="<YOUR NAME>"))
pd.DataFrame(rows).to_csv(f"{OUT}/evidence_manifest.csv", index=False)

df = pd.read_csv(p("dataset_small.csv"))
print("records", len(df), "| columns", list(df.columns))
print(df["class"].value_counts().to_dict(), "| nulls", int(df.isna().sum().sum()),
      "| duplicate rows", int(df.duplicated().sum()))
load = lambda f: pd.DataFrame([json.loads(l) for l in open(p(f)) if l.strip()])
tr, te = load("dataset_train.jsonl"), load("dataset_test.jsonl")
tr["y"] = (tr.completion.str.strip() == "positive").astype(int)
te["y"] = (te.completion.str.strip() == "positive").astype(int)
print("train", len(tr), tr.y.value_counts().to_dict(), "| test", len(te), te.y.value_counts().to_dict())

# ---------- Phase B: regex vs ipaddress vs source label ----------
RX = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
def strict(s):
    try: ipaddress.IPv4Address(s); return True
    except Exception: return False
rev = []
for i, r in df.iterrows():
    m = RX.findall(r.text)
    rev.append(dict(source_row=i+2, context=r.text.strip(), candidate=r.ip,
                    regex_match=r.ip in m, ipv4_valid=strict(str(r.ip)),
                    source_label="positive" if r["class"] != "N" else "negative", analyst_note=""))
rev = pd.DataFrame(rev)
def note(r):
    if r.ipv4_valid and r.source_label == "negative": return "Strictly valid IPv4 but source says negative - context/version-like string; review"
    if (not r.ipv4_valid) and r.source_label == "positive": return "Source positive but fails strict IPv4 (e.g. leading zeros/octet) - preserve disagreement"
    if not r.regex_match: return "Candidate not matched by simple regex in context"
    return ""
rev["analyst_note"] = rev.apply(note, axis=1)
rev.to_csv(f"{OUT}/candidate_review.csv", index=False)
print(pd.crosstab(rev.source_label, rev.ipv4_valid, rownames=["source"], colnames=["strict_valid"]))

# ---------- Phase C: leakage + models ----------
overlap = te.prompt.isin(set(tr.prompt)).sum()
print(f"LEAKAGE: {overlap}/{len(te)} test prompts are exact duplicates of training prompts")

def rule(prompt):
    ctx, cand = prompt.rsplit(";", 1); cand = cand.replace("->", "").strip()
    if not strict(cand): return 0
    # candidate must be a standalone token, not glued to digits/letters/dots
    pat = r"(?<![\w.])" + re.escape(cand) + r"(?![\w.])"
    return int(re.search(pat, ctx) is not None)
te["rule_pred"] = te.prompt.map(rule)

ml = Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1,4), lowercase=False)),
               ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))])
ml.fit(tr.prompt, tr.y)
joblib.dump(ml, f"{OUT}/ip_evidence_model.joblib")
print("Saved trained model to", f"{OUT}/ip_evidence_model.joblib")
te["ml_pred"] = ml.predict(te.prompt); te["ml_score"] = ml.predict_proba(te.prompt)[:,1]
maj = DummyClassifier(strategy="most_frequent").fit(tr.prompt, tr.y); te["majority_pred"] = maj.predict(te.prompt)

# ---------- Phase E: evaluation ----------
def cell(a, b): return {(1,1):"TP",(0,0):"TN",(0,1):"FP",(1,0):"FN"}[(a,b)]
te["actual"] = te.y.map({1:"positive",0:"negative"}); te["ml_pred_label"] = te.ml_pred.map({1:"positive",0:"negative"})
te["outcome"] = [cell(a,b) for a,b in zip(te.y, te.ml_pred)]
te[["prompt","actual","ml_pred_label","ml_score","outcome","rule_pred","majority_pred"]].to_csv(f"{OUT}/predictions.csv", index=False)
res = []
for name, col in [("majority","majority_pred"),("rule (regex+ipaddress)","rule_pred"),("TF-IDF char + LogReg","ml_pred")]:
    cm = confusion_matrix(te.y, te[col], labels=[0,1]); tn, fp, fn, tp = cm.ravel()
    res.append(dict(model=name, TP=tp, FP=fp, FN=fn, TN=tn, accuracy=accuracy_score(te.y, te[col]),
        precision=precision_score(te.y, te[col], zero_division=0), recall=recall_score(te.y, te[col]),
        f1=f1_score(te.y, te[col], zero_division=0)))
res = pd.DataFrame(res).round(4); res.to_csv(f"{OUT}/metrics_summary.csv", index=False); print(res.to_string(index=False))
print("ML AUROC %.4f | AUPRC %.4f" % (roc_auc_score(te.y, te.ml_score), average_precision_score(te.y, te.ml_score)))
print(classification_report(te.y, te.ml_pred, target_names=["negative","positive"], zero_division=0))

fig, ax = plt.subplots(1, 2, figsize=(9,3.8))
for a, (t, c) in zip(ax, [("TF-IDF + LogReg","ml_pred"),("Rule baseline","rule_pred")]):
    cm = confusion_matrix(te.y, te[c], labels=[1,0]); a.imshow(cm, cmap="Blues"); a.set_title(t)
    a.set_xticks([0,1], ["positive","negative"]); a.set_yticks([0,1], ["positive","negative"])
    a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    for (i,j),v in np.ndenumerate(cm): a.text(j,i,v,ha="center",va="center",fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150)
open(f"{OUT}/environment.txt","w").write(f"python {platform.python_version()}\npandas {pd.__version__}\nnumpy {np.__version__}\nscikit-learn {sklearn.__version__}\nseed {SEED}\n")
