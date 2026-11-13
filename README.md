# CIP-B107 AI for Forensics and Capstone Intelligence
## Case Study: AI-Assisted IPv4 Evidence Identification and Triage

### 1. Project Information

| Field | Details |
|---|---|
| Course | AI for Forensics and Capstone Intelligence |
| Case ID | ICDFA-DFIR-AI-026 |
| Lab Reference | CIP-B107-LAB01 |
| Investigation | AI-Assisted IPv4 Evidence Identification and Triage |
| Environment | Kali Linux, Python 3.x, virtual environment |
| Main Libraries | pandas, scikit-learn, matplotlib |
| Random Seed | 42 |
| Collection Date | 27 October 2026 |
| Evidence Type | Simulated mobile-device configuration export |

### 2. Project Overview

This case study documents the design and evaluation of an AI-assisted workflow for identifying candidate IPv4 evidence from a simulated mobile-device configuration export.

The investigation compares regular-expression extraction and strict IPv4 validation with an offline machine-learning classifier using TF-IDF character n-grams and Logistic Regression.

The objective was to distinguish potential IPv4 evidence from misleading text, including software versions, invalid octets, embedded addresses, and numeric fragments.

The workflow also examines historical fine-tuning logs and evaluates the limitations of the supplied training and testing dataset.

**Important forensic caveat:** The supplied files are teaching artifacts. A positive classification indicates a source-annotated candidate, not proof that an address is malicious, routable, public, or associated with suspicious activity. Independent corroboration is required before drawing investigative conclusions.

### 3. Evidence Manifest and Integrity

Five original evidence files were preserved in an isolated originals directory. The examination records describe restricted file and directory permissions and SHA-256 verification before processing.

| Evidence file | Size (bytes) | SHA-256 |
|---|---:|---|
| `dataset_small.csv` | 16,324 | `74dfb1fd05b9d277f1dda62d002cdd3d81ff6ac754d04c8f621633206ab3e56` |
| `dataset_test.jsonl` | 4,608 | `7c78cee96cf5f2475bc894713b5b4b5b631b33aaa684665fe2e59663e21e22` |
| `dataset_train.jsonl` | 21,860 | `93938b9d9b26b7dd1261d19a8e374c10fe50b5e0733f415572a0fdb8f035c27` |
| `fine_tuning_steps.txt` | 31,869 | `863c43f1029fbd6e5e62516e0beb07b544f7a413bc315ccf3c08e782bd60cd` |
| `result.csv` | 41,353 | `29b85f57d78515c7c8ad529bee77e91dd1555e4d515d43000120eafb58` |

Collection timestamp recorded in the report: 27 October 2026, 14:34:29 WAT.

The full evidence manifest, including transfer source and examiner information, is maintained separately as `evidence_manifest.csv`.

### 4. Methodology

The workflow consisted of five main phases.

#### Phase A: Dataset Profiling

The dataset was loaded using pandas and examined for record counts, missing values, duplicate rows, and source-label distribution.

- Total records: 229
- Columns: `ip`, `class`, `text`
- Positive source labels: 177
- Negative source labels: 32
- Other source classes: B = 12, C = 4, E = 4
- Missing values: 0
- Duplicate rows: 77

Strict IPv4 validation was performed using Python's `ipaddress.IPv4Address`.

The validation identified disagreements between source annotations and strict IPv4 syntax:

- Four source-positive records failed strict IPv4 validation.
- Eight source-negative records passed strict IPv4 validation.

These disagreements were retained for investigation rather than silently relabelled.

The JSONL files were also checked line by line. The split contained 183 training records and 46 testing records.

#### Phase B: Candidate Extraction and Validation

A regular expression was used to extract dotted-decimal candidates:

`(\b(?:\d{1,3}\.){3}\d{1,3}\b)`

Each extracted candidate was independently checked using strict IPv4 validation. Original text context was retained to support analyst review.

The resulting `candidate_review.csv` contained 229 records and the following fields:

- `source_row`
- `context`
- `candidate`
- `regex_match`
- `ipv4_valid`
- `source_label`
- `analyst_note`

Controlled tests examined genuine IPv4 addresses, software-version strings, invalid octets, URL-embedded addresses, and longer dotted numeric sequences.

The evidence run identified several overlapping categories, including valid candidates, invalid octets, version-like strings, URL-embedded candidates, and numeric contamination. These categories are contextual review findings and should not be treated as mutually exclusive totals.

#### Phase C: Offline Machine-Learning Classifier

An offline text-classification pipeline was developed using:

- TF-IDF character n-grams, with an n-gram range of 1–4.
- Logistic Regression.
- `max_iter=2000`.
- `class_weight='balanced'`.
- `random_state=42`.

The vectorizer was fitted using training records only.

A transparent rule-based baseline was also evaluated. It classified a candidate as positive only when it passed strict IPv4 validation and appeared as a standalone token in the text.

A significant limitation was identified: **30 of the 46 testing prompts were exact duplicates of training prompts.** This creates a data-leakage risk and limits the reliability of the reported holdout performance as evidence of generalisation.

#### Phase D: Historical Fine-Tuning Log Examination

The historical files `fine_tuning_steps.txt` and `result.csv` were parsed to examine previously recorded training metrics.

The logs contained 734 training steps and 733 result steps. Classification metrics were recorded at only four checkpoints, while validation loss and validation sequence accuracy appeared at 92 checkpoints.

Recorded final metrics at step 733 included:

| Metric | Recorded value |
|---|---:|
| Accuracy | 0.9783 |
| Precision | 0.9714 |
| Recall | 1.0000 |
| AUROC | 0.9975 |
| AUPRC | 0.9991 |
| F1-score | 0.9855 |

Training curves showed loss spikes at several points. However, the sparse classification checkpoints limit conclusions about training stability and performance over time.

These are historical results, not a newly executed fine-tuning experiment or an independent evaluation.

The historical workflow used legacy `ada` fine-tuning endpoints and embedded credentials in its original scripts. Those scripts were not executed during this examination. The old API patterns should not be treated as current implementation guidance.

#### Phase E: Evaluation and Analyst Review

The machine-learning model, rule-based baseline, and majority-class baseline were evaluated on the same 46-record holdout.

Confusion matrices, classification metrics, predictions, and analyst-review records were exported for reproducibility.

### 5. Results

#### 5.1 Holdout Evaluation

**TF-IDF Character N-grams + Logistic Regression**

| Metric | Result |
|---|---:|
| True Positives | 34 |
| False Positives | 1 |
| False Negatives | 0 |
| True Negatives | 11 |
| Accuracy | 97.83% |
| Precision | 97.14% |
| Recall | 100.00% |
| F1-score | 98.55% |
| AUROC | 0.9902 |
| AUPRC | 0.9965 |

**Rule-Based Baseline**

| Metric | Result |
|---|---:|
| True Positives | 21 |
| False Positives | 2 |
| False Negatives | 13 |
| True Negatives | 10 |
| Accuracy | 67.39% |
| Precision | 91.30% |
| Recall | 61.76% |
| F1-score | 73.68% |

**Majority-Class Baseline**

| Metric | Result |
|---|---:|
| True Positives | 34 |
| False Positives | 12 |
| False Negatives | 0 |
| True Negatives | 0 |
| Accuracy | 73.91% |
| Precision | 73.91% |
| Recall | 100.00% |
| F1-score | 85.00% |

The machine-learning classifier produced one false positive and no false negatives on the supplied holdout. However, the duplicate leakage means these figures should not be interpreted as reliable evidence of performance on unseen records.

The majority-class baseline demonstrates why accuracy alone is insufficient. Predicting every record as positive achieved 73.91% accuracy but failed to identify any negative records.

#### 5.2 Analyst Review Findings

Eight representative records were examined to understand model behaviour and disagreements between source labels, strict validation, and contextual interpretation.

Key findings included:

- A source-positive numeric fragment failed strict IPv4 validation.
- A long dotted numeric sequence produced an invalid extracted candidate.
- `0.0.0.0` passed syntactic validation but was source-labelled negative and was not treated as a meaningful evidence candidate in its context.
- An IPv4 candidate embedded in a longer URL sequence was correctly identified by the ML classifier but missed by the rule baseline.
- A software-version string was syntactically valid but contextually represented a version rather than an IP address.
- The ML model produced a false positive for `256.0.1.26`, which contains an invalid octet.
- Another invalid-octet sequence was correctly rejected by the model.
- `255.255.255.255` was correctly identified as a candidate, but its classification does not establish malicious activity.

The analyst review demonstrates that syntactic validity, source annotation, and contextual evidence interpretation are different questions.

### 6. Limitations and Forensic Considerations

#### 6.1 Duplicate Data Leakage

Thirty of the 46 test prompts were exact duplicates of training prompts. The reported ML metrics are therefore potentially optimistic.

A new evaluation should use deduplication and a leakage-free split. Where records originate from the same source or template, grouping should also be considered to reduce related-record leakage.

#### 6.2 Class Imbalance

The training dataset contained 163 positive and 20 negative records.

A classifier can achieve high accuracy by favouring the dominant class. Precision, recall, F1-score, class-specific results, and confusion matrices must therefore be considered together.

#### 6.3 Model and Rule-Based Limitations

The rule-based method is transparent and reproducible but missed 13 positive holdout records, including candidates embedded in longer text.

The ML classifier performed better on the supplied holdout but produced a false positive for an invalid-octet string. It also requires further testing before it can be considered reliable on unseen data.

#### 6.4 Historical Fine-Tuning Limitations

The historical fine-tuning logs cannot be independently reproduced from the supplied results alone. The classification metrics were recorded at only four checkpoints, and the original workflow relied on obsolete API patterns.

The historical metrics must not be presented as a new model run or as independent evidence of generalisation.

#### 6.5 Attribution and Maliciousness

Identifying an IPv4 candidate does not establish:

- That the address was active or reachable.
- That it was public rather than private or reserved.
- That it was communicating with the examined device.
- That it was associated with a particular person.
- That it was malicious or involved in an offence.

Further corroboration would require configuration provenance, network records, timestamps, and other relevant evidence.

### 7. Recommendations

The examination supports using offline machine learning as an analyst-assistance tool rather than an autonomous forensic decision-maker.

Recommended improvements include:

1. Rebuild the training and testing split to eliminate exact duplicate leakage.
2. Evaluate the model on genuinely unseen records and, where possible, separate source groups.
3. Add a strict IPv4 validation guard to suppress invalid-octet predictions.
4. Preserve source-positive but invalid candidates for manual review rather than automatically discarding them.
5. Require analyst review of ML-positive candidates before investigative conclusions are drawn.
6. Test performance on additional address contexts, software versions, malformed strings, and URL-embedded candidates.
7. Report precision, recall, F1-score, AUROC, AUPRC, and confusion matrices together.
8. Retire the historical legacy fine-tuning workflow and document secure, supported API practices if future fine-tuning is pursued.
9. Corroborate candidate addresses against device configuration provenance and relevant network logs.

### 8. Conclusion

This case study demonstrates an offline AI-assisted approach to identifying candidate IPv4 evidence in simulated mobile-device configuration data.

The TF-IDF character n-gram and Logistic Regression classifier achieved 97.83% accuracy and 98.55% F1-score on the supplied holdout. The rule-based baseline achieved 67.39% accuracy and missed 13 positive records.

However, 30 of the 46 testing prompts were exact duplicates of training prompts. This materially limits the strength of the reported ML results and makes leakage-free evaluation essential before any operational use.

The investigation also demonstrates that strict IPv4 validity alone does not establish evidential relevance. Contextual interpretation and human review remain necessary.

**Final finding:** AI can assist forensic triage by ranking and identifying candidate IP evidence, but its outputs must be validated and corroborated. No candidate address should be classified as malicious solely because of a model prediction or syntactic match.

### 9. Repository Structure

The case study can be organised as follows:

```text
CIP_B107_IP_Evidence_AI_Triage/
├── README.md
├── Report/
│   └── AI_Forensic_Triage_Examination_Report.pdf
├── Evidence_Screenshots/
│   ├── Evidence_Integrity/
│   ├── Dataset_Profiling/
│   ├── Candidate_Extraction/
│   ├── Model_Training/
│   ├── Holdout_Evaluation/
│   ├── Historical_Fine_Tuning/
│   └── Analyst_Review/
├── Commands/
│   └── forensic_commands.md
├── Supporting_Materials/
│   ├── findings_summary.md
│   ├── evidence_register.md
│   └── environment.txt
└── Results/
    ├── evidence_manifest.csv
    ├── candidate_review.csv
    ├── predictions.csv
    ├── metrics_summary.csv
    ├── historical_final_metrics.csv
    ├── confusion_matrices.png
    └── historical_training_curves.png
```

The original evidence files should remain in a restricted location and should not be uploaded publicly unless authorised. Public repository materials should exclude credentials, personal identifiers, and sensitive evidence.

### 10. Reproducibility and Deliverables

The examination produced the following outputs:

| Deliverable | Location |
|---|---|
| Evidence manifest | `out/evidence_manifest.csv` |
| Candidate review | `out/candidate_review.csv` |
| Holdout predictions | `out/predictions.csv` |
| Metrics summary | `out/metrics_summary.csv` |
| Confusion matrices | `out/confusion_matrices.png` |
| Historical training curves | `out/historical_training_curves.png` |
| Historical final metrics | `out/historical_final_metrics.csv` |
| Environment details | `out/environment.txt` |

Main scripts included:

- `lab01_pipeline.py`
- `lab01_phaseD_historical.py`
- `phase_b_evidence.py`
- `phase_b_controlled_tests.py`
- `investigate.py`
- `octet_check.py`
- `leakage_check.py`

The recorded random seed was 42.

The documented reproduction command, executed from the `work/` directory, was:

```bash
python lab01_pipeline.py ../originals
```

**Academic context:** This repository documents a practical AI-assisted digital forensics laboratory exercise. It records the methodology, evidence handling, model evaluation, limitations, and recommendations for further investigation.
