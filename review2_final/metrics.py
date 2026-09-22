"""Strict typed document-level entity scoring and explicit token calibration metrics."""
from collections import Counter
import numpy as np
from .m1 import spans


def entity_metrics(records, predictions, categories):
    if len(records) != len(predictions):
        raise ValueError("Document count mismatch")
    gold, pred, true = Counter(), Counter(), Counter()
    negative, false_positive_docs, positive, missed_docs, invalid = 0, 0, 0, 0, 0
    for row, labels in zip(records, predictions):
        if len(labels) != len(row["labels"]):
            raise ValueError("Token coverage mismatch")
        gs, ps = set(spans(row["labels"])), set(spans(labels, strict=False))
        try:
            spans(labels)
        except ValueError:
            invalid += 1
        gold.update(k for a, b, k in gs)
        pred.update(k for a, b, k in ps)
        true.update(k for a, b, k in gs & ps)
        if gs:
            positive += 1
            missed_docs += not bool(ps)
        else:
            negative += 1
            false_positive_docs += bool(ps)
    def score(tp, ng, npred):
        p, r = tp/npred if npred else 0., tp/ng if ng else None
        return {"precision": p, "recall": r,
                "f1": 2*tp/(ng+npred) if ng+npred else None,
                "support": ng, "predicted": npred, "true_positive": tp}
    per = {k: score(true[k], gold[k], pred[k]) for k in categories}
    supported = [v["f1"] for v in per.values() if v["support"]]
    return {"micro": score(sum(true.values()), sum(gold.values()), sum(pred.values())),
            "macro_f1_supported_classes": float(np.mean(supported)) if supported else None,
            "per_class": per, "pii_positive_documents": positive,
            "positive_documents_with_no_detection": missed_docs,
            "pii_negative_documents": negative, "negative_documents_with_detection": false_positive_docs,
            "negative_document_false_positive_rate": false_positive_docs/negative if negative else None,
            "invalid_bio_prediction_documents": invalid,
            "prediction_decoding": "Orphan I starts a new entity; no boundary/type tolerance"}


def calibration_metrics(probabilities, labels, bins=15, uncertainty=None):
    from sklearn.metrics import roc_auc_score, average_precision_score
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if not len(y):
        return {"tokens": 0}
    if p.shape[0] != len(y) or np.any(~np.isfinite(p)) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probabilities")
    confidence, prediction = p.max(1), p.argmax(1)
    error = (prediction != y).astype(float)
    uncertainty = 1-confidence if uncertainty is None else np.asarray(uncertainty)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    brier = (np.square(p).sum(1) - 2*p[np.arange(len(y)), y] + 1).mean()
    table, ece = [], 0.
    index = np.minimum((confidence*bins).astype(int), bins-1)
    for b in range(bins):
        chosen = index == b
        if chosen.any():
            acc, conf = 1-error[chosen].mean(), confidence[chosen].mean()
            ece += chosen.mean()*abs(acc-conf)
            table.append({"bin": b, "count": int(chosen.sum()), "accuracy": float(acc), "confidence": float(conf)})
    order = np.argsort(uncertainty, kind="stable")
    risk = np.cumsum(error[order])/np.arange(1, len(y)+1)
    points = np.unique(np.linspace(0, len(y)-1, min(100, len(y))).astype(int))
    mixed = len(np.unique(error)) == 2
    return {"tokens": len(y), "nll": float(nll), "brier": float(brier), "ece": float(ece),
            "aurc": float(risk.mean()), "accuracy": float(1-error.mean()),
            "error_auroc": float(roc_auc_score(error, uncertainty)) if mixed else None,
            "error_average_precision": float(average_precision_score(error, uncertainty)) if mixed else None,
            "reliability_bins": table,
            "risk_coverage": [{"coverage": float((i+1)/len(y)), "risk": float(risk[i])} for i in points]}
