from sklearn.metrics import classification_report, f1_score, roc_auc_score

def evaluate_classification(y_true, y_pred, y_prob):
    metrics = {
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    report = classification_report(y_true, y_pred)
    return metrics, report

