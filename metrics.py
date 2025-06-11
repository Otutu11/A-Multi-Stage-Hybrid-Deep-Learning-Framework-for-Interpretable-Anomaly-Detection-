from sklearn.metrics import f1_score, roc_auc_score

def evaluate(y_true, y_pred):
    return {
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }