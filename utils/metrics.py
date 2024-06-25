from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(labels, outputs):
    accuracy = accuracy_score(labels, outputs)
    precision = precision_score(labels, outputs)
    recall = recall_score(labels, outputs)
    f1 = f1_score(labels, outputs)
    auc_roc = roc_auc_score(labels, outputs)
    return accuracy, precision, recall, f1, auc_roc
