from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import check_dataset_for_algorithm

def calculate_metrics(y_true, y_pred, y_scores, run_name):
    aucroc = None
    if check_dataset_for_algorithm.check_dataset_for_algorithm(run_name):
        y_true_auc = np.array(y_true)
        zero = y_true_auc==0
        one = y_true_auc==1
        y_true_auc[zero] = 1
        y_true_auc[one] = 0


        aucroc = roc_auc_score(y_true_auc, y_scores)
    else:
        aucroc = roc_auc_score(y_true, y_scores)


    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "auroc": aucroc
    }
    
    return metrics
