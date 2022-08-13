from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import check_dataset_for_algorithm

def calculate_metrics(y_true, y_pred, y_scores, run_name):

	aucroc=None
	anomaly_label=None
	type = check_dataset_for_algorithm.check_dataset_for_algorithm(run_name)
	
	y_true_auc = np.array(y_true)
	zero = y_true_auc==0
	one = y_true_auc==1
	y_true_auc[zero] = 1
	y_true_auc[one] = 0
	aucroc = roc_auc_score(y_true_auc, y_scores)
	
	precision, recall, _ = precision_recall_curve(y_true_auc, y_scores)
	aucpr = auc(recall, precision)
	
	if (type):
		anomaly_label = -1
	else:
		anomaly_label = 1

	metrics = {
		"accuracy": accuracy_score(y_true, y_pred),
		"f1_score": f1_score(y_true, y_pred, average='weighted'),
		"f1_macro": f1_score(y_true, y_pred, average='macro'),
		"f1_anomaly_class": f1_score(y_true, y_pred, average='binary', pos_label=anomaly_label),
		"aucroc": aucroc,
		"auc-pr": aucpr
	}

	return metrics

