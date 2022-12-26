from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, classification_report
import numpy as np
import check_dataset_for_algorithm

def calculate_metrics(y_true, y_pred, y_scores, run_name):

	aucroc=None
	anomaly_label=None
	type = check_dataset_for_algorithm.check_dataset_for_algorithm(run_name)
	#print(type)
	
	y_true_auc = np.array(y_true)
	zero = y_true_auc!=1
	one = y_true_auc==1
	y_true_auc[zero] = 1
	y_true_auc[one] = 0
	
	if (run_name.startswith("pyod") or run_name=="autoencoder" or type):
	    aucroc = roc_auc_score(y_true, y_scores)
	else:
	    aucroc = roc_auc_score(y_true_auc, y_scores)

	anomaly_label = -1 if type else 1

	metrics = {
		"accuracy": accuracy_score(y_true, y_pred),
		"f1_score": f1_score(y_true, y_pred, average='weighted'),
		"f1_macro": f1_score(y_true, y_pred, average='macro'),
		"f1_anomalyclass": f1_score(y_true, y_pred, average='binary', pos_label=anomaly_label),
		"aucroc": aucroc,
	}

	print(classification_report(y_true, y_pred))
	return metrics

