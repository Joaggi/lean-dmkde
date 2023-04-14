from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, classification_report, average_precision_score, precision_score, recall_score
import numpy as np
import check_dataset_for_algorithm
from precision_n_scores import precision_n_scores


def calculate_metrics(y_true, y_pred, y_scores, run_name):
   
   aucroc=None
   anomaly_label=None
   type = check_dataset_for_algorithm.check_dataset_for_algorithm(run_name)
   #print(type)
   y_true_auc = np.array(y_true)
   zero = y_true_auc!=1
   one = y_true_auc==1
   y_true_auc[zero] = 0
   y_true_auc[one] = 1
   
   y_scores = np.array(y_scores)
   y_scores = (y_scores - y_scores.min()) / y_scores.max()

   if (run_name.startswith("pyod") or run_name=="autoencoder" or type):
       y_scores = y_scores
   else:
       y_scores = 1 - y_scores

    
    
   anomaly_label = -1 if type else 1
   try:  
       aucroc = roc_auc_score(y_true_auc, y_scores)
       aucpr_not_anomaly = average_precision_score(1-y_true_auc, 1-y_scores)
       aucpr_anomaly = average_precision_score(y_true_auc, y_scores)
       accuracy = accuracy_score(y_true, y_pred)
       f1_weighted = f1_score(y_true, y_pred, average='weighted')
       f1_macro = f1_score(y_true, y_pred, average='macro')
       f1_binary = f1_score(y_true, y_pred, average='binary', pos_label=anomaly_label)
       pre = precision_score(y_true, y_pred)
       rec = recall_score(y_true, y_pred)
       pre_at =  precision_n_scores(y_true, y_scores)
   except: 
       aucroc = -999
       aucpr_not_anomaly  = -999
       aucpr_anomaly  = -999
       accuracy  = -999
       f1_weighted  = -999
       f1_macro  = -999
       f1_binary  = -999
       pre  = -999
       rec = -999
       pre_at  = -999
   
   metrics = {
   	"accuracy": accuracy,
   	"f1_score": f1_weighted,
   	"f1_macro": f1_macro,
   	"f1_anomalyclass": f1_binary,
   	"aucroc": aucroc,
   	"aucpr-nan": aucpr_not_anomaly,
   	"aucpr-an": aucpr_anomaly,
   	"pre": pre,
   	"rec": rec,
    "pre_at": pre_at
   }

   print(metrics)
   
   print(classification_report(y_true, y_pred))
   return metrics

