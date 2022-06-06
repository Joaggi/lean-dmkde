from sklearn.metrics import f1_score, roc_curve

def find_best_threshold(labels, scores):
    
    _, _, thresholds = roc_curve(labels, scores)
    m, T = 0, 0
    for t in thresholds:
        preds = (scores < t).astype(int)
        metric = f1_score(labels, preds, average='weighted')
        if (metric > m):    
            m = metric
            T = t

    return T