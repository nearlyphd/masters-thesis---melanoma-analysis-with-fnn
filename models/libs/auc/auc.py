from sklearn.metrics import roc_curve, auc


def calculate_auc(testing_set, groundtruth_labels, predicted_labels):
    fpr = dict()
    tpr = dict()
    auc_metric = dict()
    diagnosis_index_dict = {v: k for k, v in testing_set.class_indices.items()}
    for i in range(len(diagnosis_index_dict)):
        diagnosis = diagnosis_index_dict[i]
        fpr[diagnosis], tpr[diagnosis], _ = roc_curve(groundtruth_labels[:, i], predicted_labels[:, i])
        auc_metric[diagnosis] = auc(fpr[diagnosis], tpr[diagnosis])

    return fpr, tpr, auc_metric
