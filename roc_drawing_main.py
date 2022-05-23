from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics as metrics


def roc_curve(labels, preds, thresholds_count=10000):
    if len(labels) == 1:
        raise Exception(f'roc_curve: labels parameter is empty')
    if len(np.unique(labels)) == 1:
        raise Exception(f'roc_curve: labels parameter is composed of only one value')

    preds_on_positive = preds[labels == 1]
    preds_on_negative = preds[labels == 0]
    min_negative = min(preds_on_negative)
    max_positive = max(preds_on_positive)
    margin = 0  # (max_positive - min_negative)/100

    thresholds = np.linspace(min_negative - margin, max_positive + margin, thresholds_count)
    true_positive_rate = [np.mean(preds_on_positive > t) for t in thresholds]
    spec = [np.mean(preds_on_negative <= t) for t in thresholds]
    false_positive_rate = [1 - s for s in spec]
    auc = np.trapz(true_positive_rate, spec)

    thresholds = np.flip(thresholds, axis=0)
    false_positive_rate.reverse(), true_positive_rate.reverse()
    false_positive_rate, true_positive_rate = np.asarray(false_positive_rate), np.asarray(true_positive_rate)
    return false_positive_rate, true_positive_rate, auc, thresholds


labels = np.load("E:\\ResearchData\\heatmap_output\\1k_psi_1\\labels.npy")  # psi 1
predictions = np.load("E:\\ResearchData\\heatmap_output\\1k_psi_1\\predictions.npy")
fpr, tpr, auc, threshold = roc_curve(labels, predictions)
roc_auc = metrics.auc(fpr, tpr)

labels2 = np.load("E:\\ResearchData\\heatmap_output\\ex_1k_1k_psi_1\\labels.npy")  # psi 1
predictions2 = np.load("E:\\ResearchData\\heatmap_output\\ex_1k_1k_psi_1\\predictions.npy")
fpr2, tpr2, auc2, threshold2 = roc_curve(labels2, predictions2)
roc_auc2 = metrics.auc(fpr2, tpr2)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='no_ex_AUC = %0.2f' % roc_auc)
plt.plot(fpr2, tpr2, 'r', label='with_ex_AUC = %0.2f' % roc_auc2)
plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(True)
plt.savefig(os.path.join("E:\\ResearchData\\heatmap_output\\",
                         "roc_curve" + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png"))
plt.cla()

labels = np.load("E:\\ResearchData\\heatmap_output\\2k_mixed_psi_05\\labels.npy")  # psi 0.5
predictions = np.load("E:\\ResearchData\\heatmap_output\\2k_mixed_psi_05\\predictions.npy")
fpr, tpr, auc, threshold = roc_curve(labels, predictions)
index_fpr_threshold = 0

for fpr_item in fpr:
    if fpr_item <= 0.1:
        index_fpr_threshold += 1
fpr = fpr[:index_fpr_threshold]
tpr = tpr[:index_fpr_threshold]
roc_auc = metrics.auc(fpr, tpr)

labels2 = np.load("E:\\ResearchData\\heatmap_output\\ex_1k_2k_mixed_psi_05\\labels.npy")  # psi 0.5
predictions2 = np.load("E:\\ResearchData\\heatmap_output\\ex_1k_2k_mixed_psi_05\\predictions.npy")
fpr2, tpr2, auc2, threshold2 = roc_curve(labels2, predictions2)
index_fpr_threshold2 = 0

for fpr_item2 in fpr2:
    if fpr_item2 <= 0.1:
        index_fpr_threshold2 += 1
fpr2 = fpr2[:index_fpr_threshold2]
tpr2 = tpr2[:index_fpr_threshold2]
roc_auc2 = metrics.auc(fpr2, tpr2)

# max_xy = max(fpr[-index_fpr_threshold], tpr[-index_fpr_threshold])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='no_ex_AUC = %0.2f' % roc_auc)
plt.plot(fpr2, tpr2, 'r', label='with_ex_AUC = %0.2f' % roc_auc2)

plt.legend(loc='lower right')
# plt.plot([0, 0.1], [0.9, 1], 'r--')
plt.xlim([0, 0.1])
plt.ylim([0.9, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(True)
plt.savefig(os.path.join("E:\\ResearchData\\heatmap_output\\",
                         "roc_curve_with_threshold" + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png"))
plt.cla()
