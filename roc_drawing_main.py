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


def output_single_roc(stats_path_list, title_list, lim_offset, save_dir, mode=False):

    if mode :
        file_suffix = "\\roc_curve_with_threshold_"
        plt.title('PSI 1 Receiver Operating Characteristic')
    else:
        file_suffix = "\\roc_curve_"
        plt.title('PSI 0.5 Receiver Operating Characteristic')
    for i in range(len(stats_path_list)):
        labels = np.load(stats_path_list[i] + "\\labels.npy")  # psi 1
        predictions = np.load(stats_path_list[i] + "\\predictions.npy")
        fpr, tpr, auc, threshold = roc_curve(labels, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b', label=f'{title_list[i]} = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, lim_offset])
        plt.ylim([(1 - lim_offset), 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid(True)
        print(save_dir+file_suffix + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png")
    plt.savefig(save_dir+file_suffix + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png")
    plt.cla()


def output_multiple_roc(stats_path_list, title_list, lim_offset, save_dir, mode=False):

    if mode :
        file_suffix = "\\roc_curve_with_threshold_"
        plt.title('PSI 1 Receiver Operating Characteristic')
    else:
        file_suffix = "\\roc_curve_"
        plt.title('PSI 0.5 Receiver Operating Characteristic')
    for i in range(len(stats_path_list)):
        labels = np.load(stats_path_list[i] + "\\labels.npy")  # psi 1
        predictions = np.load(stats_path_list[i] + "\\predictions.npy")
        fpr, tpr, auc, threshold = roc_curve(labels, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{title_list[i]} = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, lim_offset])
        plt.ylim([(1 - lim_offset), 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid(True)
        print(save_dir+file_suffix + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png")
    plt.savefig(save_dir+file_suffix + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png")
    plt.cla()

# psi_1_list = ["E:\\ResearchData\\heatmap_output\\1k_psi_1\\",
#               "E:\\ResearchData\\heatmap_output\\ex_500_1k_psi_1\\",
#               "E:\\workplace\\GAIN-pytorch-main\\logs_deepfake\\ex_500_exweight_1.5\\test_ex_500_exweight_1.5_s2f_psi_1_000000\\",
#               "E:\\ResearchData\\heatmap_output\\ex_1k_exweight_1.5_PSI_1\\"]
psi_1_list = ["E:\\ResearchData\\heatmap_output\\1k_psi_1\\",
              "E:\\ResearchData\\heatmap_output\\test_ex_500_exweight_0_orig_sampling_PSI_1\\",
              ]


# psi_05_list = ["E:\\ResearchData\\heatmap_output\\2k_mixed_psi_05\\",
#                "E:\\ResearchData\\heatmap_output\\ex_500_2k_mixed_psi_05\\",
#                "E:\\workplace\\GAIN-pytorch-main\\logs_deepfake\\ex_500_exweight_1.5\\test_ex_500_exweight_1.5_ffhq_s2_PSI_0_5\\",
#                "E:\\ResearchData\\heatmap_output\\\ex_1k_exweight_1.5_PSI_0.5\\"]
psi_05_list = ["E:\\ResearchData\\heatmap_output\\2k_mixed_psi_05\\",
               "E:\\ResearchData\\heatmap_output\\test_ex_500_exweight_0_orig_sampling_PSI_0.5\\",
               ]


def main():
    # title_list = ["no_ex_AUC",
    #               "with_500_ex_AUC",
    #               "with_500_ex_1.5_exweight_AUC",
    #               "with_1k_ex_1.5_exweight_AUC"]
    title_list = ["no_ex_AUC",
                "with_500_ex_0_exweight_AUC",]
    save_dir = "E:/ResearchData/heatmap_output/"

    # psi_05_list = ["E:\\ResearchData\\heatmap_output\\\ex_1k_exweight_1.5_PSI_0.5\\"]

    # output_single_roc(psi_05_list, title_list, lim_offset=0.1, save_dir=save_dir, mode=True)
    output_multiple_roc(psi_05_list, title_list, lim_offset=0.1, save_dir=save_dir, mode=True)
    output_multiple_roc(psi_1_list, title_list, lim_offset=1, save_dir=save_dir)

if __name__ == '__main__':
    main()
