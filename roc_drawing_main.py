from datetime import datetime
# import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score

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
        plt.title('Test on s2p1')
        lim_offset = lim_offset
    else:
        file_suffix = "\\roc_curve_"
        plt.title('Test on s2p1')
        lim_offset = 1
    for i in range(len(stats_path_list)):
        labels = np.load(stats_path_list[i] + "\\labels.npy")  # psi 1
        predictions = np.load(stats_path_list[i] + "\\predictions.npy")
        fpr, tpr, auc, threshold = roc_curve(labels, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{title_list[i]} = %0.4f' % roc_auc)
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


def output_losses(stats_path_list, title_list, x_list, img_idx_list, img_idx, lim_offset, save_dir):
    file_suffix = "\\exsup_losses_"
    plt.title('Losses')


    # x_np = np.load(x_list, allow_pickle=True)
    x_np = np.load(x_list, allow_pickle=True).astype(str)
    print(x_np.shape)
    print(x_np.shape)
    x_len = range(len(x_np))
    # count_pos = (x_np == 50).sum()
    # print(x_np)
    # print(np.count_nonzero(x_np == 49))
    # img_idx_np = np.load(img_idx_list[0], allow_pickle=True)[-25:]
    # exsup_count_np = list()
    # for i in range(25):
    #     count = 0
    #     for str_item in img_idx_np[i]:
    #         if len(str_item) == 10 and str_item[0:3] == '000' and str_item[3] in ('4', '3', '2', '1', '0'):
    #             count += 1
    #     exsup_count_np.append(count)
    # print(img_idx_np)
    # # exsup_count_np = np.count_nonzero(img_idx_np < 500, axis=1)
    # plt.xticks(x_len, x_np)
    # plt.plot(x_len, exsup_count_np, label='num of exsup')
    # plt.legend(loc='lower right')
    # plt.ylabel('Exsup Count')
    # plt.xlabel('Epoch')
    # plt.grid(True)
    # plt.savefig(save_dir + file_suffix + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png")
    # plt.cla()
    # return
    np.count_nonzero(x_np == 49)

    for i in range(len(stats_path_list)):
        if i == 0:
            continue
        losses = np.load(stats_path_list[i], allow_pickle=True)
        # output losses per epoch (sum over iteration/average)
        x_epoch_list = list()
        y_sum_list = list()
        y_avg_list = list()
        for epoch in range(15, 50):
            idx_epoch = x_np == str(epoch)
            x_epoch_list.append(epoch)
            # print(x_np)
            # print(x_np[idx_epoch])
            y_sum_list.append(np.sum(losses[idx_epoch], dtype=np.float64))
            y_avg_list.append(np.sum(losses[idx_epoch] / len(idx_epoch), dtype=np.float64))
        print(losses.shape)
        # plt.xticks(x_len, x_np)
        plt.plot(x_epoch_list, y_avg_list, label=f'{title_list[i]}')
        plt.legend(loc='upper right')
    # output losses per iteration
    # for i in range(len(stats_path_list)):
    #     if i == 0:
    #         continue
    #     losses = np.load(stats_path_list[i], allow_pickle=True)
    #     print(losses.shape)
    #     plt.xticks(x_len, x_np)
    #     plt.plot(x_len, losses, label=f'{title_list[i]}')
    #     plt.legend(loc='upper right')

    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.title("Losses per epoch(iteration avg)")
    plt.savefig(save_dir+file_suffix + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png")
    plt.cla()

    # proj = np.array([0, 0, 1, 1, 1, 1, 2]).astype(str)
    # proj_x = range(len(proj))
    # prVal = np.array([0.9, 0.8, 0.8, 0.9, 0.3, 0.2, 0.6])
    #
    # # creating the bar plot
    # plt.xticks(proj_x, proj)
    # plt.plot(proj_x, prVal)
    # plt.xlabel("gifts")
    # plt.ylabel("Value")
    # plt.title("Gift recieved")
    # plt.savefig(save_dir + file_suffix + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".png")
    # plt.cla()
# psi_1_list = ["E:\\ResearchData\\heatmap_output\\1k_psi_1\\",
#               "E:\\ResearchData\\heatmap_output\\ex_500_1k_psi_1\\",
#               "E:\\workplace\\GAIN-pytorch-main\\logs_deepfake\\ex_500_exweight_1.5\\test_ex_500_exweight_1.5_s2f_psi_1_000000\\",
#               "E:\\ResearchData\\heatmap_output\\ex_1k_exweight_1.5_PSI_1\\"]
psi_1_list = [
              "E:\\ResearchData\\heatmap_output\\test_pretrain_no_ex_sampling_epoch_50_PSI_1\\",
                "E:\\ResearchData\\heatmap_output\\test_pretrain_no_ex_no_am_orig_sampling_epoch150_PSI_1\\",
              ]


# psi_05_list = ["E:\\ResearchData\\heatmap_output\\2k_mixed_psi_05\\",
#                "E:\\ResearchData\\heatmap_output\\ex_500_2k_mixed_psi_05\\",
#                "E:\\workplace\\GAIN-pytorch-main\\logs_deepfake\\ex_500_exweight_1.5\\test_ex_500_exweight_1.5_ffhq_s2_PSI_0_5\\",
#                "E:\\ResearchData\\heatmap_output\\\ex_1k_exweight_1.5_PSI_0.5\\"]
psi_05_list = [
               "E:\\ResearchData\\heatmap_output\\cvpr_s2p1_e120\\test_cvpr_s2p1_e120_s2_PSI_1\\",
                "E:\\ResearchData\\heatmap_output\\cvpr_s2p1_e120\\test_cvpr_s2p1_e120_s2p1_debg\\",
               ]


def main():
    if True:
        # title_list = ["no_ex_AUC",
        #               "with_500_ex_AUC",
        #               "with_500_ex_1.5_exweight_AUC",
        #               "with_1k_ex_1.5_exweight_AUC"]
        # title_list = ["no_ex_AUC",
        #               "pretrain_no_ex_AUC",
        #               "with_500_ex_0.2_exweight_origSampling_AUC",
        #               "pretrain_with_500_ex_0.2_exweight_origSampling_newEx_AUC",
        #               ]
        title_list = ["EC-H on orig",
                      "EC-H on debg",
                     ]
        save_dir = "E:/ResearchData/heatmap_output/"

        # psi_05_list = ["E:\\ResearchData\\heatmap_output\\\ex_1k_exweight_1.5_PSI_0.5\\"]

        output_multiple_roc(psi_05_list, title_list, lim_offset=0.1, save_dir=save_dir, mode=True)
        output_multiple_roc(psi_05_list, title_list, lim_offset=0.1, save_dir=save_dir, mode=False)
        # output_multiple_roc(psi_1_list, title_list, lim_offset=1, save_dir=save_dir)
    else:
        losses_list = ["E:\\ResearchData\\heatmap_output\\20220617_heatmap_output_pretrain_ex_500_exweight_0.2_orig_sampling_newex_v4\\y_cl_loss_exsup_img.npy",
                       "E:\\ResearchData\\heatmap_output\\20220617_heatmap_output_pretrain_ex_500_exweight_0.2_orig_sampling_newex_v4\\y_am_loss_exsup_img.npy",
                       "E:\\ResearchData\\heatmap_output\\20220617_heatmap_output_pretrain_ex_500_exweight_0.2_orig_sampling_newex_v4\\y_ex_loss_exsup_img.npy",
                       ]
        title_list = ["cl loss",
                      "am loss",
                      "ex loss",
                      ]
        x_list = "E:\\ResearchData\\heatmap_output\\20220617_heatmap_output_pretrain_ex_500_exweight_0.2_orig_sampling_newex_v4\\x_epoch_exsup_img.npy"
        img_idx_list = ["E:\\ResearchData\\heatmap_output\\20220617_heatmap_output_pretrain_ex_500_exweight_0.2_orig_sampling_newex_v4\\img_idx.npy"]
        img_idx = 1
        lim_offset = 0
        save_dir = "E:/ResearchData/heatmap_output/20220617_heatmap_output_pretrain_ex_500_exweight_0.2_orig_sampling_newex_v4/losses/"
        output_losses(losses_list, title_list, x_list, img_idx_list, img_idx, lim_offset, save_dir)





if __name__ == '__main__':

    # y_true = np.load("E:\\ResearchData\\heatmap_output\\20220716_heatmap_output_cvpr_e100\\psi_0.5\\labels.npy")
    # y_pred = np.load("E:\\ResearchData\\heatmap_output\\20220716_heatmap_output_cvpr_e100\\psi_0.5\\predictions.npy")
    # ap = average_precision_score(y_true, y_pred)
    # print(ap)
    # y_true = np.load("E:\\ResearchData\\heatmap_output\\20220716_heatmap_output_cvpr_e100\\psi_1\\labels.npy")
    # y_pred = np.load("E:\\ResearchData\\heatmap_output\\20220716_heatmap_output_cvpr_e100\\psi_1\\predictions.npy")
    # ap = average_precision_score(y_true, y_pred)
    # print(ap)
    # exit(0)
    main()
