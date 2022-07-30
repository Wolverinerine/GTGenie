import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from scipy import interp

def statistic_total_AUC(args, KFOLD_test_labels, KFOLD_test_scores):
    """
        output all scores which can be used in case study
    """

    auc_result = []
    aupr_result = []

    fprs = []
    tprs = []
    for i in range(len(KFOLD_test_labels)):
        fpr, tpr, thresholds = roc_curve(KFOLD_test_labels[i], KFOLD_test_scores[i])
        test_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(KFOLD_test_labels[i], KFOLD_test_scores[i])
        test_aupr = auc(recall,precision)

        print('times:', int(i/5), 'Fold:', (i % 5), 'Test AUC: %.4f' % test_auc, 'Test AUPR: %.4f' % test_aupr)

        auc_result.append(test_auc)
        aupr_result.append(test_aupr)
        fprs.append(fpr)
        tprs.append(tpr)

    print('-AUC mean: %.4f±%.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          '-AUPR mean: %.4f±%.4f \n' % (np.mean(aupr_result), np.std(aupr_result)))

    mean_fpr = np.linspace(0, 1, 10000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    auc_std = np.std(auc_result)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))

    std_tpr = np.std(tpr, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(args.dataset)+' ROC Curves')
    plt.legend(loc='lower right')
    fig_dir = './ROC.png'
    plt.show()
    # plt.savefig(fig_dir)