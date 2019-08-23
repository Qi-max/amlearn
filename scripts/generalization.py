import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, \
    confusion_matrix


def classifier_predict_unit(classifier, y, y_predict,
                            feature_list, log, dataset_tag, probability,
                            output_path="", roc=True, roc_file="",
                            roc_dpi=100, output_detail=True,
                            log_importances=False):

    roc_auc = -1
    if hasattr(classifier, "predict_proba") or hasattr(classifier, "decision_function"):
        log_classifier_result(y, y_predict, log, dataset_tag, probability)
        if roc:
            roc_path_file = os.path.join(output_path, roc_file)
            roc_auc = auc_and_roc_curve(y, y_predict, log=log,
                                        dataset_tag=dataset_tag,
                                        roc_path_file=roc_path_file,
                                        roc_dpi=roc_dpi,
                                        output_detail=output_detail)
    else:
        log_classifier_result(y, y_predict, log, dataset_tag, 0.5)

    if hasattr(classifier, "feature_importances_"):
        importances = sorted(zip(classifier.feature_importances_,
                                 feature_list), reverse=True)
        if log_importances:
            log("feature importance:")
            log(str(importances))
    return roc_auc


def auc_and_roc_curve(y, y_predict, log, dataset_tag, roc_path_file,
                      roc_dpi, roc=True, figsize=(10, 10), color='red',
                      output_detail=True):
    fpr, tpr, _ = roc_curve(y, y_predict)
    roc_auc = auc(fpr, tpr)
    log('\n{} Set AUC is: {} \n'.format(dataset_tag, roc_auc))

    if output_detail:
        if roc:
            np.savetxt(fname="{}.txt".format(roc_path_file),
                       X=list(zip(fpr, tpr)))
            plt.figure(figsize=figsize)
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, color=color,
                     label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.axis('tight')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig(roc_path_file, dpi=roc_dpi)
            plt.close()
    return roc_auc


def log_classifier_result(y, y_predict, log, dataset_tag, probability,
                          target_names=None, y_labels=None,
                          print_to_screen=False):
    if target_names is None:
        target_names = ['neg', 'pos']
    if y_labels is None:
        y_labels = [0, 1]

    if probability == "full":
        probability = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.7, 0.8, 0.9]

    probability = np.atleast_1d(probability)
    for proba in probability:
        y_proba = y_predict > proba
        cls_report = classification_report(y, y_proba, target_names=target_names)
        confs_matrix = str(confusion_matrix(y, y_proba, labels=y_labels))
        log("\n{} Set Result:\n prob > {}\n".format(dataset_tag, proba))
        log(cls_report)
        log(confs_matrix)

        if print_to_screen:
            print("\n{} Set Result:\n prob > {}\n".format(dataset_tag, proba))
            print(cls_report)
            print(confs_matrix)
