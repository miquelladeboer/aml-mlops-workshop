import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from itertools import cycle


def plot_auc(fpr, tpr, roc_auc):

    n_classes = 4
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Plot all ROC curves
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test')
    plt.legend(loc="lower right")
    return plt


def plot_loss_per_epoch(epoch_losses, num_epochs):

    # plot loss function and log to Azure ML
    plt.clf()
    plt.plot(np.array(epoch_losses), 'r', label="Loss")
    plt.xticks(np.arange(1, (num_epochs+1), step=1))
    plt.xlabel("Epochs")
    plt.ylabel("Loss Percentage")
    plt.legend(loc='upper left')
    return plt


def plot_accuracy_per_epoch(epoch_accuracy, num_epochs):

    # plot Accuracy and log to Azure ML
    plt.clf()
    plt.plot(np.array(epoch_accuracy), 'b', label="Accuracy")
    plt.xticks(np.arange(1, (num_epochs+1), step=1))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Percentage")
    plt.legend(loc='upper left')
    return plt
