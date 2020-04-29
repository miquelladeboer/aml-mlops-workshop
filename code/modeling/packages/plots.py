import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
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

    # Plot all ROC curvesS
    plt2.figure(figsize=(6, 4.5))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt2.plot(fpr[i], tpr[i], color=color, lw=2,
                  label='ROC curve of class {0} (area = {1:0.2f})'
                  ''.format(i, roc_auc[i]))

    plt2.plot([0, 1], [0, 1], 'k--', lw=2)
    plt2.xlim([0.0, 1.0])
    plt2.ylim([0.0, 1.05])
    plt2.xlabel('False Positive Rate')
    plt2.ylabel('True Positive Rate')
    plt2.title('Test')
    plt2.legend(loc="lower right")
    return plt2


def plot_loss_per_epoch(epoch_losses, num_epochs):

    # plot loss function and log to Azure ML
    plt1.clf()
    plt1.plot(np.array(epoch_losses), 'r', label="Loss")
    plt1.xticks(np.arange(1, (num_epochs+1), step=1))
    plt1.xlabel("Epochs")
    plt1.ylabel("Loss Percentage")
    plt1.legend(loc='upper left')
    return plt1


def plot_accuracy_per_epoch(epoch_accuracy, num_epochs):

    # plot Accuracy and log to Azure ML
    plt2.clf()
    plt2.plot(np.array(epoch_accuracy), 'b', label="Accuracy")
    plt2.xticks(np.arange(1, (num_epochs+1), step=1))
    plt2.xlabel("Epochs")
    plt2.ylabel("Accuracy Percentage")
    plt2.legend(loc='upper left')
    return plt2


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    plt.figure(figsize=(6, 4.5))
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm1.max()/1.5

    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm1[i, j]),
                 horizontalalignment="center",
                 color="white" if cm1[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    return plt


def plot_confusion_matrix_abs(cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None):

    import matplotlib.pyplot as ax2
    import numpy as np
    import itertools

    ax2.figure(figsize=(6, 4.5))

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Reds')

    ax2.imshow(cm, interpolation='nearest', cmap=cmap)
    ax2.title(title)
    ax2.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax2.xticks(tick_marks, target_names, rotation=45)
        ax2.yticks(tick_marks, target_names)

    cm = cm
    thresh = cm.max()/2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax2.text(j, i, "{:0.4f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax2.tight_layout()
    ax2.ylabel('True label')
    ax2.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    return ax2
