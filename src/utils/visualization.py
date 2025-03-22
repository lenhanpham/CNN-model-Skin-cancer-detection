import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from src.config import config

sns.set()

def plot_training_history(hist_dict):
    """Plot training history metrics and save to directory."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(config.METRICS_TO_PLOT):
        plt.subplot(2, 2, i + 1)
        epochs = range(1, len(hist_dict[config.METRICS_TO_PLOT[0]]) + 1)
        sns.scatterplot(x=epochs, y=hist_dict[metric])
        sns.lineplot(x=epochs, y=hist_dict['val_' + metric])
        plt.title(metric.capitalize())
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend(['Train', 'Validation'])
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(labels, predictions):
    """Plot confusion matrix and save to directory."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    cm = tf.math.confusion_matrix(labels, predictions > 0.5)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix @0.5')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(os.path.join(config.PLOTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_and_prc(train_labels, train_preds, val_labels, val_preds, test_labels, test_preds):
    """Plot ROC and PRC curves and save to directory."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(16, 6))
    
    # ROC
    plt.subplot(1, 2, 1)
    for name, labels, preds, style in [
        ('Train', train_labels, train_preds, '--'),
        ('Validation', val_labels, val_preds, ':'),
        ('Test', test_labels, test_preds, '-')
    ]:
        fp, tp, _ = metrics.roc_curve(labels, preds)
        plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, linestyle=style)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 60])
    plt.ylim([20, 100.5])
    plt.grid(True)
    plt.title('ROC')
    plt.legend(loc='lower right')

    # PRC
    plt.subplot(1, 2, 2)
    for name, labels, preds, style in [
        ('Train', train_labels, train_preds, '--'),
        ('Validation', val_labels, val_preds, ':'),
        ('Test', test_labels, test_preds, '-')
    ]:
        precision, recall, _ = metrics.precision_recall_curve(labels, preds)
        plt.plot(precision, recall, label=name, linewidth=2, linestyle=style)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.title('AUPRC')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'roc_prc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()