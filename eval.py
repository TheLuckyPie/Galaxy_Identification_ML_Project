### eval.py

# Functions for evaluation/plotting

from sklearn.metrics import roc_curve, auc
        
def plot_history(history, model_res_filepath, training_param, model_param):
"""
    Function to plot Accuracy and Loss Graphs and export them.
"""
        
        epochs_range = range(training_param['epochs'])

        acc = history.history['accuracy']
        loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy',linestyle = 'dashed')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss',linestyle = 'dashed')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.savefig(os.path.join(model_res_filepath, 'LossAccuracy.png'))
        plt.show()
        plt.close()

def plot_roc_auc(model, test_data, model_res_filepath):
"""
    Function to plot ROC curve and export
"""
        labels_values, predictions = def get_preds_and_labels(model, dataset_param, test_data):
        false_pos_rate, true_pos_rate, _ = roc_curve(labels_values, predictions)
        roc_auc = auc(false_pos_rate, true_pos_rate)

        plt.plot(false_pos_rate, true_pos_rate, label=f'AUC: {roc_auc: .6f}')
        plt.xlabel("False-Positive Rate")
        plt.ylabel("True-Positive Rate")

        x = np.linspace(0, 1, 2)
        plt.plot(x,x,linestyle = 'dashed', color='black' )
        plt.legend(loc='lower right')
        plt.title('ROC Curve')

        plt.savefig(os.path.join(model_res_filepath, 'roccurve.png'))
        plt.show()
        plt.close()
        