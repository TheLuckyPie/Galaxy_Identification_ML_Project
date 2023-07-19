### eval.py

# Functions for evaluation/plotting

import os
import sys
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from utils import get_preds_and_labels

        
def get_eval(history, model_res_filepath, training_param, model_param, model, test_data, dataset_param, training_time):
        """
        Function to plot Accuracy, Loss, and ROC-AUC Curves and export them alongside model summary and other metrics
        """
        
        epochs_range = range(training_param['epochs'])

        acc = history.history['accuracy']
        loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy',linestyle = 'dashed')
        plt.legend(loc='lower right')
        plt.xlabel("Epoch Number")
        plt.ylabel("Accuracy")
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss',linestyle = 'dashed')
        plt.legend(loc='upper right')
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.title('Training and Validation Loss')

        labels_values, predictions = get_preds_and_labels(model, dataset_param, test_data)
        false_pos_rate, true_pos_rate, _ = roc_curve(labels_values, predictions)
        roc_auc = auc(false_pos_rate, true_pos_rate)

        plt.subplot(1, 3, 3)
        plt.plot(false_pos_rate, true_pos_rate, label=f'AUC: {roc_auc: .6f}')
        plt.xlabel("False-Positive Rate")
        plt.ylabel("True-Positive Rate")

        x = np.linspace(0, 1, 2)
        plt.plot(x,x,linestyle = 'dashed', color='black' )
        plt.legend(loc='lower right')
        plt.title('ROC-AUC Curve')
        plt.savefig(os.path.join(model_res_filepath, model_param['name']+'_LossAccuracyROC.pdf'))
        plt.close()

        accuracy, precision, recall, f1, rocauc_score, cm = get_binary_metrics(labels_values, predictions)
        
        metrics_df ={
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'Training Time (s)'],
                'Score': [accuracy, precision, recall, f1, rocauc_score, training_time[0]*60 + training_time[1]]}
        pd.DataFrame(metrics_df).to_csv(os.path.join(model_res_filepath,model_param['name']+'_metrics_df.csv'))

        plt.figure(figsize=(12, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=True,
            xticklabels=['Class1.1 (Smooth Galaxy)', 'Class1.2 (Disk Galaxy)'],
            yticklabels=['Class1.1', 'Class1.2'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix for ' + str(model_param['name']))

        plt.savefig(os.path.join(model_res_filepath, model_param['name']+'_ConfusionMatrix.pdf'))
        plt.close()

        #Tacked on function to export model summary as a txt file
        with open(os.path.join(model_res_filepath,'model_summary.txt'), 'w') as f:
                sys.stdout = f
                model.summary()
                sys.stdout = sys.__stdout__
        
def get_binary_metrics(labels_values, predictions):
        """
        Function to get Evaluation Metrics for Binary Classification Problem
        """
        pred_classes = np.where(predictions >= 0.5, 1, 0)

        accuracy = round(accuracy_score(labels_values, pred_classes),4)
        precision = round(precision_score(labels_values, pred_classes),4)
        recall = round(recall_score(labels_values, pred_classes),4)
        f1 = round(f1_score(labels_values, pred_classes),4)
        rocauc_score = round(roc_auc_score(labels_values, pred_classes),4)
        cm = confusion_matrix(labels_values, pred_classes)

        return accuracy, precision, recall, f1, rocauc_score, cm