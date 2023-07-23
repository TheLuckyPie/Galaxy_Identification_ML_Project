### eval.py

# Functions for evaluation/plotting

import os
import sys
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from utils import get_preds_and_labels


def get_binary_metrics(labels_values, predictions):
        """
        Function to get Evaluation Metrics for Binary Classification Problem
        """
        pred_classes = np.where(predictions >= 0.5, 1, 0)
        

        accuracy = accuracy_score(labels_values, pred_classes)
        precision = precision_score(labels_values, pred_classes)
        recall = recall_score(labels_values, pred_classes)
        f1 = f1_score(labels_values, pred_classes)
        rocauc_score = roc_auc_score(labels_values, pred_classes)
        cm = confusion_matrix(labels_values, pred_classes)

        return accuracy, precision, recall, f1, rocauc_score, cm

def get_regression_metrics(labels_values, predictions):
        """
        Function to get Evaluation Metrics for Regression Problem
        """

        # Convert probabilities to class labels
        true_labels = np.argmax(np.array(labels_values), axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        #Calculate the metrics
        mse = mean_squared_error(np.array(labels_values), predictions)
        mae = mean_absolute_error(np.array(labels_values), predictions)
        r2 = r2_score(np.array(labels_values), predictions)

        #For Classification Performance
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average = 'weighted')
        cm = confusion_matrix(true_labels, pred_labels)

        return mse,mae, r2, cm, accuracy, f1


def get_eval(history, model_res_filepath, training_param, model_param, model, test_data, dataset_param, training_time,label_param):
        """
        Function to plot Accuracy, Loss, ROC-Curves, MSE, and Confusion Matrix and export them alongside model summary and eval metrics
        """

        labels_values, predictions = get_preds_and_labels(model, dataset_param, test_data)

        acc = history.history['accuracy']
        loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        mse = history.history['mean_squared_error']
        val_mse = history.history['val_mean_squared_error']

        epochs_range = range(len(acc))

        if dataset_param['regressionmodel'] == False:
                plotwidth=18
                subplots=3

                plt.figure(figsize=(plotwidth, 6))
                plt.subplot(1, subplots, 1)
                plt.plot(epochs_range, acc, label='Training Accuracy')
                plt.plot(epochs_range, val_acc, label='Validation Accuracy',linestyle = 'dashed')
                plt.legend(loc='lower right')
                plt.xlabel("Epoch Number")
                plt.ylabel("Accuracy")
                plt.title('Training and Validation Accuracy')
        
                plt.subplot(1, subplots, 2)
                plt.plot(epochs_range, loss, label='Training Loss')
                plt.plot(epochs_range, val_loss, label='Validation Loss',linestyle = 'dashed')
                plt.legend(loc='upper right')
                plt.xlabel("Epoch Number")
                plt.ylabel("Loss")
                plt.title('Training and Validation Loss')

                
                if subplots == 3:
                        false_pos_rate, true_pos_rate, _ = roc_curve(labels_values, predictions)
                        roc_auc = auc(false_pos_rate, true_pos_rate)

                        plt.subplot(1, subplots, 3)
                        plt.plot(false_pos_rate, true_pos_rate, label=f'AUC: {roc_auc: .6f}')
                        plt.xlabel("False-Positive Rate")
                        plt.ylabel("True-Positive Rate")

                        x = np.linspace(0, 1, 2)
                        plt.plot(x,x,linestyle = 'dashed', color='black' )
                        plt.legend(loc='lower right')
                        plt.title('ROC-AUC Curve')
                        plt.savefig(os.path.join(model_res_filepath, model_param['name']+'_LossAccuracyROC.pdf'))
                        #plt.savefig(model_param['name']+'_LossAccuracyROC.pdf')
                        plt.close()

                else:
                        plt.savefig(os.path.join(model_res_filepath, model_param['name']+'_LossAccuracy.pdf'))
                        plt.close()
                #Metrics get and export
                accuracy, precision, recall, f1, rocauc_score, cm = get_binary_metrics(labels_values, predictions)
        
                metrics_df ={
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'Training Time (s)'],
                        'Score': [accuracy, precision, recall, f1, rocauc_score, training_time[0]*60 + training_time[1]]}
                pd.DataFrame(metrics_df).to_csv(os.path.join(model_res_filepath,model_param['name']+'_metrics_df.csv'))    
        else:
                plt.figure(figsize=(6, 6))
                plt.plot(epochs_range, mse, label='Training MSE')
                plt.plot(epochs_range, val_mse, label='Validation MSE',linestyle = 'dashed')
                plt.legend(loc='upper right')
                plt.xlabel("Epoch Number")
                plt.ylabel("MSE")
                plt.title('Training and Validation MSE')

                plt.savefig(os.path.join(model_res_filepath, model_param['name']+'_MSE.pdf'))
                plt.close()
        

                
                #Metrics get and export
                
                mse,mae, r2, cm, accuracy, f1 = get_regression_metrics(labels_values, predictions)

                metrics_df ={
                        'Metric': ['MSE', 'MAE', 'R2', 'Training Time (s)', 'Accuracy', 'F1_Score'],
                        'Score': [mse,mae,r2, training_time[0]*60 + training_time[1], accuracy, f1]}
                pd.DataFrame(metrics_df).to_csv(os.path.join(model_res_filepath,model_param['name']+'_metrics_df.csv'))
        
        #Plot Confusion Matrix
        plt.figure(figsize=(12, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=True,
        xticklabels=label_param['classes'],
        yticklabels=label_param['classes'])
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
