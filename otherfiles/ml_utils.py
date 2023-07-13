import numpy as np
import pandas as pd
import shutil
import os
from sklearn.model_selection import train_test_split

#Function to Check if file exists or not
#def Exist_Checker(filename):
#    if os.path.exists(filename):
#        state = True
#    else:
#        state = True
#    return state

#Function to Create Ex1 Filtered Labels and Images (Feedback to ensure that it works)
def Ex1_CreateLabels(label_source,label_destination,image_source_folder,image_destination_folder,feedback):
    
    #Read Labels from CSV file
    Image_Labels = pd.read_csv(label_source)

    #Filter Labels to only include GalaxyID, class1.1, 1.2 & 1.3
    e1_filtered_raw_labels = Image_Labels[['GalaxyID', 'Class1.1','Class1.2','Class1.3']]
    #Filter out Galaxies with label confidence greater than 80%
    e1_filtered_labels = e1_filtered_raw_labels.loc[(e1_filtered_raw_labels['Class1.1']>0.8) | (e1_filtered_raw_labels['Class1.2']>0.8) | (e1_filtered_raw_labels['Class1.3']>0.8)]
    
    #Set labels percentages to 0 if under 80% or 1 if over
    label_list = ['Class1.1','Class1.2','Class1.3']
    for label in label_list:
        e1_filtered_labels.loc[e1_filtered_labels[label] > 0.8, label] = 1
        e1_filtered_labels.loc[e1_filtered_labels[label] <= 0.8, label] = 0
    #Save Filtered Labels as a CSV file
    e1_filtered_labels.to_csv(label_destination,index=False)

    #Move Images with same Labels as filtered into a filtered image folder
    os.makedirs(image_destination_folder, exist_ok=True)
    for index, row in e1_filtered_labels.iterrows():
        filename = str(str(int(row['GalaxyID'])) + '.jpg')
        img_path = os.path.join(image_source_folder, filename)
        dest_path = os.path.join(image_destination_folder, filename)
        if os.path.exists(img_path) == True and os.path.exists(dest_path) == False:
            shutil.copy(img_path, dest_path)
            if feedback == True:
                print("Successfully Copied", filename)
        elif feedback == True:
            print("Error:", filename, "cannot be transferred; either it doesn't exist in the source folder, or already exists in the destination folder.")
    return e1_filtered_labels

#Function to Generate Train and Validation datasets (label csv and images)
def Ex1_Test_and_Validation(labels, image_source,train_destination,validation_destination,validation_size,feedback):
    #Split Filtered Data into Train and Validation Sets using Sklearn
    train_labels, val_labels = train_test_split(labels, test_size=validation_size)
    #Make Folders for Train and Validation Images
    os.makedirs(train_destination, exist_ok=True)
    os.makedirs(validation_destination, exist_ok=True)

    #Copy only Data Marked For Training
    for index, row in train_labels.iterrows():
        filename = str(str(int(row['GalaxyID'])) + '.jpg')
        src_path = os.path.join(image_source, filename)
        dest_path = os.path.join(train_destination, filename)
        if os.path.exists(src_path) == True and os.path.exists(dest_path) == False:
            shutil.copy(src_path, dest_path)
            if feedback == True:
                print("Successfully Copied", filename)

    for index, row in val_labels.iterrows():
        filename = str(str(int(row['GalaxyID'])) + '.jpg')
        src_path = os.path.join(image_source, filename)
        dest_path = os.path.join(validation_destination, filename)
        if os.path.exists(src_path) == True and os.path.exists(dest_path) == False:
            shutil.copy(src_path, dest_path)
            if feedback == True:
                print("Successfully Copied", filename)
    return train_labels, val_labels