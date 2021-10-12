#import IPython.display as ipd
import librosa
#import librosa.display
#import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
#import pandas as pd
import os

###########################################################
#Deploy the trained model from the saved file
def load_trained_model(class_list_file="list.csv", trained_model="dog-bark.pickle"):
    #class_list = pd.read_csv("list.csv", header=None)[0]
    #y = np.array(class_list.tolist())

    class_list = []
    with open(class_list_file) as file:
        for line in file:
            class_list.append(line.rstrip())

    y = np.array(class_list)

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    TRAINED_MODEL_FILE = "dog-bark.pickle"
    with open(TRAINED_MODEL_FILE, "rb") as f:
    	model = pickle.load(f)
    return model, le


###########################################################
# Predictions

def extract_feature(file_name):   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    return np.array([mfccsscaled])

def files_in_folders(folder):
    folder = os.path.normcase(folder)
    files = os.listdir(folder)
    file_paths = []
    for file in files:

        # Add wav file into the file_paths list
        if file[-4:] == ".wav":
            file_paths.append(os.path.join(folder,file))

    return file_paths

import time

def is_dog_barking(file_names):

    # load class list "list.csv" and trained model "dog-bark.pickle"
    model, le = load_trained_model()

    start = time.time()
    file_count = 0
    for file_name in file_names:
        file_count += 1
        print("#################################")
        print(f"{file_count}. Processing {file_name}...")

        prediction_feature = extract_feature(file_name)

        predicted_vector = model.predict_classes(prediction_feature)
        predicted_class = le.inverse_transform(predicted_vector)

    #    print("The predicted class is:", predicted_class[0], '\n')
        predicted_proba_vector = model.predict_proba(prediction_feature)
        predicted_proba = predicted_proba_vector[0]
    #    for i in range(len(predicted_proba)):
    #        category = le.inverse_transform(np.array([i]))
    #        print(category[0], "\t\t : ", format(predicted_proba[i], '.4f'))*/

        print("Predicted class is: ", predicted_class[0])
        print("Dog bark probability is: ", predicted_proba[3])
        if (predicted_class[0] == "dog_bark") or (predicted_proba[3] > 0.35):
            #print("#################################")
            #print(f"{file_count}. Processing {file_name}...")
            print("The dog is barking")
            #return True
        else:
            print("The dog is not barking")
            #return False
    end = time.time()
    processing_time = end - start
    print(f"Processing time for {file_count} files is: {processing_time}")
############################################################

# Validation
# Class: Dog Bark

if __name__ == '__main__':
    wav_files = files_in_folders('./testing_team')
    #print(files)
    #filename = './model/bark.wav' 
    #print(files)
    is_dog_barking(wav_files)
