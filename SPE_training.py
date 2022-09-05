import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras import layers
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
import pickle
from datetime import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.preprocessing import LabelBinarizer
import time

sampling_frequency = [80, 100, 120]  # Hz
window_length = [0.75, 1, 1.25]
EEG_WINDOWS = 6000
EEG_COLUMNS = 20
SZR_CLASSES_NUM = [0, 1, 2, 3, 4, 5, 6]
SZR_CLASSES = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']


def get_fold_data(data_dir, fold_data, dataType, labelEncoder):
    if dataType != None:
        data = fold_data.get(dataType) # train or val file name
    elif dataType == None:
        data = fold_data
    X = np.empty((len(data), EEG_WINDOWS, EEG_COLUMNS), dtype=np.float64)
    #print(X.shape)
    y = list()
    for i, fname in enumerate(data):
        # each file contains a named tupple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        y.append(seizure.seizure_type)
        #print(seizure)
        #length = len(seizure.data[0])
        #print(length)
        X[i] = np.resize(seizure.data, (EEG_WINDOWS, len(seizure.data)))
        #print(X[i].shape)
    #print(X.shape)
    #print(y.size)
    if labelEncoder != None:
      y_new = labelEncoder.transform(y)
    return X, y_new, y


def get_test_dataset(data_dir, fold_data, labelEncoder, type):
    if type == 'train':
        X_train, y_train, __ = get_fold_data(data_dir, fold_data, "train", labelEncoder)
        X_val, y_val, y_original = get_fold_data(data_dir, fold_data, "val", labelEncoder)
    elif type == 'test':
        X_train = None
        y_train = None
        X_val, y_val,y_original = get_fold_data(data_dir,fold_data, None, labelEncoder)
    return X_train, y_train, X_val, y_val, y_original


def buildModel(inputshape):
    newModel = Sequential([
        tf.keras.layers.InputLayer(input_shape=inputshape),
        # The first convolution layer, four 21x1 convolution kernels
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # The first pooling layer, maximum pooling,4 3x1 convolution kernels, step size 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # Second convolution layer, 16 23x1 convolution kernels
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # The second pooling layer, maximum pooling, four 3x1 convolution kernels with a step size of 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # The third convolution layer, 32 25x1 convolution kernels
        tf.keras.layers.Conv1D(filters=32, kernel_size=32, strides=1, padding='SAME', activation='relu'),
        # The third pooling layer, average pooling, has 4 3x1 convolution kernels with a step size of 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # The fourth convolution layer, 64 27x1 convolution kernels
        tf.keras.layers.Conv1D(filters=64, kernel_size=64, strides=1, padding='SAME', activation='relu'),
        # Flat layer, convenient full connection layer processing
        tf.keras.layers.Flatten(),
        # Fully connected layer,128 nodes
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # Fully connected layer,7 nodes
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return newModel


def train_model(data_dir, fold_data, labelEncoder, model_path, logs_dir, acc_per_fold, loss_per_fold, x_value):
    x_train, y_train, x_test, y_test, __ = get_test_dataset(data_dir, fold_data, labelEncoder,'train')
    if os.path.exists(model_path):
        # Import the trained model
        model = tf.keras.models.load_model(filepath=model_path)
    else:
        # Building CNN model
        model = buildModel((x_value,20))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # Define the TensorBoard object
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=1)
        # Training and Validation
        model.fit(x_train, y_train, epochs=20,
                  batch_size=8,
                  callbacks=[tensorboard_callback])
        #model.save('./SPE_test_model')

    # release model to test set, draw ROC, and calculate AUC
    # evaluate the model
    print("[INFO] evaluating network...")
    start = time.perf_counter()
    scores = model.evaluate(x_test, y_test)
    totaltime = time.perf_counter()-start
    print(f'Score for fold {data_dir}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    return acc_per_fold, loss_per_fold, model, x_train, y_train, x_test, y_test, totaltime


def bar_plot(acc_train, acc_valid, acc_test, x_axis_all):
    a = acc_train
    b = acc_valid
    c = acc_test

    x = np.arange(9)

    # There are three types of data: A, B, and C, and n is set to 3
    total_width, n = 0.8, 3
    # Width of the bar chart for each type
    width = total_width / n

    # Reset the X-axis coordinates
    x = x - (total_width - width) / 2
    # Draw a histogram
    plt.bar(x, a, width=width, label="train")
    plt.bar(x + width, b, width=width, label="validation")
    plt.bar(x + 2 * width, c, width=width, label="test")
    # According to legend
    plt.title("SPE_train_validation_test_accuracy")
    plt.xlabel("sample rate, window length")
    plt.ylabel("accuracy")
    for i, j in zip(x, a):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
    for i, j in zip(x + width, b):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
    for i, j in zip(x + 2 * width, c):
        plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
    plt.xticks(x, x_axis_all)
    plt.legend()
    # The bar chart is displayed
    #plt.savefig(fname="./train_validation_test.png", dpi=100)
    plt.show()


def whole_process(cross_val_file, test_file, data_dir, sr, wl):
    seizure_folds = pickle.load(open(cross_val_file, "rb"))
    test_fold = pickle.load(open(test_file, "rb"))
    k_validation_folds = list(seizure_folds.values())
    # seizure_folds = list(seizure_folds.values())
    # k_validation_folds = seizure_folds[:-1]  # Choose which dataset will be the test randomly
    # test_fold = seizure_folds[-1]
    # print(test_fold)
    le = LabelBinarizer()
    le.fit(SZR_CLASSES)
    __, __, X_test, y_test, y_original = get_test_dataset(data_dir, test_fold, le, 'test')
    currentTime = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []
    totaltime = []

    for fold_no, fold_data in enumerate(k_validation_folds):

        logs_dir = f"logs/fit/{currentTime}/fold_{fold_no}"
        acc_per_fold, loss_per_fold, model, x_train, y_train, x_val, y_val, time_t = train_model(data_dir, fold_data, le,
                                                                                         "./SPE_test_model_sr_"+str(sr)+"_wl_"+str(wl), logs_dir,
                                                                                         acc_per_fold, loss_per_fold,
                                                                                         60 * 100)
        totaltime.append(time_t)

    if not os.path.exists("./SPE_test_model_sr_"+str(sr)+"_wl_"+str(wl)):
        model.save("./SPE_test_model_sr_"+str(sr)+"_wl_"+str(wl))
    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)}')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')
    print("[INFO] Predicting network:")
    # save_confusion_matrix(test_labels, test_pred, le.classes_, logs_dir)
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter()-start
    y_pred = np.argmax(y_pred, axis=1)
    if le != None:
        y_new_test = np.argmax(le.transform(y_original), axis=1)
    y_pred_train = np.argmax(model.predict(x_train), axis=1)
    y_pred_val = np.argmax(model.predict(x_val), axis=1)
    print('Accuracy of train group:')
    acc_train = accuracy_score(np.argmax(y_train, axis=1), y_pred_train)
    print(acc_train)
    print('Accuracy of validation group:')
    acc_valid = accuracy_score(np.argmax(y_val, axis=1), y_pred_val)
    print(acc_valid)
    print('Prediction result:')
    print(y_pred)
    print('True result:')
    print(y_new_test)
    print('Accuracy of test group:')
    acc_test = accuracy_score(y_new_test, y_pred)
    print(acc_test)
    return acc_train, acc_valid, acc_test, totaltime, pred_time


def main():
    acc_train = []
    acc_valid = []
    acc_test = []
    x_axis_all = []
    x_aixs_wl = []
    x_axis_sr = []
    avr_time = []
    pred_time = []
    all_time = []
    for sr in sampling_frequency:
        for wl in window_length:
            all_start = time.perf_counter()
            data_dir = "D:/datasets/seizure_preprocessed_data(spectograms)/v1.5.2/fft/SPE_fft_seizures_wl_" + str(wl) + "_ws_0.75_sf_" + str(sr) + "_fft_min_1_fft_max_48"  # "/home/david/Documents/Machine Learning/raw_data/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
            cross_val_file = "spe_wl_" + str(wl) + "_ws_0.75_sf_" + str(sr) + "_cv_split_5_fold_seizure_wise_v1.5.2.pkl"
            test_file = "spe_wl_" + str(wl) + "_ws_0.75_sf_" + str(sr) + "_cv_split_5_fold_seizure_wise_v1.5.2_test.pkl"
            acc_tr, acc_va, acc_te, totaltime, pred_t = whole_process(cross_val_file, test_file, data_dir, sr, wl)
            acc_train.append(acc_tr)
            acc_valid.append(acc_va)
            acc_test.append(acc_te)
            x_axis_all.append(str(sr) + ',' + str(wl))
            avr = sum(totaltime)/len(totaltime)
            avr_time.append(avr)
            pred_time.append(pred_t)
            all_time.append(time.perf_counter()-all_start)
    np.savetxt('./SPE_pred_time.txt', pred_time)
    np.savetxt('./SPE_avr_evaluate_time.txt', avr_time)
    np.savetxt('./SPE_test_acc.txt', acc_test)
    np.savetxt('./SPE_x_axis.txt', x_axis_all, fmt="%s", delimiter="")
    np.savetxt('./SPE_all_time.txt', all_time)
    bar_plot(acc_train, acc_valid, acc_test, x_axis_all)




if __name__ == "__main__":
  main()
