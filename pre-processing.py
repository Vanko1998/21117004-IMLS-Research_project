# -*- coding: utf-8 -*-
import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse
import dill as pickle
import collections
from lib_extension import Substract_average_plus_P_2, IFFT, Smooth_Gaussian, Center_surround_diff, Normalise, RGB_0_255, Concatenation, FFT, Magnitude, Log10, Slice
from pipeline import Pipeline
import numpy as np
from joblib import Parallel, delayed
import warnings
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
matplotlib.use('TKAgg')

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])


def create_all(window_length, window_step,fft_min_freq, fft_max_freq, sampling_frequency, file_path):
    warnings.filterwarnings("ignore")
    type_data = pickle.load(open(file_path, 'rb'))
    pipeline_FT = Pipeline([FFT(), Magnitude(), Log10()])
    pipeline_s1 = Pipeline([Substract_average_plus_P_2(), IFFT(), Smooth_Gaussian()])
    pipeline_s2 = Pipeline([Center_surround_diff()])

    time_series_data = type_data.data
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    ft_data = []
    s1_data = []
    s2_data = []
    d_data = []

    while stop < time_series_data.shape[1]:
        signal_window = time_series_data[:, start:stop]
        window_ft = pipeline_FT.apply(signal_window)
        ft_data.append(window_ft)
        window_s1 = pipeline_s1.apply(signal_window)
        s1_data.append(window_s1)
        window_s2 = pipeline_s2.apply(signal_window)
        s2_data.append(window_s2)
        start, stop = start + step, stop + step

    ft_data = np.array(ft_data)
    s1_data = np.array(s1_data)
    s2_data = np.array(s2_data)
    pipeline_nor = Pipeline([Normalise()])
    pipeline_con = Pipeline([Concatenation()])
    pipeline_rgb = Pipeline([RGB_0_255()])

    # Normalise each window value
    #print(window_ft.size)
    #print("##############################################")
    #print(window_s1.size)
    #print("##############################################")
    #print(window_s2.size)
    window_ft_norm = pipeline_nor.apply(ft_data)
    window_s1_norm = pipeline_nor.apply(s1_data)
    window_s2_norm = pipeline_nor.apply(s2_data)
    window_all = [window_ft_norm, window_s1_norm, window_s2_norm]

    # Concatenate normalised values
    d_norm = pipeline_con.apply(window_all)
    # RGB 0-255 conversion
    d_rgb = pipeline_rgb.apply(d_norm)
    d_data.append(d_rgb)

    d_data = np.array(d_data)
    d_data = np.squeeze(d_data)
    df_new = pd.DataFrame()

    if len(d_data) >=60:
        d_data_rms = np.zeros((60,20),dtype=np.float)
        for i in range(0, 60):
            df = pd.DataFrame(d_data[i], columns=range(1, int(d_data[0].size/20)+1), index=range(1, 21))

            plot_heatmap(df)

            df_new = pd.concat([df_new, df], axis=1)
            for j in range(0, 20):
                d_data_rms[i][j] = math.sqrt(sum([x ** 2 for x in d_data[i][j]]) / len(d_data[i][j]))
                #print(d_data[i][j].size)
    elif len(d_data) < 60:
        d_data_rms = np.zeros((len(d_data),20),dtype=np.float)
        for i in range (0,len(d_data)):
            df = pd.DataFrame(d_data[i], columns=range(1, int(d_data[0].size/20)+1), index=range(1, 21))

            plot_heatmap(df)# plot time data

            df_new = pd.concat([df_new, df], axis=1)
            for j in range (0,20):
                d_data_rms[i][j] = math.sqrt(sum([x ** 2 for x in d_data[i][j]]) / len(d_data[i][j]))
                #print(d_data[i][j].size)


    #print(d_data_rms)
    #print(len(d_data))
    #print(d_data.size)
    #print(d_data[0].size)
    #print(type_data.seizure_type)
    #df_new = pd.DataFrame()
    #for i in range(0,60):
    #    df = pd.DataFrame(d_data[i],columns=range(1,61), index=range(1, 21))
    #    df_new = pd.concat([df_new, df], axis=1)
    #print(df_new)
    plot_heatmap(df_new)
    d_data_new = np.array(df_new)
    named_data_rms = seizure_type_data(patient_id=type_data.patient_id, seizure_type=type_data.seizure_type, data=d_data_rms)
    named_data = seizure_type_data(patient_id=type_data.patient_id, seizure_type=type_data.seizure_type,data=d_data_new)
    return named_data, named_data_rms, os.path.basename(file_path)


def plot_heatmap(df) -> object:
    fig, ax = plt.subplots(figsize=(100, 20))
    sns.heatmap(df, square=True)
    ax.set_title('heatmap of EEG signal', fontsize=14)
    ax.set_ylabel('Electrodes', fontsize=6)
    ax.set_xlabel('Sampling point ', fontsize=6)
    plt.show()


#These here down below are the same as in researchers' code to simplify usage
def main():
    parser = argparse.ArgumentParser(description='Generate Saliency maps and spectogram from preprocessed data')

    if platform.system() == 'Linux':
        parser.add_argument('-l','--save_data_dir', default='/slow1/out_datasets/tuh/seizure_type_classification/',
                            help='path to output updated data')
        parser.add_argument('-b','--base_save_data_dir', default='/fast1/out_datasets/tuh/seizure_type_classification/',
                            help='path to output updated data')
    elif platform.system() == 'Darwin':
        parser.add_argument('-l','--save_data_dir', default='/Users/jbtang/datasets/TUH/eeg_seizure/',
                            help='path to output updated data')
        parser.add_argument('-b','--preprocess_data_dir',
                            default='/Users/jbtang/datasets/TUH/output/seizures_type_classification/',
                            help='path to output updated data')
    elif platform.system() == 'Windows':
        parser.add_argument('-l','--save_data_dir', default='D:/datasets/seizure_result_data',
                            help='path to extracted seizure dataset')
        parser.add_argument('-b','--preprocess_data_dir',
                            default='D:/datasets/seizure_preprocessed_data(spectograms)',
                            help='path to preprocessed data')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.5.2',
                        help='path to output updated data')

    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver
    save_data_dir = os.path.join(args.save_data_dir + "/", tuh_eeg_szr_ver + "/", 'raw_seizures')
    preprocess_data_dir = os.path.join(args.preprocess_data_dir + "/", tuh_eeg_szr_ver + "/", 'fft')

    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(save_data_dir):
        fnames.extend(filenames)
    fpaths = [os.path.join(save_data_dir + "/", f) for f in fnames]

    sampling_frequency = [80, 100, 120] # Hz
    fft_min_freq = 1  # Hz
    window_length = [0.75, 1, 1.25]  # second
    fft_max_freq = 48  # Hz
    window_steps = 0.75  # second
    print('sampling_rate: ', sampling_frequency, 'window length: ', window_length, 'window step: ', window_steps, 'fft_max_freq', fft_max_freq)


    for sample_rate in sampling_frequency:
        for win_len in window_length:
            save_data_dir_rms = os.path.join(preprocess_data_dir,
                                         'RMS_fft_seizures' + '_wl_' + str(win_len) + '_ws_' + str(window_steps) \
                                         + '_sf_' + str(sample_rate) + '_fft_min_' + str(fft_min_freq) + '_fft_max_' + \
                                         str(48))
            save_data_dir_spe = os.path.join(preprocess_data_dir,
                                         'SPE_fft_seizures' + '_wl_' + str(win_len) + '_ws_' + str(window_steps) \
                                         + '_sf_' + str(sample_rate) + '_fft_min_' + str(fft_min_freq) + '_fft_max_' + \
                                         str(48))

            if not os.path.exists(save_data_dir_rms):
                os.makedirs(save_data_dir_rms)
                # Create each map in order, then create spectogram D
                for file_path in sorted(fpaths):
                    # print(file_path)
                    converted_data, converted_data_rms, file_name_base = create_all(win_len, window_steps, fft_min_freq,
                                                                                    fft_max_freq, sample_rate,
                                                                                    file_path)

                    # print(converted_data.data.ndim)
                    if converted_data_rms.data.ndim == 2:
                        pickle.dump(converted_data_rms, open(os.path.join(save_data_dir_rms, file_name_base), 'wb'))
            else:
                print('RMS_sampling_'+str(sample_rate)+'_window_'+str(win_len)+'_Pre-processed data already exists!')

            if not os.path.exists(save_data_dir_spe):
                os.makedirs(save_data_dir_spe)
                # Create each map in order, then create spectogram D
                for file_path in sorted(fpaths):
                    # print(file_path)
                    converted_data, converted_data_rms, file_name_base = create_all(win_len, window_steps, fft_min_freq,
                                                                                    fft_max_freq, sample_rate,
                                                                                    file_path)

                    #print(converted_data.data.ndim)
                    if converted_data.data.ndim == 2:
                        pickle.dump(converted_data, open(os.path.join(save_data_dir_spe, file_name_base), 'wb'))
            else:
                print('SPE_sampling_'+str(sample_rate)+'_window_'+str(win_len)+'_Pre-processed data already exists!')





if __name__ == '__main__':
    main()