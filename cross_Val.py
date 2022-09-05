import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import pickle
import collections
import numpy as np
import platform
import argparse
import pandas as pd

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])


def generate_seizure_wise_cv_folds(data_dir, num_split, wl, sr):
    #print(data_dir)
    seizures_by_type = collections.defaultdict(list)
    total_szr_num = 0
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            #print("############")
            #print(fname)
            sz = pickle.load(open(os.path.join(data_dir+'/',fname), 'rb'))
            #df = pd.DataFrame()
            #print(df.to_csv(r'cv_split_5_fold_patient_wise_v1.5.2.pkl'))
            #print(sz)
            seizures_by_type[sz.seizure_type].append(fname)
            total_szr_num += 1
    print('sampling rate:', sr, 'window_length:', wl)
    print('Total found Seizure Num =%d' % (total_szr_num))
    print('training set: ', total_szr_num*0.8, 'test set: ', total_szr_num*0.2)
    # Delete mycolnic seizures since there are only three of them
    # del seizures_by_type['MYSZ']
    kf = KFold(n_splits=num_split, shuffle=True)

    cv_split = {}
    test_group = list()
    for i in range(1, num_split+1):
        cv_split[str(i)] = {'train': [], 'val': []}

    for type, fname_list in seizures_by_type.items():
        fname_list = np.array(fname_list)
        train_list, test_list = train_test_split(fname_list, test_size=0.2)
        test_group.extend(test_list)
        for index, (train_index, val_index) in enumerate(kf.split(train_list), start=1):
            cv_split[str(index)]['train'].extend(fname_list[train_index])
            cv_split[str(index)]['val'].extend(fname_list[val_index])

    # check allocated szr num
    for i in range(1, num_split + 1):
        print('Fold %d Train Seizure Number = %d'%(i,len(cv_split[str(i)]['train'])))
        print('Fold %d Val Seizure Number = %d' %(i,len(cv_split[str(i)]['val'])))
    print(test_group)
    return cv_split, test_group


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BCH Data Training')

    if platform.system() == 'Linux':
        parser.add_argument('--save_data_dir',
                            default='/slow1/out_datasets/tuh/seizure_type_classification/',
                            help='path to output prediction')
    elif platform.system() == 'Darwin':
        parser.add_argument('--save_data_dir',
                            default='/Users/jbtang/datasets/TUH/output/seizures_type_classification/fft/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12',
                            help='path to output prediction')
    elif platform.system() == 'Windows':
        parser.add_argument('--save_data_dir',
                            default='D:/datasets/seizure_preprocessed_data(spectograms)/v1.5.2/fft',
                            help='path to output prediction')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default='v1.5.2',
                        #default='v1.4.0',
                        help='version of TUH seizure dataset')

    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver
    sampling_frequency = [80, 100, 120]  # Hz
    window_length = [0.75, 1, 1.25]

    if tuh_eeg_szr_ver == 'v1.5.2': # for v1.5.2
        print('\nGenerating seizure wise cross validation for', tuh_eeg_szr_ver)
        fold_num = 4
        for sr in sampling_frequency:
            for wl in window_length:
                save_data_dir_rms = os.path.join(args.save_data_dir, 'RMS_fft_seizures_wl_'+ str(wl)+ '_ws_0.75_sf_'+ str(sr)+ '_fft_min_1_fft_max_48')
                cv_random, test_group = generate_seizure_wise_cv_folds(save_data_dir_rms, fold_num, wl, sr)
                pickle.dump(cv_random, open('rms_wl_'+ str(wl)+ '_ws_0.75_sf_'+ str(sr)+ '_cv_split_5_fold_seizure_wise_v1.5.2.pkl', 'wb'))
                pickle.dump(test_group, open('rms_wl_'+ str(wl)+ '_ws_0.75_sf_'+ str(sr)+ '_cv_split_5_fold_seizure_wise_v1.5.2_test.pkl', 'wb'))
                #pickle.dump(cv_random, open('spe_wl_' + str(wl) + '_ws_0.75_sf_' + str(sr) + '_cv_split_5_fold_seizure_wise_v1.5.2.pkl','wb'))
                #pickle.dump(test_group, open('spe_wl_' + str(wl) + '_ws_0.75_sf_' + str(sr) + '_cv_split_5_fold_seizure_wise_v1.5.2_test.pkl','wb'))
    else:
        exit('Not supported version number %s'%tuh_eeg_szr_ver)
