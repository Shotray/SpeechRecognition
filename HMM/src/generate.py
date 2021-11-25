import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc,delta
import time
from speech import get_mfcc

def generate_mfcc_samples(read_dir = "../wav", write_dir = "../mfcc"):
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    for files in os.listdir(read_dir):
        for file in os.listdir(os.path.join(read_dir, files)):
            if not os.path.exists(os.path.join(write_dir, files)):
                os.mkdir(os.path.join(write_dir, files))
        
            origin_file_path = os.path.join(read_dir,files,file)
            mfcc_file_path = os.path.join(write_dir,files,file.split('.')[0]) + ".npy"
            fs, audio = wav.read(origin_file_path)
            mfcc_feature = mfcc(audio,samplerate=fs)
            delta_mfcc1 = delta(mfcc_feature,1)
            delta_mfcc2 = delta(mfcc_feature,2)
            mfcc_features = np.hstack((mfcc_feature,delta_mfcc1,delta_mfcc2))
            np.save(mfcc_file_path, mfcc_features.T)

def generate_my_mfcc_samples(read_dir = "../wav", write_dir = "../my_mfcc"):
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    for files in os.listdir(read_dir):
        for file in os.listdir(os.path.join(read_dir, files)):
            if not os.path.exists(os.path.join(write_dir, files)):
                os.mkdir(os.path.join(write_dir, files))
        
            origin_file_path = os.path.join(read_dir,files,file)
            mfcc_file_path = os.path.join(write_dir,files,file.split('.')[0]) + ".npy"
            fs, audio = wav.read(origin_file_path)
            mfcc_feature = get_mfcc(audio,samplerate=fs)
            np.save(mfcc_file_path, mfcc_feature.T)

def generate_training_list(list_filename = 'trainingfile_list.npy',list_path = '../lists',dir1 = 'mfcc'):
    if not os.path.exists(list_path):
        os.mkdir(list_path)
    save_path = os.path.join(list_path,list_filename)
    dir2 = ['AE','AJ','AL','AW','BD','CB','CF','CR','DL','DN','EH','EL','FC','FD','FF','FI','FJ','FK','FL','GG']
    wordids = ['1','2','3','4','5','6','7','8','9','O','Z']
    passes = ['A', 'B']
    trainingfile = []
    for d in dir2:
        for w in range(len(wordids)):
            for p in passes:
                trainingfile.append([w+1, os.path.join('..',dir1, d, f'{wordids[w]}{p}_endpt.npy')])
    np.save(save_path,np.array(trainingfile))
    print(trainingfile)
    exit()

def generate_testing_list(list_filename = 'testingfile_list.npy',list_path = '../lists',dir1 = 'mfcc'):
    if not os.path.exists(list_path):
        os.mkdir(list_path)
    save_path = os.path.join(list_path,list_filename)
    dir2 = ['AH','AR','AT','BC','BE','BM','BN','CC','CE','CP','DF','DJ','ED','EF','ET','FA','FG','FH','FM','FP','FR','FS','FT','GA','GP','GS','GW','HC','HJ','HM','HR','IA','IB','IM','IP','JA','JH','KA','KE','KG','LE','LG','MI','NL','NP','NT','PC','PG','PH','PR','RK','SA','SL','SR','SW','TC']
    wordids = ['1','2','3','4','5','6','7','8','9','O','Z']
    passes = ['A', 'B']
    testingfile = []
    for d in dir2:
        for w in range(len(wordids)):
            for p in passes:
                testingfile.append([w+1, os.path.join('..',dir1, d, f'{wordids[w]}{p}_endpt.npy')])
    np.save(save_path,np.array(testingfile))


if __name__ == "__main__":
    # generate_mfcc_samples()
    # generate_training_list()
    # generate_testing_list()

    generate_my_mfcc_samples()
    generate_training_list('trainingfile_list.npy','../mylists','my_mfcc')
    generate_testing_list('testingfile_list.npy','../mylists','my_mfcc')

