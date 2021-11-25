import os
import numpy as np
from HMM import HMM
import matplotlib.pyplot as plt

def main():
    training_file = '../lists/trainingfile_list.npy'
    testing_file = '../lists/testingfile_list.npy'
    my_training_file = '../mylists/trainingfile_list.npy'
    my_testing_file = '../mylists/testingfile_list.npy'

    dim = 39
    num_of_model = 11
    num_of_state_start = 12
    num_of_state_end = 15
    accuracy_rate = np.zeros((num_of_state_end))
    my_accuracy_rate = np.zeros((num_of_state_end))

    color = ['b','m','g','y']
    pic_path = '../pic'
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    plt.figure()
    plt.xlabel('epoches')
    plt.ylabel('log likelihood')
    plt.title('mfcc from python_speech_features')
    for num_of_state in range(num_of_state_start,num_of_state_end + 1):
        hmm = HMM(dim, num_of_model, num_of_state)
        hmm.train(training_file, epoch = 2)
        accuracy_rate[num_of_state - 1] = hmm.test(testing_file)
        print(f'num_of_state: {num_of_state}, accuracy_rate: {accuracy_rate[num_of_state-1]}')
        
        plt.plot(np.arange(len(hmm.get_likelihood())),hmm.get_likelihood(),color=color[num_of_state-num_of_state_start],label = f'log likelihood of {num_of_state}')
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)  
    plt.savefig(os.path.join(pic_path, 'log_likelihood.png'))
    plt.close()

    plt.figure()
    plt.xlabel('epoches')
    plt.ylabel('log likelihood')
    plt.title('my mfcc')
    for num_of_state in range(num_of_state_start,num_of_state_end + 1):
        hmm = HMM(dim, num_of_model, num_of_state)
        hmm.train(my_training_file, epoch = 2)
        my_accuracy_rate[num_of_state - 1] = hmm.test(my_testing_file)
        print(f'num_of_state: {num_of_state}, accuracy_rate: {my_accuracy_rate[num_of_state-1]}')
        plt.plot(np.arange(len(hmm.get_likelihood())),hmm.get_likelihood(),color=color[num_of_state-num_of_state_start],label = f'log likelihood of {num_of_state}')
    plt.savefig(os.path.join(pic_path, 'my_log_likelihood.png'))
    plt.close()

if __name__ == "__main__":
    main()