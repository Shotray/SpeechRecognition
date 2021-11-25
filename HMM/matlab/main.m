generate_mfcc_samples(); % extract mfcc features for all (training and test) samples
generate_testing_list_mat; % generate test list
generate_training_list_mat; % generate training list

% clear;
training_file_list_name = 'trainingfile_list.mat';
testing_file_list_name = 'testingfile_list.mat';

DIM = 39; % dimension of a feature vector
num_of_model = 11; % number of models: digit '0', digit '1',... digit '9', digit 'zero'

% the # of states variates from 12 to 15 and tested respectively
num_of_state_start = 12;  
num_of_state_end = 15;

accuracy_rate = zeros(1,num_of_state_end);

for num_of_state = num_of_state_start : num_of_state_end
    % notice: number of state does not including START and END node in HMM
    HMM = EM_HMMtraining(training_file_list_name, DIM, num_of_model, num_of_state);  % training phase
    accuracy_rate(num_of_state) = HMMtesting(HMM, testing_file_list_name);                       % testing phase
    fprintf('num_of_state: %d, accuracy_rate: %f\n',num_of_state, accuracy_rate(num_of_state));
    save accuracy_rate;    
end