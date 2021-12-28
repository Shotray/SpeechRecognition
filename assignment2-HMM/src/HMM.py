import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures

np.seterr(divide='ignore')  

class HMM:
    def __init__(self,dim = 39, num_of_model = 11, num_of_state = 13):
        self.dim = dim
        self.num_of_model = num_of_model
        self.num_of_state = num_of_state
        self.mean = np.zeros((dim, num_of_state, num_of_model))
        self.var = np.zeros((dim, num_of_state, num_of_model))
        self.Aij = np.zeros((num_of_state + 2, num_of_state + 2, num_of_model))
        self.likelihood = []
    
    def get_likelihood(self):
        return self.likelihood


    def load(self, model_path):
        model = np.load(model_path)
        self.mean = model['mean']
        self.var = model['var']
        self.Aij = model['Aij']

    @staticmethod
    def logGaussian(mean, var, o):
        dim = var.shape[0]
        return -1/2*(dim*np.log(2*np.pi) + np.sum(np.log(var)) + np.sum(np.square(o - mean) / var))
    
    @staticmethod
    def log_sum_alpha(log_alpha_t,aij_j):
        len_x = log_alpha_t.shape[0]
        y = np.full((len_x), -np.inf)
        ymax = -np.inf
        for i in range(len_x):
            y[i] = log_alpha_t[i] + np.log(aij_j[i])
            if y[i] > ymax:
                ymax = y[i]
        if ymax == np.inf:
            return np.inf
        else:
            sum_exp = 0
            for i in range(len_x):
                if ymax == -np.inf and y[i] == -np.inf:
                    sum_exp += 1
                else:
                    sum_exp += np.exp(y[i] - ymax)
            return ymax + np.log(sum_exp)
    
    @staticmethod
    def log_sum_beta(aij_i, mean, var, obs, beta_t1):
        len_x = mean.shape[1] # number of state
        y = np.full((len_x), -np.inf)
        ymax = -np.inf
        for j in range(len_x):
            y[j] = np.log(aij_i[j]) + HMM.logGaussian(mean[:,j], var[:,j], obs) + beta_t1[j]
            if y[j] > ymax:
                ymax = y[j]
        if ymax == np.inf:
            return np.inf
        else:
            sum_exp = 0
            for i in range(len_x):
                if ymax == -np.inf and y[i] == -np.inf:
                    sum_exp += 1
                else:
                    sum_exp += np.exp(y[i] - ymax)
            return ymax + np.log(sum_exp)

    def initial_train_model(self, training_file_list):
        sum_of_features = np.zeros((self.dim))
        sum_of_features_square = np.zeros((self.dim))
        num_of_feature = 0

        training_files = np.load(training_file_list)
        '''['1' 'mfcc\\AE\\1A_endpt.npy']'''
        for training_file in training_files:
            filename = training_file[1]
            features = np.load(filename)
            sum_of_features += np.sum(features, axis = 1)
            sum_of_features_square += np.sum(np.square(features), axis = 1)
            num_of_feature += features.shape[1] # (39,44) num_of_features += 44
        
        for k in range(self.num_of_model):
            for n in range(self.num_of_state):
                self.mean[:, n, k] = sum_of_features / num_of_feature
                self.var[:, n, k] = sum_of_features_square / num_of_feature
            for i in range(1, self.num_of_state + 1):
                self.Aij[i, i+1, k] = 0.4
                self.Aij[i, i, k] = 1 - self.Aij[i, i+1, k]
            self.Aij[0, 1, k] = 1

    @staticmethod
    def single_thread_FR(mean, var, aij, obs):
        obs = np.load(obs)
        dim, length = obs.shape
        mean = np.hstack((np.full((dim, 1), np.nan), mean, np.full((dim, 1), np.nan)))
        var = np.hstack((np.full((dim, 1), np.nan), var, np.full((dim, 1), np.nan)))

        aij[-1,-1] = 1
        N = mean.shape[1]
        log_alpha = np.full((N, length + 1), - np.inf)
        log_beta = np.full((N, length + 1), - np.inf)

        for i in range(N):
            log_alpha[i, 0] = np.log(aij[0, i]) + HMM.logGaussian(mean[:,i], var[:,i], obs[:,0])

        for t in range(1, length): # calculate alpha
            for j in range(1, N-1):
                log_alpha[j, t] = HMM.log_sum_alpha(log_alpha[1:N-1, t-1], aij[1:N-1, j]) + HMM.logGaussian(mean[:,j],var[:,j],obs[:,t])
        log_alpha[N-1,length] = HMM.log_sum_alpha(log_alpha[1:N-1,length-1], aij[1:N-1,N-1])

        log_beta[:, length-1] = np.log(aij[:, N-1])
        for t in range(length-2, -1, -1):
            for i in range(1, N-1):
                log_beta[i,t] = HMM.log_sum_beta(aij[i,1:N-1],mean[:,1:N-1],var[:,1:N-1],obs[:,t+1],log_beta[1:N-1,t+1])
        log_beta[N-1,0] = HMM.log_sum_beta(aij[0,1:N-1],mean[:,1:N-1],var[:,1:N-1],obs[:,0],log_beta[1:N-1,0])

        log_Xi = np.full((N,N,length), -np.inf)
        for t in range(length-1):
            for j in range(1, N-1):
                for i in range(1, N-1):
                    log_Xi[i,j,t] = log_alpha[i,t] + np.log(aij[i,j]) + HMM.logGaussian(mean[:,j],var[:,j],obs[:,t+1]) + log_beta[j,t+1] - log_alpha[N-1,length]
        
        for i in range(N):
            log_Xi[i,N-1,length-1] = log_alpha[i,length-1] + np.log(aij[i,N-1]) - log_alpha[N-1, length]
        
        log_gamma = np.full((N,length), -np.inf)
        for t in range(length):
            for i in range(1, N-1):
                log_gamma[i,t] = log_alpha[i,t] + log_beta[i,t] - log_alpha[N-1,length]
        gamma = np.exp(log_gamma)

        mean_numerator = np.zeros((dim, N))
        var_numerator = np.zeros((dim, N))
        denominator = np.zeros((N))
        aij_numerator = np.zeros((N, N))
        for j in range(1, N-1):
            for t in range(length):
                mean_numerator[:,j] = mean_numerator[:,j] + gamma[j,t]*obs[:,t]
                var_numerator[:,j] = var_numerator[:,j] + gamma[j,t]*np.square(obs[:,t])
                denominator[j] = denominator[j] + gamma[j,t]
        for i in range(1, N-1):
            for j in range(1, N-1):
                for t in range(length):
                    aij_numerator[i,j] = aij_numerator[i,j] + np.exp(log_Xi[i,j,t])
        
        log_likelihood = log_alpha[N-1, length]
        likelihood = np.exp(log_likelihood)

        return mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood, likelihood
    
    def muti_thread_FR(mean, var, aij, obs, k):
        obs = np.load(obs)
        dim, length = obs.shape
        mean = np.hstack((np.full((dim, 1), np.nan), mean, np.full((dim, 1), np.nan)))
        var = np.hstack((np.full((dim, 1), np.nan), var, np.full((dim, 1), np.nan)))

        aij[-1,-1] = 1
        N = mean.shape[1]
        log_alpha = np.full((N, length + 1), - np.inf)
        log_beta = np.full((N, length + 1), - np.inf)

        for i in range(N):
            log_alpha[i, 0] = np.log(aij[0, i]) + HMM.logGaussian(mean[:,i], var[:,i], obs[:,0])

        for t in range(1, length):
            for j in range(1, N-1):
                log_alpha[j, t] = HMM.log_sum_alpha(log_alpha[1:N-1, t-1], aij[1:N-1, j]) + HMM.logGaussian(mean[:,j],var[:,j],obs[:,t])
        log_alpha[N-1,length] = HMM.log_sum_alpha(log_alpha[1:N-1,length-1], aij[1:N-1,N-1])

        log_beta[:, length-1] = np.log(aij[:, N-1])
        for t in range(length-2, -1, -1):
            for i in range(1, N-1):
                log_beta[i,t] = HMM.log_sum_beta(aij[i,1:N-1],mean[:,1:N-1],var[:,1:N-1],obs[:,t+1],log_beta[1:N-1,t+1])
        log_beta[N-1,0] = HMM.log_sum_beta(aij[0,1:N-1],mean[:,1:N-1],var[:,1:N-1],obs[:,0],log_beta[1:N-1,0])

        log_Xi = np.full((N,N,length), -np.inf)
        for t in range(length-1):
            for j in range(1, N-1):
                for i in range(1, N-1):
                    log_Xi[i,j,t] = log_alpha[i,t] + np.log(aij[i,j]) + HMM.logGaussian(mean[:,j],var[:,j],obs[:,t+1]) + log_beta[j,t+1] - log_alpha[N-1,length]
        
        for i in range(N):
            log_Xi[i,N-1,length-1] = log_alpha[i,length-1] + np.log(aij[i,N-1]) - log_alpha[N-1, length]
        
        log_gamma = np.full((N,length), -np.inf)
        for t in range(length):
            for i in range(1, N-1):
                log_gamma[i,t] = log_alpha[i,t] + log_beta[i,t] - log_alpha[N-1,length]
        gamma = np.exp(log_gamma)

        mean_numerator = np.zeros((dim, N))
        var_numerator = np.zeros((dim, N))
        denominator = np.zeros((N))
        aij_numerator = np.zeros((N, N))
        for j in range(1, N-1):
            for t in range(length):
                mean_numerator[:,j] = mean_numerator[:,j] + gamma[j,t]*obs[:,t]
                var_numerator[:,j] = var_numerator[:,j] + gamma[j,t]*np.square(obs[:,t])
                denominator[j] = denominator[j] + gamma[j,t]
        for i in range(1, N-1):
            for j in range(1, N-1):
                for t in range(length):
                    aij_numerator[i,j] = aij_numerator[i,j] + np.exp(log_Xi[i,j,t])
        
        log_likelihood = log_alpha[N-1, length]
        likelihood = np.exp(log_likelihood)

        return mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood, likelihood, k

    def train(self, training_file_list = '../lists/trainingfile_list.npy', epoch = 2):
        self.initial_train_model(training_file_list)

        log_likelihood_iter = np.zeros((epoch))
        likelihood_iter = np.zeros((epoch))
        training_files = np.load(training_file_list)

        for iter in range(epoch):
            '''reset value of sum_of_features, sum_of_features_square, num_of_feature, num_of_jump'''
            sum_mean_numerator = np.zeros((self.dim, self.num_of_state, self.num_of_model))
            sum_var_numerator = np.zeros((self.dim, self.num_of_state, self.num_of_model))
            sum_aij_numerator = np.zeros((self.num_of_state, self.num_of_state, self.num_of_model))
            sum_denominator = np.zeros((self.num_of_state, self.num_of_model))
            log_likelihood = 0
            likelihood = 0

            # single thread
            # for file in tqdm(training_files):
            #     k = int(file[0]) - 1
            #     filename = file[1]
            #     mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood_i, likelihood_i = HMM.single_thread_FR(self.mean[:,:,k], self.var[:,:,k], self.Aij[:,:,k], filename)

            #     sum_mean_numerator[:,:,k] += mean_numerator[:,1:-1]
            #     sum_var_numerator[:,:,k] += var_numerator[:,1:-1]
            #     sum_aij_numerator[:,:,k] += aij_numerator[1:-1,1:-1]
            #     sum_denominator[:,k] += denominator[1:-1]

            #     log_likelihood += log_likelihood_i
            #     likelihood += likelihood_i

            # muti thread
            k_arr = [int(file[0]) - 1 for file in training_files]
            filename_arr = training_files[:,1]
            mean_arr = [self.mean[:,:,k] for k in k_arr]
            var_arr = [self.var[:,:,k] for k in k_arr]
            Aij_arr = [self.Aij[:,:,k] for k in k_arr]

            with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
                for res in tqdm(executor.map(HMM.muti_thread_FR, mean_arr, var_arr, Aij_arr, filename_arr, k_arr), total=len(k_arr), desc=f'epoch {iter}'):
                    mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood_i, likelihood_i, k = res
                    sum_mean_numerator[:,:,k] += mean_numerator[:,1:-1]
                    sum_var_numerator[:,:,k] += var_numerator[:,1:-1]
                    sum_aij_numerator[:,:,k] += aij_numerator[1:-1,1:-1]
                    sum_denominator[:,k] += denominator[1:-1]

                    log_likelihood += log_likelihood_i
                    likelihood += likelihood_i
            
            for k in range(self.num_of_model):
                for n in range(self.num_of_state):
                    self.mean[:,n,k] = sum_mean_numerator[:,n,k] / sum_denominator[n,k]
                    self.var[:,n,k] = sum_var_numerator[:,n,k] / sum_denominator[n,k] - np.square(self.mean[:,n,k])
            for k in range(self.num_of_model):
                for i in range(1, self.num_of_state + 1):
                    for j in range(1, self.num_of_state + 1):
                        self.Aij[i,j,k] = sum_aij_numerator[i-1,j-1,k] / sum_denominator[i-1,k]
                self.Aij[self.num_of_state, self.num_of_state+1, k] = 1 - self.Aij[self.num_of_state, self.num_of_state, k]
            self.Aij[self.num_of_state+1, self.num_of_state+1, k] = 1
            log_likelihood_iter[iter] = log_likelihood
            likelihood_iter[iter] = likelihood

        self.likelihood = log_likelihood_iter

        # np.savez('model', mean=self.mean, var=self.var, Aij=self.Aij)
        return
    
    @staticmethod
    def single_thread_viterbi_dist_FR(mean, var, aij, obs):
        dim, t_len = obs.shape
        mean = np.hstack((np.full((dim,1),0),mean,np.full((dim,1),0)))
        var = np.hstack((np.full((dim,1),0),var,np.full((dim,1),0)))
        aij[-1, -1] = 1
        m_len = mean.shape[1] 
        fjt = np.full((m_len, t_len), -np.inf)

        for j in range(1, m_len-1):
            fjt[j,0] = np.log(aij[0,j]) + HMM.logGaussian(mean[:,j],var[:,j], obs[:,0])

        
        for t in range(1, t_len):
            for j in range(1, m_len - 1):
                f_max, i_max, f = -np.inf, -1, -np.inf
                for i in range(1, j+1):
                    if fjt[i, t-1] > -np.inf and aij[i, j] > 0:
                        f = fjt[i,t-1] + np.log(aij[i,j]) + HMM.logGaussian(mean[:,j],var[:,j],obs[:,t])
                    if f > f_max: 
                        f_max = f
                        i_max = i 
                if i_max != -1:

                    fjt[j, t] = f_max
        fopt = -np.inf
        for i in range(1, m_len - 1):
            f = fjt[i, t_len-1] + np.log(aij[i, m_len-1])
            if f > fopt:
                fopt = f
        return fopt

    def muti_thread_test(self,filename,k):
        features = np.load(filename)
        fopt_max = -np.inf
        digit = -1
        for p in range(self.num_of_model):
                fopt = HMM.single_thread_viterbi_dist_FR(self.mean[:,:,p], self.var[:,:,p], self.Aij[:,:,p], features) 
                if fopt > fopt_max:
                    digit = p
                    fopt_max = fopt
        return 1 if digit!=k else 0

    def test(self,testing_file_list = '../lists/testingfile_list.npy'):
        testingfile = np.load(testing_file_list)  
        num_of_error = 0
        num_of_testing = testingfile.shape[0]

        #single thread
        # for u in tqdm(testingfile, desc='test'):
        #     k = int(u[0]) - 1
        #     filename = u[1]
        #     features = np.load(filename)
        #     fopt_max = -np.inf
        #     digit = -1

        #     for p in range(self.num_of_model):
        #         fopt = HMM.single_thread_viterbi_dist_FR(self.mean[:,:,p], self.var[:,:,p], self.Aij[:,:,p], features) 
        #         if fopt > fopt_max:
        #             digit = p
        #             fopt_max = fopt
        #     if digit != k: 
        #         num_of_error += 1

        # muti thread
        filename_arr = testingfile[:,1]
        k_arr = [int(file[0])-1 for file in testingfile]

        with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
            for res in tqdm(executor.map(self.muti_thread_test,filename_arr, k_arr), total=len(k_arr), desc='test'):
                num_of_error += res
        accuracy_rate = (num_of_testing - num_of_error) * 100 / num_of_testing
        return accuracy_rate



if __name__ == '__main__':
    test = HMM()
    test.load('model.npz')
    test.test('..\\mylists\\testingfile_list.npy')
    
    
