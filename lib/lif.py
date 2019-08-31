#Simulate LIF neurons
import numpy as np
import numpy.random as rand
import scipy

import cPickle, gzip
import random
from scipy.integrate import quadrature, quad
import pickle

import multiprocessing

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def mnist_spike(lamba, D, kernel, test_data = False):
    if test_data:
        rand_num = random.randint(0, test_set[0].shape[0]-1)
        mnist_rand = (test_set[0][rand_num]) 
        label_rand = test_set[1][rand_num]
    else:
        rand_num = random.randint(0, train_set[0].shape[0]-1)
        mnist_rand = (train_set[0][rand_num]) 
        label_rand = train_set[1][rand_num]

    new_var = np.zeros([784, D])
    for i in range(mnist_rand.shape[0]):
        x = np.random.uniform(0, 1, D)
        #spike if pixel intensity is grater than 1-lamba*I_i 
        store_bool = (x < (lamba*mnist_rand[i])).astype(float)
        new_var[i,:] = np.convolve(store_bool, kernel)[0:1000]
    return new_var, label_rand

def convolve_online(s, h, kernel, t_offset):
    if len(s.shape) > 1:
        n = s.shape[0]
        for i in range(n):
            for idx in np.nonzero(h[i,:])[0]:
                st = t_offset + idx
                en = min(s.shape[0], st + kernel.shape[0])
                ln = en-st
                #print st,en,ln,idx,t_offset
                s[i,st:en] += kernel[0:ln]
    else:
        for idx in np.nonzero(h)[0]:
            st = t_offset + idx
            en = min(s.shape[0], st + kernel.shape[0])
            ln = en-st
            #print st,en,ln,idx,t_offset
            s[st:en] += kernel[0:ln]

def convolve_online_v2(s, sp_idx, time_idx, kernel, t_offset):
    st = t_offset + time_idx
    en = min(s.shape[1], st + kernel.shape[0])
    ln = en-st
    s[sp_idx, st:en] += kernel[0:ln]

class ParamsLIF_Recurrent(object):    
    def __init__(self, kernel, dt = 0.001, tr = 0.003, mu = 1, reset = 0, xsigma = 1, n1 = 100, n2 = 10, tau = 1,
        c = 0.99, sigma = 20, sigma1 = 10, sigma2 = 2, batch_size = 32, n_input = 784, eta = 5e-1, eta_B = 0,
        p = 0.2):

        self.dt = dt                    #Step size
        self.tr = tr                    #Refractory period
        self.mu = mu                    #Threshold
        self.reset = reset              #Reset potential
        self.xsigma = xsigma            #Std dev of input x
        self.n_input = n_input          #Number of neurons in input layer
        self.n1 = n1                    #Number of neurons in first layer
        self.n2 = n2                    #Number of neurons in second layer
        self.n = n1 + n2                #Total number of neurons
        self.tau = tau                  #Time constant
        self.c = c                      #Correlation between noise inputs
        self.sigma = sigma              #Std dev of noise process
        self.kernel = kernel            #Kernel to apply to spike trains
        self.batch_size = batch_size    #Batch size
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.eta = eta                  #Learning rate
        self.eta_B = eta_B              #Feedback weights learning rate
        self.p = p                      #RDD window size

class LIF_Recurrent(object):

    def __init__(self, params, t = 1, parallel = False, n_proc = 4):
        self.parallel = parallel
        self.setup(params, t)

    def setup(self, params = None, t = None):
        if params is not None:
            self.params = params
        if t is not None:
            self.t = t

        #Initialize voltage and spike train variables
        #Simulation time in timestep units
        self.T =  np.ceil(self.t/self.params.dt).astype(int)
        #Refractory period in timestep units
        self.Tr = np.ceil(self.params.tr/self.params.dt).astype(int)
        self.times = np.linspace(0,self.t,self.T)

        #Input to each neuron in first layer weights        
        self.W1 = self.params.sigma1*np.ones(self.params.n1)
        self.W2 = self.params.sigma2*np.ones(self.params.n2)
        self.W = 50*np.random.randn(self.params.n1, self.params.n_input)+10
        #The feedforward/recurrent connections
        self.U = np.zeros((self.params.n, self.params.n))        
        self.U[self.params.n1:, 0:self.params.n1] = 50*np.random.randn(self.params.n2, self.params.n1)+10
        self.sh = np.zeros((self.params.n, self.T))
        self.B = 50*np.random.randn(self.params.n2, self.params.n1)+10
        #self.U[0:self.params.n1, self.params.n1:] = 500*np.random.randn(self.params.n1, self.params.n2)

    def save(self, fn_out):
        to_save = {
            'params': self.params,
            'W1': self.W1,
            'W2': self.W2,
            'W': self.W,
            'U': self.U,
            'B': self.B
        }
        pickle.dump(to_save, open(fn_out, "wb"))

    def restore(self, fn_in):
        with open(fn_in, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.W1 = data['W1']
            self.W2 = data['W2']
            self.W = data['W']
            self.U = data['U']
            self.B = data['B']

    def simulate(self, deltaT = None, pool = None, test_data = False):
        if self.parallel and pool is not None:
            #(inp, v, h, u, sh, y) = self.simulate_parallel
            results = [pool.apply(self.simulate_parallel, args=(deltaT,)) for x in range(8)]
            #Combine results and return

            return results
        else:
            return self.simulate_single(deltaT, test_data = test_data)

    def simulate_single(self, deltaT = None, test_data = False):

        inp = np.zeros((self.params.batch_size, self.params.n_input, self.T))
        v = np.zeros((self.params.batch_size, self.params.n, self.T))
        h = np.zeros((self.params.batch_size, self.params.n, self.T))
        sh = np.zeros((self.params.batch_size, self.params.n, self.T))
        y = np.zeros(self.params.batch_size, dtype = int)

        #if deltaT is provided then in blocks of deltaT we compute the counterfactual trace... the evolution without spiking.
        if deltaT is not None:
            u = np.zeros((self.params.batch_size, self.params.n, self.T))
        else:
            u = None

        vt = np.zeros(self.params.n)
        ut = np.zeros(self.params.n)
        r = np.zeros(self.params.n)

        #Generate new noise with each sim
        xi = self.params.sigma*rand.randn(self.params.batch_size, self.params.n1+1, self.T)/np.sqrt(self.params.tau)
        xi[0,:] = xi[0,:]*np.sqrt(self.params.c)
        xi[1:,:] = xi[1:,:]*np.sqrt(1-self.params.c)
        xi_l2 = self.params.sigma*rand.randn(self.params.batch_size, self.params.n2, self.T)/np.sqrt(self.params.tau)

        #Run through each item in batch
        for idx in range(self.params.batch_size):
            
            x, label = mnist_spike(lamba = 0.02, D = self.T, kernel = self.params.kernel, test_data = test_data)
            inp[idx,:,:] = x
            y[idx] = label

            #Simulate t seconds
            for i in range(self.T):
                #ut is not reset by spiking. ut is set to vt at the start of each block of deltaT
                if deltaT is not None:
                    if i%deltaT == 0:
                        ut = vt
                dv = -vt/self.params.tau + np.dot(self.U, sh[idx,:,i])
                dv[0:self.params.n1] += np.dot(self.W, (x[:,i])) + np.multiply(self.W1, xi[idx,0,i] + xi[idx,1:,i])
                dv[self.params.n1:] += np.multiply(self.W2, xi_l2[idx,:,i])
                vt = vt + self.params.dt*dv
                ut = ut + self.params.dt*dv
                #Find neurons that spike
                s = vt>self.params.mu
                #Update sh based on spiking.....
                for s_idx in np.nonzero(s)[0]:
                    convolve_online_v2(sh[idx,:,:], s_idx, i, self.params.kernel, 0)
                #Save the voltages and spikes
                h[idx,:,i] = s.astype(int)
                v[idx,:,i] = vt
                if deltaT is not None:
                    u[idx,:,i] = ut
                #Make spiking neurons refractory
                r[s] = self.Tr
                #Set the refractory neurons to v_reset
                vt[r>0] = self.params.reset
                vt[vt<self.params.reset] = self.params.reset
                ut[ut<self.params.reset] = self.params.reset
                #Decrement the refractory counters
                r[r>0] -= 1

        return (inp, v, h, u, sh, y)                                     

    def simulate_parallel(self, deltaT = None):

        print("Simulating!")

        inp = np.zeros((self.params.batch_size, self.params.n_input, self.T))
        v = np.zeros((self.params.batch_size, self.params.n, self.T))
        h = np.zeros((self.params.batch_size, self.params.n, self.T))
        sh = np.zeros((self.params.batch_size, self.params.n, self.T))
        y = np.zeros(self.params.batch_size, dtype = int)

        #if deltaT is provided then in blocks of deltaT we compute the counterfactual trace... the evolution without spiking.
        if deltaT is not None:
            u = np.zeros((self.params.batch_size, self.params.n, self.T))
        else:
            u = None

        vt = np.zeros(self.params.n)
        ut = np.zeros(self.params.n)
        r = np.zeros(self.params.n)

        #Generate new noise with each sim
        xi = self.params.sigma*rand.randn(self.params.batch_size, self.params.n1+1, self.T)/np.sqrt(self.params.tau)
        xi[0,:] = xi[0,:]*np.sqrt(self.params.c)
        xi[1:,:] = xi[1:,:]*np.sqrt(1-self.params.c)
        xi_l2 = self.params.sigma*rand.randn(self.params.batch_size, self.params.n2, self.T)/np.sqrt(self.params.tau)

        #Run a number of simulations in parallel

        #out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))

        #Run through each item in batch
        for idx in range(self.params.batch_size):
            
            x, label = mnist_spike(lamba = 0.02, D = self.T, kernel = self.params.kernel)
            inp[idx,:,:] = x
            y[idx] = label

            #Simulate t seconds
            for i in range(self.T):
                #ut is not reset by spiking. ut is set to vt at the start of each block of deltaT
                if deltaT is not None:
                    if i%deltaT == 0:
                        ut = vt
                dv = -vt/self.params.tau + np.dot(self.U, sh[idx,:,i])
                dv[0:self.params.n1] += np.dot(self.W, (x[:,i])) + np.multiply(self.W1, xi[idx,0,i] + xi[idx,1:,i])
                dv[self.params.n1:] += np.multiply(self.W2, xi_l2[idx,:,i])
                vt = vt + self.params.dt*dv
                ut = ut + self.params.dt*dv
                #Find neurons that spike
                s = vt>self.params.mu
                #Update sh based on spiking.....
                for s_idx in np.nonzero(s)[0]:
                    convolve_online_v2(sh[idx,:,:], s_idx, i, self.params.kernel, 0)
                #Save the voltages and spikes
                h[idx,:,i] = s.astype(int)
                v[idx,:,i] = vt
                if deltaT is not None:
                    u[idx,:,i] = ut
                #Make spiking neurons refractory
                r[s] = self.Tr
                #Set the refractory neurons to v_reset
                vt[r>0] = self.params.reset
                vt[vt<self.params.reset] = self.params.reset
                ut[ut<self.params.reset] = self.params.reset
                #Decrement the refractory counters
                r[r>0] -= 1

        return (inp, v, h, u, sh, y)    

    def sigma_prime(self, x):
        return (x>0).astype(float)

    def backprop(self, inp, sh, y, h):
        #Get averaged activity for each layer and input
        mean_activity = np.mean(sh, 2)
        mean_inp = np.mean(inp,2)
        hidden = mean_activity[:,0:self.params.n1]

        #Filtered spike trains
        output = mean_activity[:,self.params.n1:]
        #Number of spikes in window...
        #output = np.sum(h,2)[:,self.params.n1:]

        W = self.W
        U = self.U[self.params.n1:, 0:self.params.n1]

        #Convert to one-hot
        y_hot = np.zeros((self.params.batch_size, self.params.n2))
        for idx in range(self.params.batch_size):
            y_hot[idx,y[idx]] = 1.0

        #Backprop
        e2 = np.multiply((output - y_hot), self.sigma_prime(np.matmul(U, hidden.T)).T)
        e1 = np.multiply(np.matmul(e2, U), self.sigma_prime(np.matmul(W, mean_inp.T)).T)

        #Gradient updates
        gradU = np.matmul(e2.T, hidden)
        gradW = np.matmul(e1.T, mean_inp)
        U -= self.params.eta*gradU
        W -= self.params.eta*gradW

        loss = np.mean(np.power((y_hot - output),2)/2)
        acc = 100*np.mean(np.argmax(output,1) == y)

        return loss, acc

    def feedbackalignment(self, inp, sh, y, h):
        #Get averaged activity for each layer and input
        mean_activity = np.mean(sh, 2)
        mean_inp = np.mean(inp,2)
        hidden = mean_activity[:,0:self.params.n1]
        
        #Filtered spike trains
        output = mean_activity[:,self.params.n1:]
        #Number of spikes in window...
        #output = np.sum(h,2)[:,self.params.n1:]

        W = self.W
        U = self.U[self.params.n1:, 0:self.params.n1]
        B = self.B

        #Convert to one-hot
        y_hot = np.zeros((self.params.batch_size, self.params.n2))
        for idx in range(self.params.batch_size):
            y_hot[idx,y[idx]] = 1.0

        #Feedback alignment
        e2 = np.multiply((output - y_hot), self.sigma_prime(np.matmul(U, hidden.T)).T)
        e1 = np.multiply(np.matmul(e2, B), self.sigma_prime(np.matmul(W, mean_inp.T)).T)

        #Gradient updates
        gradU = np.matmul(e2.T, hidden)
        gradW = np.matmul(e1.T, mean_inp)
        U -= self.params.eta*gradU
        W -= self.params.eta*gradW

        loss = np.mean(np.power((y_hot - output),2)/2)
        acc = 100*np.mean(np.argmax(output,1) == y)

        #Compute metrics
        alignment = 0
        frob_err = 0
        for idx in range(self.params.batch_size):
            f1 = np.linalg.norm(np.dot(e2[idx,:], U))
            f2 = np.linalg.norm(np.dot(e2[idx,:], B))
            alignment += np.dot(e2[idx,:], np.dot(U, np.dot(B.T, e2[idx,:].T)))/f1/f2
            frob_err += np.linalg.norm(U - B, 'fro')
        alignment /= self.params.batch_size
        frob_err /= self.params.batch_size

        metrics = (alignment, frob_err)
        return loss, acc, metrics

    def rdd(self, inp, sh, y, h, v, u, deltaT):
        #Get averaged activity for each layer and input
        mean_activity = np.mean(sh, 2)
        mean_inp = np.mean(inp,2)
        hidden = mean_activity[:,0:self.params.n1]

        #Filtered spike trains
        output = mean_activity[:,self.params.n1:]
        #Number of spikes in window...
        #output = np.sum(h,2)[:,self.params.n1:]

        U = self.U[self.params.n1:, 0:self.params.n1]
        B = self.B

        #Convert to one-hot
        y_hot = np.zeros((self.params.batch_size, self.params.n2))
        for idx in range(self.params.batch_size):
            y_hot[idx,y[idx]] = 1.0

        #Backprop
        e2 = np.multiply((output - y_hot), self.sigma_prime(np.matmul(U, hidden.T)).T)

        #Compute matrix R
        #Compute the BT vector... 32*20 = 640
        output_end = sh[:,self.params.n1:].reshape((self.params.batch_size, self.params.n2, -1, deltaT))[:,:,:,-1]
        n_rdd_bins = output_end.shape[2]
        y_hot_dup = np.repeat(y_hot[...,None], n_rdd_bins, axis=2)
        loss_fine = np.sum(np.power((y_hot_dup - output_end),2)/2,1)
        
        #Compute RDD indicator function
        #Dimensions BT x n1
        #Compute z for the n1 neurons
        z = np.max(u.reshape((self.params.batch_size, self.params.n, -1, deltaT)), 3).transpose((0,2,1))[:,:,0:self.params.n1]
        almost_spike = (z < 1) & (z > 1-self.params.p)
        barely_spike = (z > 1) & (z < 1+self.params.p)
        
        n_almost_spike = np.sum(almost_spike, 1)
        n_barely_spike = np.sum(barely_spike, 1)
        
        #Take averages
        ave_loss_almost = np.zeros(n_barely_spike.shape)
        ave_loss_barely = np.zeros(n_barely_spike.shape)
        for idx_b in range(self.params.batch_size):
            for idx_n in range(self.params.n1):
                if n_almost_spike[idx_b,idx_n] > 0:
                    ave_loss_almost[idx_b,idx_n] = np.sum(np.multiply(loss_fine[idx_b,:], almost_spike[idx_b,:,idx_n]))/n_almost_spike[idx_b,idx_n]
                if n_barely_spike[idx_b,idx_n] > 0:
                    ave_loss_barely[idx_b,idx_n] = np.sum(np.multiply(loss_fine[idx_b,:], barely_spike[idx_b,:,idx_n]))/n_barely_spike[idx_b,idx_n]
        causal_effect = ave_loss_barely - ave_loss_almost

        #Feedback weight updates
        gradB = np.matmul(e2.T, (np.matmul(e2, B) - causal_effect))
        B -= self.params.eta_B*gradB

        loss = np.mean(np.power((y_hot - output),2)/2)
        acc = 100*np.mean(np.argmax(output,1) == y)

        #Compute metrics
        alignment = 0
        frob_err = 0
        for idx in range(self.params.batch_size):
            f1 = np.linalg.norm(np.dot(e2[idx,:], U))
            f2 = np.linalg.norm(np.dot(e2[idx,:], B))
            alignment += np.dot(e2[idx,:], np.dot(U, np.dot(B.T, e2[idx,:].T)))/f1/f2
            frob_err += np.linalg.norm(U - B, 'fro')
        alignment /= self.params.batch_size
        frob_err /= self.params.batch_size

        metrics = (alignment, frob_err)
        return loss, acc, metrics

    def train_BP(self):
        #Simulate a minibatch
        (inp, v, h, u, sh, y) = self.simulate()
        #Update weights
        loss, acc = self.backprop(inp, sh, y, h)
        return loss, acc

    def train_FA(self):
        #Simulate a minibatch
        (inp, v, h, u, sh, y) = self.simulate()
        #Update weights
        loss, acc, metrics = self.feedbackalignment(inp, sh, y, h)
        return loss, acc, metrics

    def train_RDD(self, deltaT):
        #Simulate a minibatch
        (inp, v, h, u, sh, y) = self.simulate(deltaT)
        #Update feedback weights
        loss, acc, metrics = self.rdd(inp, sh, y, h, v, u, deltaT)
        #Update remaining weights
        loss, acc, _ = self.feedbackalignment(inp, sh, y, h)
        return loss, acc, metrics

    def train_just_RDD(self, deltaT):
        #Simulate a minibatch
        (inp, v, h, u, sh, y) = self.simulate(deltaT)
        #Update feedback weights
        loss, acc, metrics = self.rdd(inp, sh, y, h, v, u, deltaT)
        return loss, acc, metrics

    def eval(self):
        results = self.simulate(test_data = True)
        (inp, v, h, u, sh, y) = results

        mean_activity = np.mean(sh, 2)
        mean_inp = np.mean(inp,2)
        hidden = mean_activity[:,0:self.params.n1]
        output = mean_activity[:,self.params.n1:]

        #Convert to one-hot
        y_hot = np.zeros((self.params.batch_size, self.params.n2))
        for idx in range(self.params.batch_size):
            y_hot[idx,y[idx]] = 1.0

        loss = np.mean(np.power((y_hot - output),2)/2)
        acc = 100*np.mean(np.argmax(output,1) == y)
        
        return loss, acc, results