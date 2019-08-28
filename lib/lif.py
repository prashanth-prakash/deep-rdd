#Simulate LIF neurons
import numpy as np
import numpy.random as rand
import scipy

import cPickle, gzip
import random
from scipy.integrate import quadrature, quad

# Load the dataset
f = gzip.open('./mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def mnist_spike(lamba, D, kernel):
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
        
def activation(u,xw,std,gl,theta):
    #scipy quadrature
    #inpstd = std*w
    y_th = np.divide(theta-xw,std)
    y_r = np.divide(xw,std)
    first = np.exp(-u**2+ 2*y_th*u)
    second = np.exp(-u**2+ 2*y_r*u)
    integral = 1/u*(first-second)

#    mu=np.diff(mu)
    return integral

def integrate(xw,std,gl,theta):
    return quad(activation,0,np.inf,args=(xw, std,gl,theta))

def gradient(mu):
    return np.diff(mu)

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

def firingrate_LIF(params, W, S):
    #Computes the firing rate as a function of weights W and mean inputs S
    tau = params.tau
    Vth = params.mu
    Vr = params.reset
    Vrest = params.reset
    dt = params.dt
    sigma_xi = params.sigma*dt
    I = np.dot(W, S)
    sigma = sigma_xi*np.sqrt(np.sum(W**2))
    Yth = (Vth - Vrest - I)/sigma
    Yr = (Vr - Vrest - I)/sigma
    f = lambda u: (1/u)*np.exp(-u**2)*(np.exp(2*Yth*u)-np.exp(2*Yr*u))
    quad = scipy.integrate.quad(f, 0, np.inf)[0]
    return 1/(tau*quad)

class ParamsLIF_Recurrent(object):    
    def __init__(self, kernel, dt = 0.001, tr = 0.003, mu = 1, reset = 0, xsigma = 1, n1 = 100, n2 = 10, tau = 1,
        c = 0.99, sigma = 20, sigma1 = 10, sigma2 = 2, batch_size = 32, n_input = 784, eta = 1e-1):

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

class LIF_Recurrent(object):

    def __init__(self, params, t = 1):
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

    def simulate(self, deltaT = None):

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

    def backprop(self, inp, sh, y):
        #Get averaged activity for each layer and input
        mean_activity = np.mean(sh, 2)
        mean_inp = np.mean(inp,2)
        hidden = mean_activity[:,0:self.params.n1]
        output = mean_activity[:,self.params.n1:]
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

    def train_BP(self):
        #Simulate a minibatch
        (inp, v, h, u, sh, y) = self.simulate()
        #Update weights
        loss, acc = self.backprop(inp, sh, y)
        return loss, acc

    def feedbackalignment(self, inp, sh, y):
        #Get averaged activity for each layer and input
        mean_activity = np.mean(sh, 2)
        mean_inp = np.mean(inp,2)
        hidden = mean_activity[:,0:self.params.n1]
        output = mean_activity[:,self.params.n1:]
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

        return loss, acc

    def train_FA(self):
        #Simulate a minibatch
        (inp, v, h, u, sh, y) = self.simulate()
        #Update weights
        loss, acc = self.feedbackalignment(inp, sh, y)
        return loss, acc

    def eval(self):
        raise NotImplementedError