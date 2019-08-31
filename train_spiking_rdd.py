import numpy as np
import multiprocessing as mp

from lib.lif import LIF_Recurrent, ParamsLIF_Recurrent

tau_s = 0.020
dt = 0.001
t = 1
deltaT = 20
parallel = True
n_proc = 4

t_filter = np.linspace(0, 1, 2000)
exp_filter = np.exp(-t_filter/tau_s)
exp_filter = exp_filter/np.sum(exp_filter)
ds = exp_filter[0]

params = ParamsLIF_Recurrent(exp_filter, dt = dt)
lif = LIF_Recurrent(params, t = t, parallel = parallel)

if parallel:
    pool = mp.Pool(processes = n_proc)

N = 1000 #Number of iterations
fn_out = './results/spiking_rdd_N_%d_batchsize_%d.pkl'%(N, params.batch_size)

#Test out the train_RDD params
#self = lif
#(inp, v, h, u, sh, y) = self.simulate(deltaT)
#mean_activity = np.mean(sh, 2)
#mean_inp = np.mean(inp,2)
#hidden = mean_activity[:,0:self.params.n1]
#
##Filtered spike trains
#output = mean_activity[:,self.params.n1:]
##Number of spikes in window...
##output = np.sum(h,2)[:,self.params.n1:]
#
#U = self.U[self.params.n1:, 0:self.params.n1]
#B = self.B
#
##Convert to one-hot
#y_hot = np.zeros((self.params.batch_size, self.params.n2))
#for idx in range(self.params.batch_size):
#    y_hot[idx,y[idx]] = 1.0
#
##Backprop
#e2 = np.multiply((output - y_hot), self.sigma_prime(np.matmul(U, hidden.T)).T)
#
##Compute matrix R
##Compute the BT vector... 32*20 = 640
#output_all = sh[:,self.params.n1:]
#output_split = output_all.reshape((self.params.batch_size, self.params.n2, -1, deltaT))
#output_end = output_split[:,:,:,-1]
#n_rdd_bins = output_end.shape[2]
#y_hot_dup = np.repeat(y_hot[...,None], n_rdd_bins, axis=2)
#loss_fine = np.sum(np.power((y_hot_dup - output_end),2)/2,1)
#
##Compute RDD indicator function
##Dimensions BT x n1
##Compute z for the n1 neurons
#p = 0.2
#z = np.max(u.reshape((self.params.batch_size, self.params.n, -1, deltaT)), 3).transpose((0,2,1))[:,:,0:self.params.n1]
#almost_spike = (z < 1) & (z > 1-p)
#barely_spike = (z > 1) & (z < 1+p)
#
#n_almost_spike = np.sum(almost_spike, 1)
#n_barely_spike = np.sum(barely_spike, 1)
#
##Take averages
#ave_loss_almost = np.zeros(n_barely_spike.shape)
#ave_loss_barely = np.zeros(n_barely_spike.shape)
#for idx_b in range(self.params.batch_size):
#	for idx_n in range(self.params.n1):
#		if n_almost_spike[idx_b,idx_n] > 0:
#			ave_loss_almost[idx_b,idx_n] = np.sum(np.multiply(loss_fine[idx_b,:], almost_spike[idx_b,:,idx_n]))/n_almost_spike[idx_b,idx_n]
#		if n_barely_spike[idx_b,idx_n] > 0:
#			ave_loss_barely[idx_b,idx_n] = np.sum(np.multiply(loss_fine[idx_b,:], barely_spike[idx_b,:,idx_n]))/n_barely_spike[idx_b,idx_n]
#causal_effect = ave_loss_barely - ave_loss_almost

for j in range(N):
    print("Iteration: %d"%j)
    train_loss, train_acc = lif.train_RDD(deltaT)
    print("Training loss: %f, training accuracy: %f"%(train_loss, train_acc))

#Save weights and such...
lif.save(fn_out)