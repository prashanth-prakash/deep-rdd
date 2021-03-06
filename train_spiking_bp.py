import numpy as np
import multiprocessing as mp
import pickle
from lib.lif import LIF_Recurrent, ParamsLIF_Recurrent
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random


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

M = 1      #Number of repetitions
N = 3000    #Number of iterations
batch_size = 32

model_out = './results/spiking_bp_M_%d_N_%d_batchsize_%d.pkl'%(M, N, batch_size)
results_out = './results/spiking_bp_M_%d_N_%d_batchsize_%d_results.pkl'%(M, N, batch_size)


val_eta= np.logspace(-1,0,6)
val_sigma = np.logspace(-1,2,6)
losses = np.zeros((M,N))
accs = np.zeros((M,N))
alignments = np.zeros((M,N))
frob_errs = np.zeros((M,N))

print("Validating over learnining:",val_eta,"and\n standard deviation of noise:",val_sigma)
for i in range(M):
    print("Repeat: %d/%d"%(i+1,M))
    params = ParamsLIF_Recurrent(exp_filter,eta = val_eta[4], dt = dt, batch_size = batch_size, 
                                         sigma = val_sigma[4])
    lif = LIF_Recurrent(params, t = t, parallel = parallel)
    for j in range(N):
        print("Iteration: %d"%j)
        train_loss, train_acc = lif.train_BP()
        print("Training loss: %f, training accuracy: %f"%(train_loss, train_acc))
        losses[i,j] = train_loss
        accs[i,j] = train_acc

            #Save weights and results with each run...
        lif.save(model_out)
        to_save = {
                'losses': losses,
                'accs': accs,
                'alignment': alignments,
                'frob_errs': frob_errs
        }
        pickle.dump(to_save, open(results_out, "wb"))