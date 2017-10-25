import numpy as np
from lib.lif import LIF, ParamsLIF

q = 3              #Dimension of learnt vector
dt = 0.001         #Simulation timestep
DeltaT = 20        #Number of timebins over which learning rule is applied
tsim = 60             #Total simulation time
T = (tsim/dt)/DeltaT #Number of learning blocks
N = 50             #Number of repeated simulations
c = 0.5           #Correlation coefficient
x = 0              #Input
n = 2              #Number of neurons
sigma = 10         #Their noise level
mu = 1             #Threshold
tau = 1            #Neuron timescale
eta = .2           #Learning rate
p = 0.1           #Learning window
tau_s = 0.020      #Output filter timescale
wmax = 20          #Maximum unit weight

#w1 = 2
#w2 = 6

B1 = 3
B2 = 300
x = 1
y = 0.12
z = -0.3

cost_fun = lambda s1, s2: (B1*s1-x)**2 + (z+B2*s2 - B2*(B1*s1-y)**2)**2

#perturb_rate = 0.01 # Proportion of points that are perturbations
#                    # 1% = 10 Hz. Only half of these are spikes, so the injected noise rate is 5Hz

mvals = [0.0025, 0.005, 0.01, 0.015]
M = len(mvals)

# Filename for results
fn_out = './sweeps/learningbeta_maxv_fixedx_sweepw_banana_perturbation.npz'

params_lif = ParamsLIF(sigma = sigma, tau = tau, mu = mu, c = c)
lif = LIF(params_lif, t = tsim)
lif.x = 0

t_filter = np.linspace(0, 1, 2000)
exp_filter = np.exp(-t_filter/tau_s)
exp_filter = exp_filter/np.sum(exp_filter)
ds = exp_filter[0]

wvals = np.linspace(2, wmax, Wn)
beta_rd = np.zeros((Wn, Wn, n, T))
beta_rd_true = np.zeros((Wn, Wn, n, T))
beta_fd_true = np.zeros((Wn, Wn, n, T))
beta_sp = np.zeros((Wn, Wn, n, T, M))

for i, w0 in enumerate(wvals):
    print("W0=%d"%w0)
    for j, w1 in enumerate(wvals):
        print("W1=%d"%w1)

        #init weights
        lif.W = np.array([w0, w1])

        count = 0
        print("N=%d"%idx)
        #Simulate LIF
        (v_raw, h_raw, _, _) = lif.simulate()
        s1 = np.convolve(h_raw[0,:], exp_filter)[0:h_raw.shape[1]]
        s2 = np.convolve(h_raw[1,:], exp_filter)[0:h_raw.shape[1]]
        
        V = np.zeros((n, q))
        
        abvthr = np.zeros(n)
        blothr = np.zeros(n)
        
        cost_raw = cost_fun(s1, s2)
        #Break the simulation and voltage into blocks
        nB = h_raw.shape[1]/DeltaT
        hm = h_raw.reshape((n, nB, DeltaT))
        vm = v_raw.reshape((n, nB, DeltaT))
        
        v = np.max(vm, 2)
        h = np.max(hm, 2)
        cost_r = cost_raw.reshape((nB, DeltaT))
        cost = np.squeeze(cost_r[:,-1])
    

        ptb = 2*(np.random.rand(*h.shape) < 0.5)-1
        #Create a perturbed set of trains
        for idx2, perturb_rate in enumerate(mvals):
            dU = np.zeros(U.shape[0:2])
            qtb = np.random.rand(*h_raw.shape) < perturb_rate
            h_perturb = h_raw.copy()
            h_perturb[qtb == True] = ptb[qtb == True]
            s1_perturb = np.convolve(h_perturb[0,:], exp_filter)[0:h.shape[1]]
            s2_perturb = np.convolve(h_perturb[1,:], exp_filter)[0:h.shape[1]]
            cost_perturbed = cost_fun(s1_perturb, s2_perturb)
            for t in range(v.shape[1]):
                for k in range(n):
                    #If this timebin is a perturbation time then update U
                    if qtb[k,t]:
                        dU[k,:] = (np.dot(U[k,:,idx2], s_lsm[:,t])-ptb[k,t]*cost_perturbed[t])*s_lsm[:,t]
                        U[k,:,idx2] = U[k,:,idx2] - eta*dU[k,:]
            beta_sp[i,j,:,idx,idx2] = np.mean(np.dot(U[:,:,idx2], s_lsm[:,-100:]),1)

        #Then just repeat the learning rule as before
        dV = np.zeros(V.shape)
        bt = [False, False]
        for t in range(nB):
            for k in range(n):
                if (v[k,t] < mu):
                    if k == 0:
                        c1_blo_1 = np.hstack((c1_blo_1, cost[t]))
                    else:
                        c2_blo_1 = np.hstack((c2_blo_1, cost[t]))
                if (v[k,t] >= mu):
                    if k == 0:
                        c1_abv_1 = np.hstack((c1_abv_1, cost[t]))
                    else:
                        c2_abv_1 = np.hstack((c2_abv_1, cost[t]))
    
                if (v[k,t] > mu - p) & (v[k,t] < mu):
                    if k == 0:
                        c1_blo_p = np.hstack((c1_blo_p, cost[t]))
                    else:
                        c2_blo_p = np.hstack((c2_blo_p, cost[t]))
                    blothr[k] += 1
                    if bt[k] == False:
                        #ahat = np.array([1, 0, -(v[k,t]-mu)])
                        ahat = np.array([1, 0, 0])
                        dV[k,:] += (np.dot(V[k,:], ahat)+cost[t])*ahat                    
                        bt[k] = True
                elif (v[k,t] < mu + p) & (v[k,t] >= mu):
                    if k == 0:
                        c1_abv_p = np.hstack((c1_abv_p, cost[t]))
                    else:
                        c2_abv_p = np.hstack((c2_abv_p, cost[t]))
                    abvthr[k] += 1
                    #Only do the update when firing...
                    if bt[k] == True:
                        #ahat = np.array([1, (v[k,t]-mu), 0])
                        ahat = np.array([1, 0, 0])
                        dV[k,:] += (np.dot(V[k,:], ahat)-cost[t])*ahat                                        
                        count += 1
                        V[k,:] = V[k,:] - eta*dV[k,:]#*count/(count+1)
                        dV[k,:] = np.zeros((1,q))
                        bt[k] = False
                    
                beta_rd[i,j,k,t] = V[k,0]
            beta_rd_true[i,j,0,t] = np.mean(c1_abv_p)-np.mean(c1_blo_p)
            beta_rd_true[i,j,1,t] = np.mean(c2_abv_p)-np.mean(c2_blo_p)
            beta_fd_true[i,j,0,t] = np.mean(c1_abv_1)-np.mean(c1_blo_1)
            beta_fd_true[i,j,1,t] = np.mean(c2_abv_1)-np.mean(c2_blo_1) 

#Save the results
np.savez(fn_out, wvals = wvals, beta_rd = beta_rd, beta_rd_true = beta_rd_true, beta_fd_true = beta_fd_true,\
 beta_sp = beta_sp)