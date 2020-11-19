from brian2 import *
import numpy as np

rcParams.update({'font.size': 20})

def ISI_CV(trains, t_min, t_max):
    # number of neurons
    num_neu = len(trains)
    # ISI mean values for each neuron
    isi_mu1 = () * second
    isi_mu2 = () * second
    # ISI standard deviation for each neuron
    isi_std1 = () * second
    isi_std2 = () * second
    for idx in range(num_neu):
        # calc. ISIs for times between t_min and t_max
        train = diff(trains[idx][argmax(trains[idx] > t_min * second):argmax(trains[idx] > t_max * second)])
        if len(train) > 1:
            if idx % 100 < 25 and idx < 2500:
                # calc. mean
                isi_mu1 = append(isi_mu1, mean(train))
                # calc. standard deviation
                isi_std1 = append(isi_std1, std(train))
            else:
                # calc. mean
                isi_mu2 = append(isi_mu2, mean(train))
                # calc. standard deviation
                isi_std2 = append(isi_std2, std(train))

    # calc. ISI CV for every neuron
    CV1 = isi_std1 / isi_mu1
    CV2 = isi_std2 / isi_mu2
    # calc. mean of all ISI CVs
    cv1 = mean(CV1)
    cv2 = mean(CV2)

    # plot histograms
    gca().set_ylim([0, 105])
    hist([CV1, CV2], bins=50, range=[0, 2],
             weights=[np.ones(len(CV1)) / len(CV1) * 100., np.ones(len(CV2)) / len(CV2) * 100.],
             label=['assembly', 'control'])  # arguments are passed to np.histogram
    xlabel('ISI CV')
    ylabel('Percent [%]')
    # title('ISI CV distribution ')
    legend()
    show()
    return cv1, cv2


def spiking_corr(trains, t_min, t_max, chosen_neu):
    print('please wait...')
    # symmetric bi-exponential kernel
    TK = np.arange(-1, 1, 0.001)
    K = ((1 / (50 * ms)) * exp(-np.abs(TK * 1000) / (50))) - ((1 / (200 * ms)) * exp(-np.abs(TK * 1000) / (200)))

    # array of spiking correlations in assembly and control group
    X1 = ()
    X2 = ()

    for i in chosen_neu:
        for j in chosen_neu:
            if i != j:
                # calc. only for times between t_min and t_max
                trains_short_i = trains[i][argmax(trains[i] >= t_min * second):argmax(trains[i] >= t_max * second)]
                trains_short_j = trains[j][argmax(trains[j] >= t_min * second):argmax(trains[j] >= t_max * second)]
                T = np.arange(t_min, t_max, 0.001)
                spikes_i = np.zeros(len(T))
                for k in range(len(trains_short_i)):
                    spikes_i[int((trains_short_i[k] / second - t_min) * 1000)] = 1
                spikes_j = np.zeros(len(T))
                for k in range(len(trains_short_j)):
                    spikes_j[int((trains_short_j[k] / second - t_min) * 1000)] = 1
                # convolutions with kernel
                F_i = convolve(spikes_i, K)
                F_j = convolve(spikes_j, K)
                # check if in assembly
                if i % 100 < 25 and j % 100 < 25 and i < 2500 and j < 2500:
                    X1 = append(X1, corrcoef(F_i, F_j)[0, 1])
                elif (i % 100 >= 25 or i > 2500) and (j % 100 >= 25 or j > 2500):
                    X2 = append(X2, corrcoef(F_i, F_j)[0, 1])

    # plot histograms
    figure()
    gca().set_ylim([0, 105])
    hist([X1, X2], bins=20, range=[-0.3, 1],
             weights=[np.ones(len(X1)) / len(X1) * 100., np.ones(len(X2)) / len(X2) * 100.],
             label=['assembly', 'control'])
    xlabel('spiking correlation')
    ylabel('Percent [%]')
    # title('spiking correlation distribution')
    legend()
    show()

    X1 = X1[where(logical_not(isnan(X1)))]
    X2 = X2[where(logical_not(isnan(X2)))]
    return mean(X1), mean(X2)


# ###########################################
# defining the simulation episodes
# ###########################################
sim_episodes = {'init': 1,
                'no_learning': 5,
                'set_learning': 14,
                'asyn_irg': 5,
                'pre_assembly': 1,
                'assembly': 5,
                'wait_rebalance': 14,
                'post_rebalance': 6}

# just adding times up
sim_times = {'init': 1,
             'no_learning': 6,
             'set_learning': 20,
             'asyn_irg': 25,
             'pre_assembly': 26,
             'assembly': 31,
             'wait_rebalance': 45,
             'post_rebalance': 50}

# ###########################################
# Defining network model parameters
# ###########################################

NE = 8000          # Number of excitatory cells
NI = int(NE/4)         # Number of inhibitory cells

tau_ampa = 5.0*ms   # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms  # GABAergic synaptic time constant
epsilon = 0.02      # Sparseness of synaptic connections

tau_stdp = 20*ms    # STDP time constant

# defaultclock.dt = 0.2*ms

# ###########################################
# Neuron model
# ###########################################

gl = 10.0*nsiemens   # Leak conductance
el = -60*mV          # Resting potential
er = -80*mV          # Inhibitory reversal potential
vt = -50.*mV         # Spiking threshold
memc = 200.0*pfarad  # Membrane capacitance
bgcurrent = 200*pA   # External current

eqs_neurons='''
dv/dt = I_n/memc : volt (unless refractory)
I_n = (-gl*(v-el)-(I_e+I_i)+bgcurrent) : amp
I_i = g_gaba*(v-er) : amp
I_e = g_ampa*v : amp
dg_ampa/dt = -g_ampa/tau_ampa : siemens
dg_gaba/dt = -g_gaba/tau_gaba : siemens
'''

# ###########################################
# Initialize neuron group
# ###########################################

neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                      reset='v=el', refractory=5*ms, method='euler')
neurons.v = np.random.randint(-60,-50,10000)*mV

Pe = neurons[:NE]
Pi = neurons[NE:]

# ###########################################
# Connecting the network
# ###########################################

con_e = Synapses(Pe, neurons, on_pre='g_ampa += 0.3*nS')
con_e.connect(p=epsilon)
con_ii = Synapses(Pi, Pi, on_pre='g_gaba += 3*nS')
con_ii.connect(p=epsilon)

# ###########################################
# Inhibitory Plasticity
# ###########################################

eqs_stdp_inhib = '''
w : 1
dApre/dt=-Apre/tau_stdp : 1 (event-driven)
dApost/dt=-Apost/tau_stdp : 1 (event-driven)
'''
alpha = 3*Hz*tau_stdp*2  # Target rate parameter
gmax = 100               # Maximum inhibitory weight

con_ie = Synapses(Pi, Pe, model=eqs_stdp_inhib,
                  on_pre='''Apre += 1.
                         w = clip(w+(Apost-alpha)*eta, 0, gmax)
                         g_gaba += w*nS''',
                  on_post='''Apost += 1.
                          w = clip(w+Apre*eta, 0, gmax)
                       ''')
con_ie.connect(p=epsilon)
con_ie.w = 1e-10

# ###########################################
# Setting up monitors
# ###########################################

I_rec_e = StateMonitor(Pe[0:1], ('v', 'I_i', 'I_e', 'I_n'), record=[0])
S_rec_e = SpikeMonitor(Pe[0:1])
I_rec_i = StateMonitor(Pi[0:1], ('v', 'I_i', 'I_e', 'I_n'), record=[0])
S_rec_i = SpikeMonitor(Pi[0:1])
SM_e = SpikeMonitor(Pe)
SM_i = SpikeMonitor(Pi)

# ###########################################
# Run without plasticity
# ###########################################
eta = 0          # Learning rate

# free run for the system to settle down (initialize)
run(sim_episodes['init']*second,report='text')

SM_n_time = SpikeMonitor(neurons, record=False)
run(sim_episodes['no_learning']*second,report='text')
count_0_1 = np.array(SM_n_time.count).reshape((100,100))
heat_no_learning = count_0_1 / sim_episodes['no_learning']

# ###########################################
# Run with plasticity
# ###########################################
eta = 1e-2          # Learning rate
run(sim_episodes['set_learning']*second, report='text')
count_0_2 = np.array(SM_n_time.count).reshape((100,100))

# ###########################################
# Run after plasticity
# ###########################################
run(sim_episodes['asyn_irg']*second, report='text')
count_0_3 = np.array(SM_n_time.count).reshape((100,100))
heat_asyn_irg = (count_0_3 - count_0_2) / sim_episodes['asyn_irg']

# ###########################################
# turn off plasticity
# ###########################################
eta = 0          # Learning rate
run(sim_episodes['pre_assembly']*second, report='text')
count_0_4 = np.array(SM_n_time.count).reshape((100,100))
heat_reset_learning = (count_0_4 - count_0_3) / sim_episodes['pre_assembly']

# ###########################################
# change the connection probability
# ###########################################
p_rc = 0.1

# adding new connections between 625 randomly chosen neurons
con_e.connect(condition='i%100<25 and j%100<25 and i<2500 and j<2500 and i!=j',
              p=p_rc)

run(sim_episodes['assembly']*second,report='text')
count_0_5 = np.array(SM_n_time.count).reshape((100,100))
heat_change_connect = (count_0_5 - count_0_4) / sim_episodes['assembly']

# ###########################################
# run after change in connection probability
# ###########################################
eta = 1e-2

run(sim_episodes['wait_rebalance']*second,report='text')
count_0_6 = np.array(SM_n_time.count).reshape((100,100))

# ###########################################
# asymmetry returned
# ###########################################
run(sim_episodes['post_rebalance']*second,report='text')
count_0_7 = np.array(SM_n_time.count).reshape((100,100))
heat_post_rebalance = (count_0_7 - count_0_6) / sim_episodes['post_rebalance']

print(defaultclock.dt)

###########################################
#Make plots
###########################################
imshow(heat_no_learning, origin='lower', cmap='rainbow')
# title('without inhibitory plasticity')
yticks([])
xticks([])
clim(0, 200)
cbar = colorbar()
cbar.set_label('firing rate [Hz]')
show()

imshow(heat_asyn_irg, origin='lower', cmap='rainbow')
# title('after inhibitory plasticity')
clim(0, 200)
cbar = colorbar()
yticks([])
xticks([])
cbar.set_label('firing rate [Hz]')
show()

imshow(heat_reset_learning, origin='lower', cmap='rainbow')
# title('turning off plasticity')
clim(0, 200)
cbar = colorbar()
yticks([])
xticks([])
cbar.set_label('firing rate [Hz]')
show()

imshow(heat_change_connect, origin='lower', cmap='rainbow')
# title('immediately after increasing connection probability')
clim(0, 200)
cbar = colorbar()
yticks([])
xticks([])
cbar.set_label('firing rate [Hz]')
show()

imshow(heat_post_rebalance, origin='lower', cmap='rainbow')
# title('asymmetry returned after increasing connection probability')
clim(0, 200)
cbar = colorbar()
yticks([])
xticks([])
cbar.set_label('firing rate [Hz]')
show()

i, t = SM_e.it
# plotting before and after STDP
subplot(211)
plot(t/ms, i, 'k.', ms=0.25)
title('synchronous regular')
xlabel('')
yticks([])
xlim((sim_times['no_learning']-0.2)*1e3, sim_times['no_learning']*1e3)
subplot(212)
plot(t/ms, i, 'k.', ms=0.25)
xlabel('time (ms)')
yticks([])
title('asynchronous irregular')
xlim((sim_times['asyn_irg']-0.2)*1e3, sim_times['asyn_irg']*1e3)
show()

plot(t/ms, i, 'k.', ms=0.25)
xlabel('time (ms)')
yticks([])
xlim((sim_times['assembly']-0.2)*1e3, sim_times['assembly']*1e3)
# title('change in the connection probability')
show()

subplot(211)
# plot(S_rec_e.t/ms, S_rec_e.i, 'o')
plot(I_rec_e.t/ms, I_rec_e.v[0]/mV)
title('membrane voltage')
ylabel('voltage (mV)')
# yticks([])
subplot(212)
plot(I_rec_e.t/ms, I_rec_e.I_n[0]/pA, label='net_cur')
plot(I_rec_e.t/ms, I_rec_e.I_i[0]/pA, label='inh_cur')
plot(I_rec_e.t/ms, I_rec_e.I_e[0]/pA, label='exc_cur')
ylabel('current (pA)')
xlabel('time (ms)')
title('membrane currents')
legend(loc='lower right')
show()

plot(t/ms, i, 'k.', ms=0.25)
xlabel('time (ms)')
yticks([])
show()


trains = SM_e.spike_trains()

#ISI CV before STDP
cv1_A, cv2_A = ISI_CV(trains, sim_times['init'], sim_times['no_learning'])
#ISI CV when balanced state is reached
cv1_B, cv2_B = ISI_CV(trains, sim_times['set_learning'], sim_times['asyn_irg'])
#assembly
cv1_C, cv2_C = ISI_CV(trains, sim_times['pre_assembly'], sim_times['assembly'])
#after rebalancing
cv1_D, cv2_D = ISI_CV(trains, sim_times['wait_rebalance'], sim_times['post_rebalance'])

rand_set_exc = np.random.randint(0, 8000, size=300)

#before STDP
X1_A, X2_A = spiking_corr(trains, sim_times['init'], sim_times['no_learning'], rand_set_exc)
#balanced
X1_B, X2_B = spiking_corr(trains, sim_times['set_learning'], sim_times['asyn_irg'], rand_set_exc)
#assembly
X1_C, X2_C = spiking_corr(trains, sim_times['pre_assembly'], sim_times['assembly'], rand_set_exc)
#after rebalancing
X1_D, X2_D = spiking_corr(trains, sim_times['wait_rebalance'], sim_times['post_rebalance'], rand_set_exc)

