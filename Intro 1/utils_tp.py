import numpy as np
import matplotlib.pyplot as plt

def synthetise_signal(x, N, sigma2):
    """
    - x, the pattern,
    - N the number of events,
    - sigma2, the noise variance, (the square of standard deviation)
    - T, the signal duration
    - Fe, the sampling frequency.
    """

    Fe = 8192.  # Sampling frequency
    Nx = 1024  # number of sample
    T = 10
    
    if N * x.shape[0] > Fe * T / 4:
        raise Exception('The number of events is too high')

    P     = np.zeros(N)
    z_pur = 10

    # How it works :
    #   Generate random position for pattern, convoluate, and test whether there is no overlapping.
    #   500 trials so far.

    n = 0 # Number of trials 
    while np.max(z_pur) > 1 and n < 500:

        for i in np.arange(0, N):
            P[i] = np.random.randint(1, T*Fe)
        P = np.sort(P)

        z_pur = np.zeros(T*int(Fe))
        z_pur[P.astype(int)] = 1.0

        # SECURITY TO PREVENT CLOSE EVENTS
        for i in np.arange(0, N-1):
            if P[i+1] - P[i] < 1024:
                z_pur = np.ones(T*int(Fe))

        z_pur = np.convolve(z_pur, x, 'same')

        # increase the number of trial
        n=n+1

    bruit =  np.random.normal(0, np.sqrt(sigma2), int(Fe)*T);

    z = z_pur + bruit ;

    time = np.arange(0, T, 1 / Fe)

    # Plot stuff
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 4))

    ax1.plot(time, z)
    #ax1.plot(time, z_pur)

    ax1.set_title('Synthetised noisy signal')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    ax2.plot(time, z_pur)

    ax2.set_title('pure signal')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')

    return (z, P, time, z_pur)



def filtering(z, patern):

    y = np.convolve(z, patern[::-1], 'same')
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    #Plots stuff

    Fe = 8192.  # Sampling frequency
    Nx = 1024  # number of sample
    T = 10
    
    t_patern = np.arange(-Nx / 2, Nx / 2) / Fe
    t = np.arange(0, T, 1 / Fe)

    ax1.plot(t, y)
    #ax1.plot(time, z_pur)

    ax1.set_title('Output of filter')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    ax2.plot(t_patern, patern)

    ax2.set_title('Patern')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')   

    return y



def detect_events(y, treshold, Ptrue):

    lagmin = 10
    
    d = y > treshold

    # Equivalent of find in matlab
    def indices(a, func):
        return [i for (i, val) in enumerate(a) if func(val)]

    inds = indices(d, lambda x: x == True)

    buf = []
    positions = []
    for i in inds:
        if len(buf) == 0:
            buf.append(i)
        else:
            if i <= max(buf) + lagmin:
                buf.append(i)
            
            else:
                positions.append( int(np.median(np.array(buf))) )
                buf = [i]

    positions.append( int(np.median(np.array(buf))) )

    positions = np.array(positions)


    # Plot stuff

    Fe = 8192.  # Sampling frequency
    Nx = 1024  # number of sample
    T = 10
    
    t = np.arange(0, T, 1 / Fe)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

    ax1.plot(t, y)
    ax1.axhline(treshold)

    ax1.set_title('Output of filter')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    markerline, stemlines, baselin = ax2.stem(t[Ptrue.astype(int)], 1.2 + np.zeros(Ptrue.shape), linestyle='steps--', label='True events')
    plt.setp(stemlines, linewidth=2, color='b')
    plt.setp(baselin, linewidth=0.1, color='w')


    markerline2, stemlines2, baselin2 = ax2.stem(t[positions.astype(int)], 1 + np.zeros(positions.shape), linestyle='-.', color='r', label='Detected events')
    plt.setp(stemlines2, linewidth=2, color='red')
    plt.setp(markerline2, color='red')
    plt.setp(baselin2, linewidth=0.1, color='w')

    ax2.axis([np.min(t), np.max(t), -0.1, 1.8])

    ax2.set_title('Detection')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')

    ax2.legend()

    return np.array(positions)



def autocorr(x):
    gamma = np.correlate(x, x, mode='full') / (np.linalg.norm(x) * np.linalg.norm(x))

    Fe = 8192.  # Sampling frequency
    Nx = 1024  # number of sample
    
    t_patern = np.arange(-Nx , Nx-1) / Fe
    f, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t_patern, gamma)
    ax.set_title('Autocorrelation function')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')




def crossCorr(x1, x2):
    gamma = np.correlate(x1, x2, mode='full') / (np.linalg.norm(x1) * np.linalg.norm(x2))
    #gamma = plt.xcorr(x1, x2, normed=True)

    Fe = 8192.  # Sampling frequency
    Nx = 1024  # number of sample
    
    t_patern = np.arange(-Nx , Nx-1) / Fe
    f, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t_patern, gamma)
    ax.set_title('Cross-correlation function')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def motif_bip(freq,t) :
    bip = np.zeros(len(t))
    bip[257:764] = (1+np.cos(2*np.pi*8192/512*t[257:764]))/2 * np.sin(2*np.pi*freq*t[257:764]) 
    fig=plt.figure()
    plt.plot(t,bip)
    return bip

