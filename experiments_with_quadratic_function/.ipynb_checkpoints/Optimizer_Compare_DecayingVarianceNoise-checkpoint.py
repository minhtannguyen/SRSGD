# -*- coding: utf-8 -*-
"""
Compare different optimization algorithms with exact gradient
1. GD
2. GD + Constant momentum
3. GD + Constant momentum (Pytorch)
4. GD + Constant momentum (Sutskever)
5. Nesterov Accelerated Gradient
6. Nesterov Accelerated Gradient (Pytorch)
7. Nesterov Accelerated Gradient (Sutskever)
8. Nesterov Accelerated Gradient + Scheduled Restart
9. Nesterov Accelerated Gradient + Scheduled Restart (Pytorch)
10. Nesterov Accelerated Gradient + Scheduled Restart (Sutskever)
11. Nesterov Accelerated Gradient + Adaptive Restart
12. Nesterov Accelerated Gradient + Adaptive Restart (Pytorch)
13. Nesterov Accelerated Gradient + Adaptive Restart (Sutskever)
14. Nesterov Accelerated Gradient + Adaptive Restart + Laplacian Smoothing
15. Nesterov Accelerated Gradient + Adaptive Restart + Laplacian Smoothing (Pytorch)
16. Nesterov Accelerated Gradient + Adaptive Restart + Laplacian Smoothing (Sutskever)
"""
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------------------------------------
# Define the function
# Here we consider the convex quadratic function
# f(x) = 0.5*x*P*x - b*x
# nabla f(x) = P*x - b = 0 ==> opt = inv(P)*b
#------------------------------------------------------------------------------
n = 1000
A = np.zeros((n, n))
for i in range(1, n-1):
    A[i, i+1] = 1.
    A[i, i-1] = 1.
A[0, 1] = 1.
A[n-1, n-2] = 1.
P = 2.*np.eye(n) - A
b = np.zeros(n)
b[0] = 1
opt = np.dot(np.linalg.pinv(P), b)


def path(x):
    return 0.5*np.dot(x, np.dot(P, x)) - np.dot(x, b)

def pathgrad(x, iter_count):
    return np.dot(P, x) - b

def noisygrad(x, iter_count):
    return np.dot(P, x) - b + np.random.normal(0, 0.1/(iter_count/100.+1.), (n)) # Small noise will destroy NAG + restarting, e.g. 0.005
    #return np.dot(P, x) - b + np.random.normal(0, 0.01, (n))

def LS_noisygrad(x, iter_count):
    vec = np.dot(P, x) - b + np.random.normal(0, 0.1/(iter_count/100.+1.), (n))
    #vec = np.dot(P, x) - b + np.random.normal(0, 0.01, (n))
    
    # Perform Laplacian Smoothing
    ndim = len(vec)
    vec_LS = np.zeros(shape=(1, ndim))
    order = 1
    
    if order >= 1:
        Mat = np.zeros(shape=(order, 2*order+1))
        Mat[0, order-1] = 1.; Mat[0, order] = -2.; Mat[0, order+1] = 1.
        
        for i in range(1, order):
            Mat[i, order-i-1] = 1.; Mat[i, order+i+1] = 1.
            Mat[i, order] = Mat[i-1, order-1] - 2*Mat[i-1, order] + Mat[i-1, order+1]
            
            Mat[i, order-i] = -2*Mat[i-1, order-i] + Mat[i-1, order-i+1]
            Mat[i, order+i] = Mat[i, order-i]
            
            for j in range(0, i-1):
                Mat[i, order-j-1] = Mat[i-1, order-j-2] - 2*Mat[i-1, order-j-1] + Mat[i-1, order-j]
                Mat[i, order+j+1] = Mat[i, order-j-1]
        
        for i in range(order+1):
            vec_LS[0, i] = Mat[-1, order-i]
        
        for i in range(order):
            vec_LS[0, -1-i] = Mat[-1, order-i-1]
    
    sigma=10. #1. #100. # For high dimensional problem reduce sigma
    if order >= 1:
        vec = np.squeeze(np.real(np.fft.ifft(np.fft.fft(vec)/(1+(-1)**order*sigma*np.fft.fft(vec_LS)))))
    
    return vec


#------------------------------------------------------------------------------
# Optimization algorithms
#------------------------------------------------------------------------------
def gd(x0, gradient, smoothness=1., n_iterations=100):
    '''
    Gradient descent for a smooth function with Lipschitz constant = smoothness
    The optimal step size: 1./smoothness
    '''
    x = x0
    xs = [x0]
    for t in range(0, n_iterations):
        x = x - (1./smoothness)*gradient(x, t)
        xs.append(x)
    return xs

def mgd(x0, gradient, smoothness=1., n_iterations=100):
    '''
    Gradient descent with constant momentum for a smooth function with Lipschitz function
    with Lipschitz constant = smoothness
    '''
    x = x0
    y = x0
    xs = [x0]
    for t in range(1, n_iterations+1):
        #x2 = y - (1./smoothness/(10**(t/10000)))*gradient(y, t)
        x2 = y - (1./smoothness)*gradient(y, t)
        y2 = x2 + 0.9*(x2-x)
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag(x0, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent for smooth function with Lipschitz constant = smoothness
    '''
    x = x0
    y = x0
    xs = [x0]
    for t in range(1, n_iterations+1):
        #x2 = y - (1./smoothness/(10**(t/10000)))*gradient(y, t)
        x2 = y - (1./smoothness)*gradient(y, t)
        y2 = x2 + (t-1.)/(t+2.)*(x2-x)
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag_s(x0, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent for smooth function with Lipschitz constant = smoothness
    Sutskever's version
    '''
    x = x0
    y = x0
    xs = [x0]
    for t in range(1, n_iterations+1):
        x2 = (t-1.)/(t+2.)*x + (1./smoothness)*gradient(y, t)
        y2 = y - x2
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag_p(x0, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent for smooth function with Lipschitz constant = smoothness
    Pytorch version
    '''
    x = x0
    y = x0
    xs = [x0]
    for t in range(1, n_iterations+1):
        x2 = (t-1.)/(t+2.)*x + gradient(y, t)
        y2 = y - (1./smoothness)*x2
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag_adaptive_restarting(x0, func, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent with adaptive restarting for smooth
    function with Lipschitz constant = smoothness.
    '''
    x = x0
    y = x0
    xs = [x0]
    k = 1
    for t in range(1, n_iterations+1):
        #x2 = y - (1./smoothness/(10**(t/10000)))*gradient(y, t)
        x2 = y - (1./smoothness)*gradient(y, t)
        y2 = x2 + (k-1.)/(k+2.)*(x2-x)
        
        # Restarting
        if func(x2) > func(x):
            k = 1
        else:
            k += 1
        
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag_s_adaptive_restarting(x0, func, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent with adaptive restarting for smooth function with Lipschitz constant = smoothness
    Sutskever's version
    '''
    x = x0
    y = x0
    xs = [x0]
    k = 1
    for t in range(1, n_iterations+1):
        x2 = (k-1.)/(k+2.)*x + (1./smoothness)*gradient(y, t)
        y2 = y - x2
        
        # Restarting
        if func(x2) > func(x):
            k = 1
        else:
            k += 1
        
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag_p_adaptive_restarting(x0, func, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent with adaptive restarting for smooth function with Lipschitz constant = smoothness
    Pytorch version
    '''
    x = x0
    y = x0
    xs = [x0]
    k = 1
    for t in range(1, n_iterations+1):
        x2 = (k-1.)/(k+2.)*x + gradient(y, t)
        y2 = y - (1./smoothness)*x2
        
        # Restarting
        if func(x2) > func(x):
            k = 1
        else:
            k += 1
        
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag_scheduled_restarting(x0, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent with scheduled restarting for smooth
    function with Lipschitz constant = smoothness.
    '''
    x = x0
    y = x0
    xs = [x0]
    k = 1#0
    for t in range(1, n_iterations+1):
        #x2 = y - (1./smoothness/(10**(t/10000)))*gradient(y, t)
        x2 = y - (1./smoothness)*gradient(y, t)
        y2 = x2 + (k-1.)/(k+2.)*(x2-x)
        
        '''
        #if k >= 1000: # 20, 20*2^1, 20*2^2, ...
        if k >= 200: # Constant scheduling, no noise
            k = 1
        else:
            k += 1
        '''
        
        if t < 10000:
            if k >= 200:
                k = 1
            else:
                k += 1
        elif t < 30000:
            if k >= 200*(2*1):
                k = 1
            else:
                k += 1
        '''
        elif t < 50000:
            if k >= 200*(2**2):
                k = 1
            else:
                k += 1
        '''
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag_s_scheduled_restarting(x0, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent with scheduled restarting for smooth function with Lipschitz constant = smoothness
    Sutskever's version
    '''
    x = x0
    y = x0
    xs = [x0]
    k = 1
    for t in range(1, n_iterations+1):
        x2 = (k-1.)/(k+2.)*x + (1./smoothness)*gradient(y, t)
        y2 = y - x2
        
        '''
        #if k >= 1000: # 20, 20*2^1, 20*2^2, ...
        if k >= 1000: # Constant scheduling, no noise
            k = 1
        else:
            k += 1
        '''
        
        if t < 2000:
            if k >= 20:
                k = 1
            else:
                k += 1
        elif t < 10000:
            if k >= 20*(2*1):
                k = 1
            else:
                k += 1
        elif t < 50000:
            if k >= 20*(2**2):
                k = 1
            else:
                k += 1
        
        x = x2
        y = y2
        xs.append(y)
    return xs

def nag_p_scheduled_restarting(x0, gradient, smoothness=1., n_iterations=100):
    '''
    Nesterov accelerated gradient descent with scheduled restarting for smooth function with Lipschitz constant = smoothness
    Pytorch version
    '''
    x = x0
    y = x0
    xs = [x0]
    k = 1
    for t in range(1, n_iterations+1):
        x2 = (k-1.)/(k+2.)*x + gradient(y, t)
        y2 = y - (1./smoothness)*x2
        
        '''
        #if k >= 1000: # 20, 20*2^1, 20*2^2, ...
        if k >= 1000: #1000: # Constant scheduling, no noise
            k = 1
        else:
            k += 1
        '''
        
        if t < 2000:
            if k >= 100:
                k = 1
            else:
                k += 1
        elif t < 10000:
            if k >= 100*(2*1):
                k = 1
            else:
                k += 1
        elif t < 50000:
            if k >= 100*(2**2):
                k = 1
            else:
                k += 1
        
        x = x2
        y = y2
        xs.append(y)
    return xs

#------------------------------------------------------------------------------
# Test the optimization algorithms
#------------------------------------------------------------------------------
its = 50000

# GD
xs_gd = gd(np.zeros(n), noisygrad, 4, its)
ys_gd = [abs(path(xs_gd[i]) - path(opt)) for i in range(0, its)]

# GD + constant momentum
xs_mgd = mgd(np.zeros(n), noisygrad, 4, its)
ys_mgd = [abs(path(xs_mgd[i]) - path(opt)) for i in range(0, its)]

# NAG
xs_nag = nag(np.zeros(n), noisygrad, 4, its)
ys_nag = [abs(path(xs_nag[i]) - path(opt)) for i in range(0, its)]

# NAG + adaptive restarting
xs_nag_ar = nag_adaptive_restarting(np.zeros(n), path, noisygrad, 4, its)
ys_nag_ar = [abs(path(xs_nag_ar[i]) - path(opt)) for i in range(0, its)]

# NAG + scheduled restarting
xs_nag_sr = nag_scheduled_restarting(np.zeros(n), noisygrad, 4, its)
ys_nag_sr = [abs(path(xs_nag_sr[i]) - path(opt)) for i in range(0, its)]


SMALL_SIZE = 11
MEDIUM_SIZE = 11
BIGGER_SIZE = 11

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

f1 = plt.figure()
ax = plt.subplot(111, xlabel='x', ylabel='y', title='Stochastic Optimization Algorithms -- Quadratic Function')
#for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#             ax.get_xticklabels() + ax.get_yticklabels()):
#    item.set_fontsize(30)

plt.figure(1, figsize=(6.5,6))
plt.clf()
plt.plot(ys_gd, 'b', lw=1, label='GD')
plt.plot(ys_mgd, 'g', lw=1, label='GD + Momentum')
plt.plot(ys_nag, 'r', lw=1, label='NAG')
plt.plot(ys_nag_ar, 'k', lw=1, label='NAG + Adaptive Restart')
plt.plot(ys_nag_sr, 'm', lw=1, label='NAG + Scheduled Restart')

#plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlim([0, its])
plt.ylim(1e-5, 1e1)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('|f(x) - f(x*)|')
#plt.show()
plt.savefig('Quadratic_Decaying_Noise.pdf')


# Save data to txt for visualization
with open('GD_Quadratic_Decaying_Noise.txt', 'w') as filehandle:
    filehandle.writelines("%20.14f\n" % item1 for item1 in ys_gd)
with open('MGD_Quadratic_Decaying_Noise.txt', 'w') as filehandle:
    filehandle.writelines("%20.14f\n" % item1 for item1 in ys_mgd)
with open('NAG_Quadratic_Decaying_Noise.txt', 'w') as filehandle:
    filehandle.writelines("%20.14f\n" % item1 for item1 in ys_nag)
with open('NAGAR_Quadratic_Decaying_Noise.txt', 'w') as filehandle:
    filehandle.writelines("%20.14f\n" % item1 for item1 in ys_nag_ar)
with open('NAGSR_Quadratic_Decaying_Noise.txt', 'w') as filehandle:
    filehandle.writelines("%20.14f\n" % item1 for item1 in ys_nag_sr)
