import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, varSigma, lengthscale):
    if x2 is None: 
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    K = varSigma*np.exp(-np.power(d, 2)/lengthscale)
    return K

def lin_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma*x1.dot(x1.T)
    else:
        return varSigma*x1.dot(x2.T)
def white_kernel(x1, x2, varSigma):
    if x2 is None:
        return varSigma*np.eye(x1.shape[0])
    else:
        return np.zeros(x1.shape[0], x2.shape[0])
def periodic_kernel(x1, x2, varSigma, period, lenthScale):
    if x2 is None: 
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    return varSigma*np.exp(-(2*np.sin((np.pi/period)*d)**2)/lenthScale**2)

def gp_prediction(x1, y1, xstar, lengthScale, varSigma):
    k_starX = rbf_kernel(xstar,x1,varSigma,lengthScale)
    k_xx = rbf_kernel(x1, None, varSigma, lengthScale)
    k_starstar = rbf_kernel(xstar,None,varSigma,lengthScale)
    mu = k_starX.dot(np.linalg.inv(k_xx)).dot(y1)
    var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx)).dot(k_starX.T)

    for i in range(k_xx.shape[0]):
        print(k_xx[i][i])

    return mu, var, xstar

N = 5 
x = np.linspace(-3.1,3,N)
y = np.sin(2*np.pi/x) + x*0.1 #+ 0.3*np.random.randn(x.shape[0])
x = np.reshape(x,(-1,1))
y = np.reshape(y,(-1,1))
x_star = np.linspace(-6, 6, 500)
x_star = np.reshape(x_star,(-1,1))
mu_star, var_star, x_star = gp_prediction(x, y, x_star, 10, 100)
mu_star=np.reshape(mu_star,(1,-1))
mu_star=np.ndarray.flatten(mu_star)
'''for i in range(var_star.shape[0]):
    print(var_star[i][i])'''

Nsamp = 100
f_star = np.random.multivariate_normal(mu_star, var_star, Nsamp)
'''# choose index set for the marginal6
x = np.linspace(-6, 6, 200).reshape(-1, 1)
# compute covariance matrix
#K = rbf_kernel(x, None, 1.0, 8.0)
K = lin_kernel(x, None, 1.0)
# create mean vector
mu = np.zeros(x.shape[0])
# draw samples 20 from Gaussian distribution
f = np.random.multivariate_normal(mu, K, 20)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, f.T)'''

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_star, f_star.T,zorder=1)
ax.scatter(x, y, 200, 'cyan', '*', zorder=2)

fig.savefig('picture4')