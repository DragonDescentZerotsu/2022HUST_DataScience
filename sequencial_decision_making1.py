import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.stats as stat

def f(x, A=1, B=0, C=0):
    return A*(6*x-2)**2*np.sin(12*x-4) + B*(x-0.5) + C 

def rbf_kernel(x1, x2, varSigma, lengthscale):
    if x2 is None: 
        d = cdist(x1, x1)
    else:
        d = cdist(x1, x2)
    K = varSigma*np.exp(-np.power(d, 2)/lengthscale)
    return K

def gp_prediction(x1, y1, xstar, lengthScale, varSigma):
    k_starX = rbf_kernel(xstar,x1,varSigma,lengthScale)
    k_xx = rbf_kernel(x1, None, varSigma, lengthScale)
    k_starstar = rbf_kernel(xstar,None,varSigma,lengthScale)
    mu = k_starX.dot(np.linalg.inv(k_xx)).dot(y1)
    var = k_starstar - (k_starX).dot(np.linalg.inv(k_xx)).dot(k_starX.T)
    return mu, var, xstar

def expected_improvement(f_max, mu, varSigma, x):
    alpha=(f_max-mu)*stat.norm.cdf(f_max,mu,varSigma)+varSigma*stat.norm(mu,varSigma).pdf(f_max)
    return alpha

iterations=11
#choose the start points randomly
x_star = np.linspace(-0.1, 1, 200)
index=np.random.permutation(x_star.shape[0])
x_know=x_star[index[0:3]]
x_star = np.delete(x_star, index[0:3])
f_min=np.min(f(x_know)) #初始化f_min

for j in range(iterations):
    x_star = np.reshape(x_star,(-1,1))
    x_know = np.reshape(x_know,(-1,1))
    y=f(x_know)
    #y = np.reshape(y,(-1,1))
    mu_star, var_star, x_star = gp_prediction(x_know, y, x_star, 0.01, 1)
    mu_star=np.reshape(mu_star,(1,-1))
    mu_star=np.ndarray.flatten(mu_star)
    x_star=np.ndarray.flatten(x_star)

    variance_star=[]
    for i in range(var_star.shape[0]):
        variance_star.append(var_star[i][i])

    alpha=expected_improvement(f_min,mu_star,variance_star,x_star)
    ind = np.argmax(alpha)
    f_min=np.min(mu_star)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(-0.1, 1, 200),f(np.linspace(-0.1, 1, 200)), linestyle='--', color='r' , linewidth=3, label='True function')
    
    ax.plot(x_star, mu_star.T,zorder=1, label='Surrogate function')
    ax.plot(x_star,alpha,color='magenta')

    plt.legend()

    ax.fill_between(x_star,mu_star-np.sqrt(variance_star),mu_star+np.sqrt(variance_star),color='darkorange',alpha=0.3)
    ax.scatter(x_know, y, 200, 'cyan', '*', zorder=3)

    fig.savefig('./图片/'+'picture'+str(j))

    x_know=np.ndarray.flatten(x_know)
    x_know = np.append(x_know, x_star[ind])
    x_star=np.delete(x_star,ind)

#fig.savefig('picturet')
