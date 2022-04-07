import numpy as np
import GPy
from GPyOpt.methods import BayesianOptimization


def f(x, alpha=1.0, beta=0.5, gamma=0.2):
    return alpha*np.sin(2*np.pi*x/10) + ((beta*np.sin((2*np.pi*x/0.5))) +
                (gamma*x) + (0.1*np.random.randn(x.shape[0])))


kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=4.0)
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 10)}]
opt = BayesianOptimization(f=f, 
                           domain=domain, 
                           model_type='GP', 
                           initial_design_numdata = 5, 
                           kernel=kernel, 
                           acquisition_type='EI')
opt.run_optimization(max_iter=10)
opt.plot_acquisition()
