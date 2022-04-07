import numpy as np
import GPy
from GPyOpt.methods import BayesianOptimization


def f(x, beta=0.2, alpha1=1.0, alpha2=1.0):
    return np.sin(3.0*x) - alpha1*x + alpha2*x**2 + beta*np.random.randn(x.shape[0])


kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=4.0)
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-3, 3)}]
opt = BayesianOptimization(f=f, domain=domain, model_type='GP', initial_design_numdata = 1, kernel=kernel, acquisition_type='EI')
opt.run_optimization(max_iter=6)
opt.plot_acquisition()
