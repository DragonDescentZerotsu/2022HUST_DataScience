import GPy
from GPyOpt.methods import BayesianOptimization
import numpy as np

def f(x, alpha=1.0, beta=0.5, gamma=0.2):
    return alpha*np.sin(2*np.pi*x/10) + ((beta*np.sin((2*np.pi*x/0.5))) +
                (gamma*x) + (0.1*np.random.randn(x.shape[0])))

kernel_rbf = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=4.0)
kernel_cmpnd = ((GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=4.0)) +
                (GPy.kern.StdPeriodic(input_dim=1, variance=1.0, period=10.0)) +
                (GPy.kern.StdPeriodic(input_dim=1, variance=1.0, period=0.5)) +
                (GPy.kern.Linear(input_dim = 1)) +
                (GPy.kern.White(input_dim=1, variance=0.1)))
domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,10)}]

opt = BayesianOptimization(f=f, 
                            domain=domain,
                            model_type='GP',
                            kernel=kernel_cmpnd,
                            acquisition_type='EI',
                            initial_design_numdata=2)
opt.run_optimization(max_iter=10)
opt.plot_acquisition()
