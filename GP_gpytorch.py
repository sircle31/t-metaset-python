import torch
import gpytorch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import scipy.io as sio
from sklearn.metrics import mean_squared_error

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        num_tasks = train_y.shape[-1]
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.num_outputs = num_tasks
        self.likelihood = likelihood  # Re-assigning now is safe        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_outputs]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([self.num_outputs])),
            batch_shape=torch.Size([self.num_outputs])
        )
        self.train_x = train_x
        self.train_y =train_y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

    def train_model(self, training_iterations=200, lr=0.1, verbose=False):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        
        print_interval = training_iterations // 10
        for i in range(training_iterations):
#        for i in tqdm(range(training_iterations), desc='Training Progress'):            
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            if verbose and (i + 1) % print_interval == 0:
                print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item():.3f}')
            optimizer.step()

    def pred(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self(x))
        return predictions.mean

    def roughness(self):
        self.eval()
        lengthscale_value = self.covar_module.base_kernel.raw_lengthscale[:self.num_outputs].detach().numpy().squeeze()
        return lengthscale_value            
            
    def compute_error(self):
        self.pred_train_test() # #$%
        train_y_flattened = self.train_y.numpy().flatten()
        mean_flattened = self.mean_train.numpy().flatten()
        self.mse_train = mean_squared_error(train_y_flattened, mean_flattened)
        print(f"MSE for Training Data: {self.mse_train:.4e}")