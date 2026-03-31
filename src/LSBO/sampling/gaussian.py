import torch

torch.set_float32_matmul_precision("highest")

import gpytorch
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# Multi-task Variational GP:
# https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/index.html

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, X, output_indices=None, observation_noise=False, *args, **kwargs) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode
        self.likelihood.eval()

        # Old version, leads to weird shape bc of default taking 10 samples w/ CensoredGaussianLikelihood
        mvn = self(X)
        dist = self.likelihood(
            mvn, censoring=torch.zeros(X.shape[0]), device=device, censored=False
        )  # Normal(loc: torch.Size([10, 5000]), scale: torch.Size([10, 5000]))

        return GPyTorchPosterior(dist)

def mvn_sample(mvn: MultivariateNormal):
    function_samples = mvn.rsample().unsqueeze(0)

    # Standard sampling
    if function_samples.isfinite().all():
        return function_samples

    # First failure, add jitter
    mvn_with_jitter = MultivariateNormal(mvn.loc, gpytorch.add_jitter(mvn.lazy_covariance_matrix, jitter_val=1e-5))
    function_samples = mvn_with_jitter.rsample().unsqueeze(0)

    if function_samples.isfinite().all():
        return function_samples

    # Second failure, convert to double precision with jitter
    mvn_double_precision = MultivariateNormal(
        mvn.loc.double(), gpytorch.add_jitter(mvn.lazy_covariance_matrix.double(), jitter_val=1e-5)
    )
    function_samples = mvn_double_precision.rsample().unsqueeze(0)

    if function_samples.isfinite().all():
        return function_samples

    # This is wild just return the mean
    function_samples = mvn.loc.unsqueeze(0)

    return function_samples
