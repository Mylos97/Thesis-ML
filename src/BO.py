import torch
import torch.nn as nn
import warnings
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from helper import convert_to_json

def bayesian_optimization(ML_model, device, plan):
    print("Running BO")
    dtype = torch.float64
    encoded_plan = ML_model.encoder(plan)
    latent_vector = encoded_plan[0]
    indexes = encoded_plan[1]
    d = latent_vector.shape[1]
    n = 10
    BATCH_SIZE = 3
    NUM_RESTARTS = 10
    RAW_SAMPLES = 256
    bounds = torch.tensor([[-6.0] * d, [6.0] * d], device=device, dtype=dtype) 

    def get_latencies(plans) -> list[torch.Tensor]:
        print(plans[0])
        quit()
        results = []
        for plan in plans:
            results.append(plan[0].sum().item())
        convert_to_json(plans)
        return results
    
    def objective_function(X):
        v_hat = [latent_vector + v for v in X]
        model_results = []
        for v in v_hat:
            model_results.append(ML_model.decoder(v, indexes))
        results = get_latencies(model_results)

        return torch.tensor(results).double()

    def gen_initial_data():
        train_x = unnormalize(
            torch.rand(n, d, device=device, dtype=dtype), 
            bounds=bounds)
        train_obj = objective_function(train_x).unsqueeze(-1)
        best_observed_value = train_obj.min().item()
        return train_x, train_obj, best_observed_value


    def get_fitted_model(train_x, train_obj, state_dict=None):
        model = SingleTaskGP(
            train_X=normalize(train_x, bounds), 
            train_Y=train_obj,
            outcome_transform=Standardize(m=1)
        )
        if state_dict is not None:
            model.load_state_dict(state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.to(train_x)
        fit_gpytorch_mll(mll)
        return model

    def optimize_acqf_and_get_observation(acq_func):
        """Optimizes the acquisition function, and returns a
        new candidate and a noisy observation"""

        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack(
                [
                    torch.zeros(d, dtype=dtype, device=device),
                    torch.ones(d, dtype=dtype, device=device),
                ]
            ),
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        new_x = unnormalize(candidates.detach(), bounds=bounds)
        new_obj = objective_function(new_x).unsqueeze(-1)
        return new_x, new_obj

    torch.manual_seed(42)
    N_BATCH = 25
    best_observed = []
    train_x, train_obj, best_value = gen_initial_data()
    best_observed.append(best_value)
    state_dict = None
    for _ in range(N_BATCH):

        # fit the model
        model = get_fitted_model(
            train_x=train_x,
            train_obj=train_obj,
            state_dict=state_dict,
        )

        # define the qNEI acquisition function
        qEI = qExpectedImprovement(
            model=model, best_f=train_obj.min()
        )

        # optimize and get new observation
        new_x, new_obj = optimize_acqf_and_get_observation(qEI)

        train_x = torch.cat((train_x, new_x))
        train_obj = torch.cat((train_obj, new_obj))

        best_value = train_obj.min().item()
        best_observed.append(best_value)

        state_dict = model.state_dict()
        print(".", end="")