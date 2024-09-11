import torch
import torch.nn as nn
import json
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from helper import convert_to_json

class LSBOResult:
    def __init__(
        self,
        ml_model,
        model,
        model_results,
        tree,
        train_x,
        train_obj,
        state_dict,
        best_values
    ):
        self.ml_model = ml_model
        self.model = model
        self.model_results = model_results
        self.tree = tree
        self.train_x = train_x
        self.train_obj = train_obj
        self.state_dict = state_dict
        self.best_values = best_values

    def update(self, new_x, new_obj):
         # update training points
        self.train_x = torch.cat((self.train_x, new_x))
        self.train_obj = torch.cat((self.train_obj, new_obj))

        # update progress
        best_value = self.train_obj.max().item()
        self.best_values.append(best_value)

        self.state_dict = self.model.state_dict()


def latent_space_BO(ML_model, device, plan, previous: LSBOResult = None) -> LSBOResult:
    json_result = ""
    print('Running latent space Bayesian Optimization', flush=True)
    dtype = torch.float64
    for tree,target in plan:
        encoded_plan = ML_model.encoder(tree)
    latent_vector = encoded_plan[0]
    indexes = encoded_plan[1]
    d = latent_vector.shape[1]
    n = 1
    BATCH_SIZE = 1
    NUM_RESTARTS = 1
    RAW_SAMPLES = 256
    is_initial_run = False
    bounds = torch.tensor([[-6.0] * d, [6.0] * d], device=device, dtype=dtype)

    def get_latencies(plans) -> list[torch.Tensor]:
        results = []
        for plan in plans:
            results.append(plan[0].sum().item())

        #convert_to_json(plans)
        return results

    def objective_function(X):
        v_hat = [latent_vector + v for v in X]
        model_results = []
        for v in v_hat:
            model_results.append(ML_model.decoder(v.float(), indexes)) # Cast to float() because of warning from BoTorch
        results = get_latencies(model_results)

        return torch.tensor(results)

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

    if previous is None:
        is_initial_run = True
        best_observed = []
        train_x, train_obj, best_value = gen_initial_data()
        best_observed.append(best_value)
        state_dict = None
        model_results = []

        previous = LSBOResult(ML_model, None, model_results, tree, train_x, train_obj, state_dict, best_observed)

    model = get_fitted_model(
        train_x=previous.train_x,
        train_obj=previous.train_obj.double(),
        state_dict=previous.state_dict,
    )

    previous = LSBOResult(ML_model, model, previous.model_results, tree, previous.train_x, previous.train_obj, previous.state_dict, previous.best_values)

    qEI = qExpectedImprovement(
        model=model, best_f=previous.train_obj.min()
    )

    new_x, new_obj = optimize_acqf_and_get_observation(qEI)
    v_hat = [latent_vector + v for v in new_x]
    for v in v_hat:
        # Cast to float() because of warning from BoTorch
        decoded = ML_model.decoder(v.float(), indexes)
        x = ML_model.softmax(decoded[0])
        previous.model_results.append([x.detach().numpy().tolist()[0], decoded[1].detach().numpy().tolist()[0]])

    if is_initial_run:
        previous.train_x = new_x
        previous.train_obj = new_obj
    else:
        previous.update(new_x, new_obj)
    #json_result = json.dumps({"data": model_results})

    print('Finish Bayesian Optimization for latent space', flush=True)

    return previous
