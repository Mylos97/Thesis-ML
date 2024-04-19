import torch
import torch.nn as nn
from ax.service.ax_client import AxClient, ObjectiveProperties
from helper import make_dataloader
from train import train, evaluate

def hyperparameterBO(model_class: nn.Module,  data, in_dim:int, out_dim:int , loss_function:nn.MSELoss | nn.BCELoss, device: torch.device):
    BATCH_SIZE = 64
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.1, 0.1])
    train_loader = make_dataloader(x=train_dataset, batch_size=BATCH_SIZE)
    val_loader = make_dataloader(x=val_dataset, batch_size=BATCH_SIZE)
    test_loader = make_dataloader(x=test_dataset, batch_size=BATCH_SIZE)

    def train_evaluate(params):
        model, _ = train(model_class=model_class, data_loader=train_loader, 
                      in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device, parameters=params)
        loss = evaluate(model=model, data_loader=val_loader, loss_function=loss_function, device=device)
        return loss
    
    ax_client = AxClient()
    parameters = [
            {
                "name": "lr",  
                "type": "range",  
                "bounds": [1e-6, 0.1],  
                "value_type": "float"  
            },
            {
                "name": "dropout",  
                "type": "range",  
                "bounds": [0.0, 0.5],
                "value_type": "float"  
            },
            {
                "name": "gradient_norm",  
                "type": "range",  
                "bounds": [0.5, 2.5],
                "value_type": "float"  
            },
            
    ]
    ax_client.create_experiment(
        name="tune_model",  
        parameters=parameters,
        objectives={"loss": ObjectiveProperties(minimize=True)}, 
    )
    ax_client.attach_trial(
        parameters={"lr": 0.00001, "dropout": 0.1, "gradient_norm": 1.0}
    )

    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(trial_index=0, raw_data=train_evaluate(baseline_parameters))

    for _ in range(25):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))
    
    ax_client.get_max_parallelism()
    ax_client.get_trials_data_frame()
    
    best_parameters, values = ax_client.get_best_parameters()
    mean, covariance = values
    print("Best parameters", best_parameters)
    print("With mean", mean)

    combined_train_valid_set = torch.utils.data.ConcatDataset([
        train_loader.dataset.dataset,
        val_loader.dataset.dataset,
    ])
    combined_train_valid_loader = torch.utils.data.DataLoader(
        combined_train_valid_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    df = ax_client.get_trials_data_frame()
    best_arm_idx = df.trial_index[df["accuracy"] == df["accuracy"].max()].values[0]
    best_arm = ax_client.get_trial_parameters(best_arm_idx)
    best_model, tree = train(model_class=model_class, data_loader=combined_train_valid_loader, in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device, parameters=best_arm)
    test_accuracy = evaluate(best_model, data_loader=test_loader, loss_function=loss_function, device=device)

    print(f"Classification Accuracy (test set): {round(test_accuracy*100, 2)}%")