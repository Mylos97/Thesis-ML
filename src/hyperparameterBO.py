import torch
import torch.nn as nn
from ax.service.ax_client import AxClient, ObjectiveProperties
from helper import make_dataloader
from train import train, evaluate

def do_hyperparameter_BO(model_class: nn.Module,  data, in_dim:int, out_dim:int , loss_function:nn.Module, device: torch.device, weights:dict=None):
    BATCH_SIZE = 64
    TRIALS = 1
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.1, 0.1])
    train_loader = make_dataloader(x=train_dataset, batch_size=BATCH_SIZE)
    val_loader = make_dataloader(x=val_dataset, batch_size=BATCH_SIZE)
    test_loader = make_dataloader(x=test_dataset, batch_size=BATCH_SIZE)
    print(f'Training dataset size: {len(train_dataset)} Test dataset size: {len(test_dataset)} Validation dataset size: {len(val_dataset)}')
    print(f'Training batches: {len(train_loader)} Test batches: {len(test_loader)} Validation batches: {len(val_loader)} \n')

    def train_evaluate(params):
        model, _ = train(model_class=model_class, training_data_loader=train_loader, val_data_loader=val_loader,
                      in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device, parameters=params, weights=weights)
        loss = evaluate(model=model, val_data_loader=val_loader, loss_function=loss_function, device=device)
        print(f'Validation loss for the model after training {loss}')

        return loss

    ax_client = AxClient(verbose_logging=False)
    parameters = [
            {
                'name': 'lr',
                'type': 'range',
                'bounds': [1e-6, 0.1],
                'value_type': 'float'
            },
            {
                'name': 'dropout',
                'type': 'range',
                'bounds': [0.0, 0.5],
                'value_type': 'float'
            },
            {
                'name': 'gradient_norm',
                'type': 'range',
                'bounds': [0.5, 2.5],
                'value_type': 'float'
            },

    ]
    ax_client.create_experiment(
        name='tune_model',
        parameters=parameters,
        objectives={'loss': ObjectiveProperties(minimize=True)},
    )
    ax_client.attach_trial(
        parameters={'lr': 0.00001, 'dropout': 0.1, 'gradient_norm': 1.0}
    )

    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(trial_index=0, raw_data=train_evaluate(baseline_parameters))

    for _ in range(TRIALS):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))

    ax_client.get_max_parallelism()
    ax_client.get_trials_data_frame()

    best_parameters, _ = ax_client.get_best_parameters()

    combined_train_valid_set = torch.utils.data.ConcatDataset([
        train_loader.dataset.dataset,
        val_loader.dataset.dataset,
    ])
    combined_train_valid_loader = torch.utils.data.DataLoader(
        combined_train_valid_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    df = ax_client.get_trials_data_frame()
    best_arm_idx = df.trial_index[df['loss'] == df['loss'].max()].values[0]
    best_arm = ax_client.get_trial_parameters(best_arm_idx)
    print(f'\n Training best model with parameters: {best_parameters}')
    best_model, tree = train(model_class=model_class, training_data_loader=combined_train_valid_loader, val_data_loader=val_loader, in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device, parameters=best_arm, weights=weights)
    test_accuracy = evaluate(best_model, val_data_loader=test_loader, loss_function=loss_function, device=device)

    print(f'Best model loss test set: {test_accuracy}')
    return best_model, tree
