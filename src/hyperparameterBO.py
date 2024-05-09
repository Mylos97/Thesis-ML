import torch
import torch.nn as nn
from ax.service.ax_client import AxClient, ObjectiveProperties
from helper import get_data_loaders
from train import train, evaluate


def do_hyperparameter_BO(model_class: nn.Module,  data, in_dim:int, out_dim:int , loss_function:nn.Module, device: torch.device, weights:dict=None):
    TRIALS = 10

    def train_evaluate(params):
        batch_size = params.get('batch_size', 32)
        train_loader, val_loader, test_loader = get_data_loaders(data=data, batch_size=batch_size)
        print(f'Batch size: {batch_size}')
        print(f'Training batches: {len(train_loader)} Test batches: {len(test_loader)} Validation batches: {len(val_loader)} \n', flush=True)
        model, _ = train(model_class=model_class, training_data_loader=train_loader, val_data_loader=val_loader,
                      in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device, parameters=params, weights=weights)
        loss = evaluate(model=model, val_data_loader=val_loader, loss_function=loss_function, device=device)
        print(f'Validation loss for the model after training {loss}', flush=True)

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
        {
            'name': 'patience',
            'type': 'range',
            'bounds': [1, 50],
            'value_type': 'int'
        },
        {
            'name': 'batch_size',
            'type': 'range',
            'bounds': [32, 128],
            'value_type': 'int'
        },
    ]

    ax_client.create_experiment(
        name='tune_model',
        parameters=parameters,
        objectives={'loss': ObjectiveProperties(minimize=True)},
    )
    ax_client.attach_trial(
        parameters={'lr': 1e-6, 'dropout': 0.1, 'gradient_norm': 1.0, 'patience': 1, 'batch_size':32}
    )

    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(trial_index=0, raw_data=train_evaluate(baseline_parameters))

    for i in range(TRIALS):
        print(f'Started Hyperparameter BO trial {i}', flush=True)
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))

    ax_client.get_max_parallelism()
    ax_client.get_trials_data_frame()

    best_parameters, _ = ax_client.get_best_parameters()
    train_loader, val_loader, test_loader = get_data_loaders(data=data, batch_size=best_parameters.get('batch_size', 32))

    combined_train_valid_set = torch.utils.data.ConcatDataset([
        train_loader.dataset.dataset,
        val_loader.dataset.dataset,
    ])
    combined_train_valid_loader = torch.utils.data.DataLoader(
        combined_train_valid_set,
        batch_size=best_parameters.get('batch_size', 32),
        shuffle=True
    )
    df = ax_client.get_trials_data_frame()
    best_arm_idx = df.trial_index[df['loss'] == df['loss'].max()].values[0]
    best_arm = ax_client.get_trial_parameters(best_arm_idx)
    print(f'\nBest model training with parameters: {best_parameters}', flush=True)
    best_model, tree = train(model_class=model_class, training_data_loader=combined_train_valid_loader, val_data_loader=val_loader, in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device, parameters=best_arm, weights=weights)
    test_accuracy = evaluate(best_model, val_data_loader=test_loader, loss_function=loss_function, device=device)

    print(f'Best model loss test set: {test_accuracy}', flush=True)
    return best_model, tree