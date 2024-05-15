import torch
import datetime
import torch.nn as nn
from ax.service.ax_client import AxClient, ObjectiveProperties
from helper import get_data_loaders, get_relative_path
from train import train, evaluate
from ax.utils.notebook.plotting import render
from OurModels.EncoderDecoder.bvae import BVAE 

def do_hyperparameter_BO(model_class: nn.Module,  data, in_dim:int, out_dim:int , loss_function:nn.Module, device: torch.device, lr, epochs, trials, plots, weights:dict=None):
    def train_evaluate(params):
        batch_size = params.get('batch_size', 32)
        train_loader, val_loader, test_loader = get_data_loaders(data=data, batch_size=batch_size)
        if model_class == BVAE:
            l_function = loss_function(parameters.get('beta',1.0))
        else:
            l_function = loss_function()
        
        print(f'Batch size: {batch_size}')
        print(f'Training batches: {len(train_loader)} Test batches: {len(test_loader)} Validation batches: {len(val_loader)} \n', flush=True)
        model, _ = train(model_class=model_class, training_data_loader=train_loader, val_data_loader=val_loader,
                      in_dim=in_dim, out_dim=out_dim, loss_function=l_function, device=device, parameters=params, epochs=epochs, weights=weights)
        loss = evaluate(model=model, val_data_loader=val_loader, loss_function=l_function, device=device)
        print(f'Validation loss for the model after training {loss}', flush=True)
        return loss

    ax_client = AxClient()
    parameters = [
        {
            'name': 'lr',
            'type': 'range',
            'bounds': lr,
            'value_type': 'float',
            "log_scale": True,
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
            'bounds': [32, 64],
            'value_type': 'int'
        },
    ]

    if model_class == BVAE:
        parameters.append({
            'name': 'beta',
            'type': 'range',
            'bounds': [1.0, 4.0],
            'value_type': 'float',
            "log_scale": True,
        })

    ax_client.create_experiment(
        name='tune_model',
        parameters=parameters,
        objectives={'loss': ObjectiveProperties(minimize=True)},
    )
    
    for _ in range(trials):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))

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

    print(f'\nBest model training with parameters: {best_parameters}', flush=True)

    if model_class == BVAE:
        l_function = loss_function(best_parameters.get('beta', 1.0))
    else:
        l_function = loss_function()
    
    best_model, tree = train(model_class=model_class, training_data_loader=combined_train_valid_loader, val_data_loader=val_loader, in_dim=in_dim, out_dim=out_dim, loss_function=l_function, device=device, parameters=best_parameters, epochs=epochs, weights=weights)
    ax_path = get_relative_path('test.json', 'Logs')  #get_relative_path(file_name=f'{type(best_model).__name__}{datetime.datetime.now().strftime("%d-%H:%M:%S")}.json', dir='Logs')
    ax_client.save_to_json_file(ax_path)
    test_accuracy = evaluate(best_model, val_data_loader=test_loader, loss_function=l_function, device=device)
    
    if plots:
        try:
            render(ax_client.get_optimization_trace())
            render(ax_client.get_contour_plot(param_x="lr", param_y="batch_size", metric_name="loss"))
        except:
            print('Could not produce plots')
    
    print(f'Best model loss test set: {test_accuracy}', flush=True)
    return best_model, tree