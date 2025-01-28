import torch
import datetime
import torch.nn as nn
import json
from ax.service.ax_client import AxClient, ObjectiveProperties
from helper import generate_latency_map_intersect, get_data_loaders, get_relative_path, make_dataloader
from train import train, evaluate
from ax.utils.notebook.plotting import render
from OurModels.EncoderDecoder.bvae import BVAE
from OurModels.EncoderDecoder.model import VAE


def do_hyperparameter_BO(
    model_class: nn.Module,
    data,
    in_dim:int,
    out_dim:int,
    loss_function:nn.Module,
    device: torch.device,
    parameters_path: str,
    lr,
    epochs,
    trials,
    plots,
    test_data = None,
    val_data = None,
    weights:dict=None,
    best_parameters=None
    ):
    def train_evaluate(params):
        batch_size = params.get('batch_size', 32)

        train_loader, val_loader, test_loader = get_data_loaders(
            data=data,
            test_data=test_data,
            val_data=val_data,
            batch_size=batch_size
        )

        print(f"Train data: {train_loader}")
        print(f"Test data: {test_loader}")
        print(f"Val data: {val_loader}")

        if model_class == BVAE:
            l_function = loss_function(
                beta=parameters.get('beta', 1.0),
                #beta=4,
            )
            #l_function = loss_function(beta=1.5)
        else:
            l_function = loss_function()

        print(f'Batch size: {batch_size}')
        print(f'Training batches: {len(train_loader)} Test batches: {len(test_loader)} Validation batches: {len(val_loader)} \n', flush=True)
        model, _ = train(
            model_class=model_class,
            training_data_loader=train_loader,
            test_data_loader=test_loader,
            val_data_loader=val_loader,
            in_dim=in_dim,
            out_dim=out_dim,
            loss_function=l_function,
            device=device,
            parameters=params,
            epochs=epochs,
            weights=weights
        )
        print(f"Training on {device}")
        loss = evaluate(
            model=model,
            val_data_loader=test_loader,
            loss_function=l_function,
            device=device
        )
        print(f'Test loss for the model after training {loss}', flush=True)
        return loss

    is_retraining = best_parameters is not None
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
            'bounds': [0.0, 0.4],
            'value_type': 'float'
        },
        {
            'name': 'gradient_norm',
            'type': 'range',
            'bounds': [0.5, 2.5],
            'value_type': 'float'
        },
        {
            'name': 'batch_size',
            'type': 'range',
            'bounds': [32, 64],
            #'bounds': [2, 64],
            'value_type': 'int'
        },
        {
            'name': 'patience',
            'type': 'range',
            'bounds': [5, 50],
            'value_type': 'int'
        },
    ]


    if model_class == BVAE:
        parameters.append({
            'name': 'beta',
            'type': 'range',
            'bounds': [5.0, 150.0],
            'value_type': 'float',
            "log_scale": True,
        })

        parameters.append({
            'name': 'z_dim',
            'type': 'range',
            'bounds': [1, 31],
            'value_type': 'int',
            'is_ordered': True,
            'sort_values' : True
        })

    """
    if model_class == VAE or model_class == BVAE:
        parameters.append({
            'name': 'z_dim',
            'type': 'choice',
            'values': [2, 4, 8, 16, 32],
            'value_type': 'int',
            'is_ordered': True,
            'sort_values' : True
        })
        parameters.append({
            'name': 'z_dim',
            'type': 'range',
            'bounds': [2, 31],
            'value_type': 'int',
            'is_ordered': True,
            'sort_values' : True
        })
    """

    torch.manual_seed(42)

    if best_parameters is None:
        ax_client.create_experiment(
            name='tune_model',
            parameters=parameters,
            objectives={
                'loss': ObjectiveProperties(minimize=True),
                'kld': ObjectiveProperties(minimize=True)
            },
        )

        trial_eval_map = {}

        for _ in range(trials):
            parameters, trial_index = ax_client.get_next_trial()
            raw_data = train_evaluate(parameters)
            print(f"Parameters: {parameters}")
            print(f"raw_data: {raw_data}")
            ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            #trial_eval_map[raw_data] = parameters

        #best_parameters, _ = ax_client.get_best_parameters()
        print(f"Best parameters: {ax_client.get_pareto_optimal_parameters()}")
        print(f"Best parameters: {list(ax_client.get_pareto_optimal_parameters().items())[0][1][0]}")
        best_parameters, _ = list(ax_client.get_pareto_optimal_parameters().items())[0][1]
        #print(f"Trial eval map: {sorted([key for key, value in trial_eval_map.items()])}")
        print(f"Loss of best_parameters {list(filter(lambda x: x[1] == best_parameters, trial_eval_map.items()))}")


    if best_parameters is not None and (model_class == BVAE or model_class == VAE):
        batch_size = best_parameters.get('batch_size')
        samples_needed = batch_size-(batch_size%len(data))
        print("Starting batch generation ", batch_size, " lenght of data: ", len(data))
        print("with samples needed: ", samples_needed)

        if samples_needed > 0:
            for i in range(samples_needed+batch_size*10):
                elem = data.__getitem__(i%len(data))
                data.append(elem)

    train_loader, val_loader, test_loader = get_data_loaders(
        data=data,
        test_data=test_data,
        val_data=val_data,
        batch_size=best_parameters.get('batch_size', 32),
    )

    """
    if is_retraining:
        combined_train_valid_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=best_parameters.get('batch_size', 32),
            shuffle=True
        )
    else:
    """
    combined_train_valid_set = torch.utils.data.ConcatDataset([
        train_loader.dataset,
        val_loader.dataset,
    ])

    combined_train_valid_loader = torch.utils.data.DataLoader(
        combined_train_valid_set,
        batch_size=best_parameters.get('batch_size', 32),
        shuffle=True
    )

    print(f'\nBest model training with parameters: {best_parameters}', flush=True)

    if model_class == BVAE:
        l_function = loss_function(
            beta=best_parameters.get('beta', 1.0),
           # beta=4,
        )
        #l_function = loss_function(beta=1.5)
    else:
        l_function = loss_function()

    best_model, tree = train(
        model_class=model_class,
        training_data_loader=train_loader,
        test_data_loader=test_loader,
        val_data_loader=val_loader,
        in_dim=in_dim,
        out_dim=out_dim,
        loss_function=l_function,
        device=device,
        parameters=best_parameters,
        epochs=epochs,
        weights=weights
    )
    test_accuracy = evaluate(best_model, val_data_loader=test_loader, loss_function=l_function, device=device)

    # write best parameters to file
    if not is_retraining:
        #with open(get_relative_path(f"{type(best_model).__name__}.json", 'HyperparameterLogs'), 'w') as file:
        with open(parameters_path, 'w') as file:
            json.dump(best_parameters, file)

    if plots:
        try:
            ax_path = get_relative_path(f"{type(best_model).__name__}.json", 'Logs')  #get_relative_path(file_name=f'{type(best_model).__name__}{datetime.datetime.now().strftime("%d-%H:%M:%S")}.json', dir='Logs')
            ax_client.save_to_json_file(ax_path)
            render(ax_client.get_optimization_trace())
            render(ax_client.get_contour_plot(param_x="lr", param_y="batch_size", metric_name="loss"))
        except:
            print('Could not produce plots')

    print(f'Best model loss test set: {test_accuracy}', flush=True)
    return best_model, tree
