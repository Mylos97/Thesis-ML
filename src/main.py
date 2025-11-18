import torch
import argparse
import torch.onnx
import ast
import json
from exporter import export_model
from OurModels.PairWise.model import Pairwise
from OurModels.CostModel.model import CostModel
from OurModels.EncoderDecoder.model import VAE
from OurModels.EncoderDecoder.bvae import BVAE

from helper import load_autoencoder_data, load_pairwise_data, load_costmodel_data, get_relative_path, get_weights_of_model_by_path, Beta_Vae_Loss, set_weights, load_autoencoder_data_from_str
from hyperparameterBO import do_hyperparameter_BO


def main(args) -> None:
    torch.manual_seed(42)
    model_class = None
    loss_function = None
    data = None
    weights = None
    path = None
    test_data = None
    val_data = None
    lr = ast.literal_eval(args.lr)
    epochs = args.epochs
    trials = args.trials
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    args_name = args.name if ".onnx" in args.name else f"{args.name}.onnx"
    #torch.autograd.set_detect_anomaly(True)
    print(f"Started training model {args.model} at {args.model_path}", flush=True)

    if args.model == "vae":
        model_class = VAE
        data, in_dim, out_dim = load_autoencoder_data(path=path, retrain_path=args.retrain, device=device, num_ops=args.operators, num_platfs=args.platforms)
        loss_function = torch.nn.CrossEntropyLoss

    if args.model == 'bvae':
        """
        data, in_dim, out_dim = load_autoencoder_data(path=get_relative_path('train.txt', 'Data/splits/tpch/bvae'), retrain_path=args.retrain, device=device, num_ops=args.operators, num_platfs=args.platforms)
        test_data, _, _ = load_autoencoder_data(path=get_relative_path('test.txt', 'Data/splits/tpch/bvae'), retrain_path='', device=device, num_ops=args.operators, num_platfs=args.platforms)
        val_data, _, _ = load_autoencoder_data(path=get_relative_path('test.txt', 'Data/splits/tpch/bvae'), retrain_path='', device=device, num_ops=args.operators, num_platfs=args.platforms)
        #data, in_dim, out_dim = load_autoencoder_data(path=get_relative_path('train.txt', 'Data/splits/tpch/bvae'), retrain_path=args.retrain, device=device, num_ops=args.operators, num_platfs=args.platforms)
        """
        data, in_dim, out_dim = load_autoencoder_data(path=get_relative_path('test-queries.txt', 'Data/splits/imdb/training'), retrain_path=args.retrain, device=device, num_ops=args.operators, num_platfs=args.platforms)
        test_data, _, _ = load_autoencoder_data(path=get_relative_path('test-queries.txt', 'Data/splits/imdb/training'), retrain_path='', device=device, num_ops=args.operators, num_platfs=args.platforms)
        val_data, _, _ = load_autoencoder_data(path=get_relative_path('test-queries.txt', 'Data/splits/imdb/training'), retrain_path='', device=device, num_ops=args.operators, num_platfs=args.platforms)
        model_class = BVAE
        loss_function = Beta_Vae_Loss

    if args.model == "pairwise":
        data, in_dim, out_dim = load_pairwise_data(path=path, device=device)
        model_class = Pairwise
        loss_function = torch.nn.BCELoss

    if args.model == "cost":
        #data, in_dim, out_dim = load_costmodel_data(path=get_relative_path('training.txt', 'Data/splits'), device=device)
        data, in_dim, out_dim = load_costmodel_data(path=get_relative_path('pointwise.txt', 'Data/splits/imdb/training'), device=device)
        test_data, _, _ = load_costmodel_data(path=get_relative_path('pointwise.txt', 'Data/splits/imdb/training'), device=device)
        val_data, _, _ = load_costmodel_data(path=get_relative_path('pointwise.txt', 'Data/splits/imdb/training'), device=device)
        """
        data, val_data, test_data = torch.utils.data.random_split(
            data, [0.8, 0.1, 0.1]
        )
        """
        model_class = CostModel
        loss_function = torch.nn.L1Loss

    print(
        f"Succesfully loaded data with in_dimensions:{in_dim} out_dimensions:{out_dim}",
        flush=True,
    )

    if args.retrain:
        # dont do shit
        print("Retraining a model, not actually running hyperparameterBO")
        with open(args.parameters) as file:
            best_parameters = json.load(file)
            #best_parameters["batch_size"] = 7

        weights = get_weights_of_model_by_path(args.model_path)

        best_model, x = do_hyperparameter_BO(
                model_class=model_class,
                data=data,
                test_data=test_data,
                val_data=val_data,
                in_dim=in_dim,
                out_dim=out_dim,
                loss_function=loss_function,
                device=device,
                lr=lr,
                weights=weights,
                epochs=epochs,
                trials=trials,
                plots=args.plots,
                best_parameters=best_parameters,
                parameters_path=args.parameters
        )

        if args.model_path is not None:
            model_name = args.model_path
        else:
            model_name = f"{args.model}.onnx" if len(args_name) < 6 else args.name
            model_name = get_relative_path(model_name, "Models")

        export_model(
            model=best_model, x=x, model_name=model_name
        )

    else:
        best_model, x = do_hyperparameter_BO(
                model_class=model_class,
                data=data,
                test_data=test_data,
                val_data=val_data,
                in_dim=in_dim,
                out_dim=out_dim,
                loss_function=loss_function,
                device=device,
                lr=lr,
                weights=weights,
                epochs=epochs,
                trials=trials,
                plots=args.plots,
                parameters_path=args.parameters
        )

        if args.model_path is not None:
            model_name = args.model_path
        else:
            model_name = f"{args.model}.onnx" if len(args_name) < 6 else args.name
            model_name = get_relative_path(model_name, "Models")

        export_model(
            model=best_model, x=x, model_name=model_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vae')
    parser.add_argument('--model-path', default='./src/Data/vae.onnx')
    parser.add_argument('--parameters', default='./src/HyperparameterLogs/BVAE.json')
    parser.add_argument('--retrain', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--lr', type=str, default='[1e-6, 0.1]')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--plots', type=bool, default=False)
    parser.add_argument('--platforms', type=int, default=9)
    parser.add_argument('--operators', type=int, default=43)
    args = parser.parse_args()
    main(args)
