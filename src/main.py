import torch
import argparse
import torch.onnx
import ast
from exporter import export_model
from OurModels.PairWise.model import Pairwise
from OurModels.CostModel.model import CostModel
from OurModels.EncoderDecoder.model import VAE
from helper import (
    load_autoencoder_data,
    load_pairwise_data,
    load_costmodel_data,
    get_relative_path,
    get_weights_of_model,
)
from hyperparameterBO import do_hyperparameter_BO
from latentspaceBO import latent_space_BO


def main(args) -> None:
    model_class = None
    loss_function = None
    data = None
    weights = None
    path = None
    lr = ast.literal_eval(args.lr)
    epochs = args.epochs
    trials = args.trials
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_name = args.name if ".onnx" in args.name else f"{args.name}.onnx"
    print(f"Started training model {args.model}", flush=True)

    if args.retrain:
        print(f"Retraining model {args.model}")
        weights = get_weights_of_model(args.model)
        path = args.retrain

    if args.model == "vae":
        data, in_dim, out_dim = load_autoencoder_data(path=path, device=device)
        model_class = VAE
        loss_function = torch.nn.CrossEntropyLoss()

    if args.model == "pairwise":
        data, in_dim, out_dim = load_pairwise_data(path=path, device=device)
        model_class = Pairwise
        loss_function = torch.nn.BCELoss()

    if args.model == "cost":
        data, in_dim, out_dim = load_costmodel_data(path=path, device=device)
        model_class = CostModel
        loss_function = torch.nn.L1Loss()

    print(
        f"Succesfully loaded data with in_dimensions:{in_dim} out_dimensions:{out_dim}",
        flush=True,
    )

    best_model, x = do_hyperparameter_BO(
        model_class=model_class,
        data=data,
        in_dim=in_dim,
        out_dim=out_dim,
        loss_function=loss_function,
        device=device,
        lr=lr,
        weights=weights,
        epochs=epochs,
        trials=trials,
    )

    # if args.model == 'vae': does not work
    #    latent_space_BO(best_model, device, x)

    model_name = f"{args.model}.onnx" if len(args_name) < 6 else args.name
    export_model(
        model=best_model, x=x, model_name=get_relative_path(model_name, "Models")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vae")
    parser.add_argument("--retrain", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--lr", type=str, default="[1e-6, 1e-3]")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()
    main(args)
