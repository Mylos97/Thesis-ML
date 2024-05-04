import torch
import argparse
import torch.onnx
from exporter import export_model
from OurModels.PairWise.model import Pairwise
from OurModels.CostModel.model import CostModel
from OurModels.EncoderDecoder.model import VAE
from helper import load_autoencoder_data, load_pairwise_data, load_costmodel_data, get_relative_path, get_weights_of_model
from hyperparameterBO import do_hyperparameter_BO
from latentspaceBO import latent_space_BO

def main(args) -> None:
    model_class = None
    loss_function = None
    data = None
    weights = None
    path = get_relative_path('no-co-encodings.txt', 'Data')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.retrain:
        print(f'Retraining model {args.model}')
        weights = get_weights_of_model(args.model)
        path = args.retrain

    if args.model == 'vae':
        data, in_dim, out_dim = load_autoencoder_data(path=path, device=device)
        model_class = VAE
        loss_function = torch.nn.HuberLoss()

    if args.model == 'pairwise':
        data, in_dim, out_dim = load_pairwise_data(path=path, device=device)
        model_class = Pairwise
        loss_function = torch.nn.BCELoss()

    if args.model == 'cost':
        data, in_dim, out_dim = load_costmodel_data(path=path, device=device)
        model_class = CostModel
        loss_function = torch.nn.MSELoss()

    best_model, x = do_hyperparameter_BO(model_class=model_class, data=data, in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device)
    
    if args.model == 'vae':
        latent_space_BO(best_model, device, x)
    
    model_name = f'{args.model}.onnx'
    export_model(model=best_model, x=x, model_name=get_relative_path(model_name, 'Models'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vae")
    parser.add_argument("--retrain", type=str, default="")
    args = parser.parse_args()
    main(args)
