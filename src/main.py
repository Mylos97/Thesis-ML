import torch
import argparse
import torch.onnx
from exporter import export_model
from OurModels.PairWise.model import Pairwise
from OurModels.Classifier.model import TreeConvolution256
from OurModels.EncoderDecoder.model import VAE
from helper import load_autoencoder_data, load_pairwise_data, load_classifier_data, get_relative_path, get_weights_of_model
from hyperparameterBO import do_hyperparameter_BO
from latentspaceBO import latent_space_BO

def main(args) -> None:
    model_class = None
    loss_function = None
    data = None
    weights = None
    path = get_relative_path('data.txt', 'Data')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    
    if args.retrain:
        print(f'Retraining model {args.model}')
        weights = get_weights_of_model(args.model)
        path = args.retrain
    
    if args.model == 'vae':
        data, in_dim, out_dim = load_autoencoder_data(device=device, path=path)
        model_class, params = VAE, [in_dim, out_dim]
        loss_function = torch.nn.CrossEntropyLoss()

    if args.model == 'pairwise':
        data, in_dim = load_pairwise_data()
        model_class, params = Pairwise, [in_dim]
        loss_function = None

    if args.model == 'treeconv':
        data, in_dim, out_dim = load_classifier_data()
        model_class, params = TreeConvolution256, [in_dim, out_dim]
        loss_function = None
    
    best_model, x = do_hyperparameter_BO(model_class=model_class, data=data, in_dim=in_dim, out_dim=out_dim, loss_function=loss_function, device=device)
    latent_space_BO(best_model, device, x)
    model_name = f'{args.model}.onnx'
    #export_model(model=best_model, x=x, model_name=get_relative_path(model_name, 'Models'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vae")
    parser.add_argument("--retrain", type=str, default="")
    args = parser.parse_args()
    main(args)