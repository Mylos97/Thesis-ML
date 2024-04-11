import torch
import argparse
from trainer import train
from exporter import export_model
from OurModels.PairWise.model import Pairwise
from OurModels.Classifier.model import TreeConvolution256
from OurModels.EncoderDecoder.vae import VAE
from helper import load_autoencoder_data, load_pairwise_data, load_classifier_data, get_relative_path, get_weights_of_model
import torch.onnx

def main(args):
    model_class = None
    params = None
    loss_function = None
    data = None
    weights = None
    path = get_relative_path(f'{args.model}.txt', 'Data')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.retrain:
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
    

    best_model, x = train(model_class=model_class, params=params, loss_function=loss_function, data=data, device=device, weights=weights)
    model_name = f'{args.model}.onnx'
    export_model(model=best_model, x=x, model_name=get_relative_path(model_name, 'Models'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vae")
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--retrain", type=str, default="")
    args = parser.parse_args()
    main(args)