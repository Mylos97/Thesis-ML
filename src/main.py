import torch
import argparse
from trainer import train
from exporter import export_model
from datetime import datetime
from OurModels.EncoderDecoder.model import TreeAutoEncoder 
from OurModels.PairWise.model import Pairwise
from OurModels.Classifier.model import TreeConvolution256
from helper import load_autoencoder_data, load_pairwise_data, load_classifier_data, get_relative_path

def main(args):
    params = {"class":None, "params":None}
    loss_function = None
    data = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'autoencoder':
        data, in_dim, out_dim = load_autoencoder_data(device=device)
        params["class"], params["params"] = TreeAutoEncoder, [in_dim, out_dim]
        loss_function = torch.nn.CrossEntropyLoss()
    
    if args.model == 'pairwise':
        data, in_dim = load_pairwise_data()
        params["class"], params["params"] = Pairwise, [in_dim, out_dim]
        loss_function = None

    if args.model == 'treeconv':
        data, in_dim, out_dim = load_classifier_data()
        params["class"], params["params"] = TreeConvolution256, [in_dim, out_dim]
        loss_function = None

    best_model, x = train(params=params, loss_function=loss_function, data=data, device=device)
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") if args.save else ""
    model_name = f'{args.model}{current_time}.onnx'
    export_model(model=best_model, x=x, model_name=get_relative_path(model_name, 'Models'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="autoencoder")
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()
    main(args)