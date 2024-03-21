import torch
import argparse
from trainer import train
from exporter import export_model
from datetime import datetime
from OurModels.EncoderDecoder.model import TreeAutoEncoder as autoencoder_model
from helper import load_autoencoder_data, get_relative_path

def main(current_model):
    model = None
    loss_function = None
    data = None
    model_class = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if current_model == 'autoencoder':
        data, in_dim, out_dim = load_autoencoder_data(device=device)
        model = autoencoder_model(in_dim=in_dim, out_dim=out_dim, dropout_prob=0.1) # number of elements per tuple and number of platforms
        model_class = autoencoder_model
        loss_function = torch.nn.CrossEntropyLoss()

    best_model, x = train(model=model, loss_function=loss_function, data=data, device=device, model_class=model_class)
    #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") remove this when we are training the model
    #model_name = f'{current_model}:{current_time}'
    model_name = f'{current_model}'
    export_model(model=best_model, x=x, model_name=get_relative_path(model_name, 'Models'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="autoencoder")
    args = parser.parse_args()
    main(current_model=args.model)