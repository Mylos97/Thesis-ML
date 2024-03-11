import torch
import argparse
from trainer import train
from exporter import export_model
from datetime import datetime
from OurModels.EncoderDecoder.model import TreeAutoEncoder as autoencoder_model
from helper import build_trees

def main(current_model):
    model = None
    loss_function = None
    data = None

    if current_model == 'autoencoder':
        model = autoencoder_model(dim=10)
        loss_function = torch.nn.BCELoss() 
        data = "Data/autoencoder.txt"
    
    best_model, x = train(model=model, loss_function=loss_function, data=data)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    export_model(model=best_model, x=x, model_name=f'Models/{current_model}:{current_time}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="autoencoder")
    args = parser.parse_args()
    main(current_model=args.model)