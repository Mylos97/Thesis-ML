import torch
from torch.utils.data import DataLoader
from torch import Tensor
from helper import set_weights
from datetime import datetime

EPOCHS = 2

def train(model_class, training_data_loader, val_data_loader, in_dim, out_dim , loss_function, device, parameters, weights=None) -> tuple[torch.nn.Module, tuple[list[Tensor], list[Tensor]]]:
    lr = parameters.get('lr', 0.001)
    gradient_norm = parameters.get('gradient_norm', 1.0)
    dropout = parameters.get('dropout', 0.1)
    model = model_class(in_dim = in_dim,
                        out_dim = out_dim,
                        dropout_prob = dropout)
    if weights:
        set_weights(weights=weights, model=model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    counter = 0
    patience = parameters.get('patience', 10)
    time = datetime.now().strftime("%H:%M:%S")

    print(f'Starting training model epochs:{EPOCHS} training samples: {len(training_data_loader)} lr:{lr} optimizer:{optimizer.__class__.__name__} gradient norm:{gradient_norm} drop out: {dropout} patience: {patience} at {time}', flush=True)
    for epoch in range(EPOCHS):
        loss_accum = 0
        model.train()

        for tree, target in training_data_loader:
            prediction = model(tree)
            loss = loss_function(prediction, target.float())
            if torch.isnan(prediction).any():
                print('Found nan values exiting', flush=True)
                break

            loss_accum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)
            optimizer.step()

        loss_accum /= len(training_data_loader)

        print(f'Sampled prediction {prediction[0]} and target {target[0]}', flush=True)
        print(f'Epoch  {epoch} training loss: {loss_accum}', flush=True)
        val_loss = evaluate(model=model, val_data_loader=val_data_loader, loss_function=loss_function, device=device)

        counter += 1
        if val_loss < best_val_loss:
            print(f'Got better validation loss {val_loss} than {best_val_loss}', flush=True)
            best_val_loss = val_loss
            counter = 0

        if counter > patience:
            print(f'Early stopping on Epoch {epoch} training loss: {loss} validation loss: {val_loss} model has not improved for {patience} epochs', flush=True)
            break

    return model, tree


def evaluate(model: torch.nn.Module, val_data_loader: DataLoader, loss_function, device: torch.device) -> float:
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for tree, target in val_data_loader:
            prediction = model(tree)
            loss = loss_function(prediction, target.float())
            val_loss += loss.item()

    val_loss /= len(val_data_loader)

    return val_loss
