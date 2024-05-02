import torch
from torch.utils.data import DataLoader
from torch import Tensor

EPOCHS = 100

def train(model_class, data_loader, in_dim, out_dim , loss_function, device, parameters) -> tuple[torch.nn.Module, tuple[list[Tensor], list[Tensor]]]:
    lr = parameters.get("lr", 0.001)
    gradient_norm = parameters.get("gradient_norm", 1.0)
    dropout = parameters.get("dropout", 0.1)
    model = model_class(in_dim = in_dim,
                        out_dim = out_dim,
                        dropout_prob = dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training model epochs:{EPOCHS} lr:{lr} optimizer:{optimizer.__class__.__name__} gradient norm:{gradient_norm} drop out: {dropout}")
    for epoch in range(EPOCHS):
        loss_accum = 0
        test_loss = 0

        model.train()
        for tree, target in data_loader:

            prediction = model(tree)
            loss = loss_function(prediction, target.float())
            loss_accum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)
            optimizer.step()
        loss_accum /= len(data_loader)

        print("Epoch", epoch, "training loss:", loss_accum, "test loss:", test_loss)

    return model, tree


def evaluate(model: torch.nn.Module, data_loader: DataLoader, loss_function, device: torch.device) -> float:
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for tree, target  in data_loader:
            prediction = model(tree)
            loss = loss_function(prediction, target.float())
            val_loss += loss.item()

    return val_loss
