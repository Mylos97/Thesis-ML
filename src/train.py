import torch
import sys
import math
from torch.utils.data import DataLoader
from torch import Tensor
from helper import set_weights
from datetime import datetime
from OurModels.EncoderDecoder.model import VAE
from OurModels.EncoderDecoder.bvae import BVAE
from annealing import Annealer

def train(
    model_class,
    training_data_loader,
    val_data_loader,
    test_data_loader,
    in_dim,
    out_dim,
    loss_function,
    device,
    parameters,
    epochs,
    weights=None,
) -> tuple[torch.nn.Module, tuple[list[Tensor], list[Tensor]]]:
    lr = parameters.get("lr", 0.001)
    gradient_norm = parameters.get("gradient_norm", 1.0)
    dropout = parameters.get("dropout", 0.1)
    z_dim = parameters.get("z_dim", 128)
    weight_decay = parameters.get("weight_decay", 0.001) #bounds between 0, 0.1
    batch_size = parameters.get("batch_size" , 1)
    model = model_class(
        in_dim=in_dim, out_dim=out_dim, dropout_prob=dropout, z_dim=z_dim
    )

    if weights:
        set_weights(weights=weights, model=model, device=device)

    model.to(device)

    if isinstance(model, VAE) or isinstance(model, BVAE):
        print("Setting model to training mode", flush=True)
        model.training = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float(sys.maxsize)
    best_test_loss = float(sys.maxsize)
    counter = 0
    patience = parameters.get("patience", 10)
    time = datetime.now().strftime("%H:%M:%S")
    annealing_agent = Annealer(epochs, shape='linear')

    print(
        f"Starting training model epochs:{epochs} training samples: {len(training_data_loader)} lr:{lr} optimizer:{optimizer.__class__.__name__} gradient norm:{gradient_norm} drop out: {dropout} patience: {patience} at {time}",
        flush=True,
    )
    for epoch in range(epochs):
        loss_accum = 0
        model.train()

        for tree, target in training_data_loader:
            torch.set_printoptions(profile="full")
            prediction = model(tree)
            log_target = torch.log(target + 1)
            loss = loss_function(prediction, log_target.float())
            loss_accum += loss.item()
            #loss_accum += loss["loss"].item()
            loss_accum = min(loss_accum, sys.maxsize)
            #kld = annealing_agent(loss["kld"])
            optimizer.zero_grad()
            #(loss["loss"] + loss["kld"]).backward()
            #loss["loss"].backward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)
            optimizer.step()
            annealing_agent.step()
            # print(f'Sampled prediction {prediction[0]}:{prediction[0].shape} and target {target[0]}:{target[0].shape}', flush=True)

        assert len(training_data_loader) != 0
        #loss_accum /= len(training_data_loader)
        loss_accum /= batch_size

        print(f"Epoch  {epoch} training loss: {loss_accum}", flush=True)
        val_loss = evaluate(
            model=model,
            val_data_loader=val_data_loader,
            loss_function=loss_function,
            device=device,
            batch_size=batch_size,
        )

        counter += 1
        if val_loss["loss"] < best_val_loss:
            print(
                f"Got better validation loss {val_loss['loss']} than {best_val_loss}",
                flush=True,
            )
            best_val_loss = val_loss["loss"]
            counter = 0

        if counter > patience:
            print(
                f"Early stopping on Epoch {epoch} training loss: {loss} validation loss: {val_loss} model has not improved for {patience} epochs",
                flush=True,
            )
            break

    # measure models inference performance with test data
    test_loss_accum = 0
    model.train()

    for tree, target in test_data_loader:
        prediction = model(tree)
        log_target = torch.log(target + 1)
        loss = loss_function(prediction, log_target.float())
        test_loss_accum += loss.item()
        #test_loss_accum += loss["loss"].item()
        test_loss_accum = min(test_loss_accum, sys.maxsize)
        optimizer.zero_grad()
        loss.backward()
        #loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)
        optimizer.step()
        # print(f'Sampled prediction {prediction[0]}:{prediction[0].shape} and target {target[0]}:{target[0].shape}', flush=True)

    assert len(test_data_loader) != 0
    #test_loss_accum /= len(test_data_loader)
    test_loss_accum /= batch_size

    print(f"Test loss: {test_loss_accum}", flush=True)
    test_loss = evaluate(
        model=model,
        val_data_loader=test_data_loader,
        loss_function=loss_function,
        device=device,
        batch_size=batch_size,
    )

    if test_loss["loss"] < best_test_loss:
        print(
            f"Got better test loss {test_loss['loss']} than {best_test_loss}",
            flush=True,
        )
        best_test_loss = test_loss["loss"]

    return model, tree


def evaluate(
    model: torch.nn.Module,
    val_data_loader: DataLoader,
    loss_function,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    val_loss = 0
    val_kld = 0
    if isinstance(model, VAE) or isinstance(model, BVAE):
        model.training = True

    for tree, target in val_data_loader:
        prediction = model(tree)
        log_target = torch.log(target + 1)
        loss = loss_function(prediction, log_target.float())
        #if math.isnan(loss['loss'].item()):
        if math.isnan(loss.item()):
            val_loss = sys.maxsize
        else:
            #val_loss += min(loss["loss"].item(), sys.maxsize)
            val_loss += loss.item()
            val_loss = min(val_loss, sys.maxsize)
        #loss["loss"].backward()
        loss.backward()
        #val_kld += loss["kld"].item()

    assert len(val_data_loader) != 0
    #val_loss /= len(val_data_loader)
    val_loss /= batch_size
    #val_kld /= len(val_data_loader)
    val_loss = min(val_loss, sys.maxsize)

    return {
        "loss": val_loss,
        #"kld": val_kld
    }
