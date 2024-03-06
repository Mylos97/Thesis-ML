import torch
import os
from helper import make_dataloader

def train(model, loss_function, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.1, 0.1])
    hyper_param = [{"lr": 0.0001, "gradient_norm":1.0, "epochs":100, "batch_size": 32}]
    best_loss = 10000000
    best_model = None

    for p in hyper_param:
        val_loss = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
        train_dataloader = make_dataloader(train_dataset, p["batch_size"], device)
        val_dataloader = make_dataloader(val_dataset, p["batch_size"], device)
        test_dataloader = make_dataloader(test_dataset, p["batch_size"], device)


        model = train_model(model, loss_function, p["epochs"], optimizer, p["gradient_norm"], train_dataloader, test_dataloader)        

        for tree, target in val_dataloader:
            prediction = model(tree)
            loss = loss_function(prediction, target)
            val_loss += loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    return best_model, tree




def train_model(model, loss_function, epochs, optimizer, gradient_norm, train_dataloader, test_dataloader):
    for epoch in range(epochs):
        loss_accum = 0
        test_loss = 0
        
        model.train()
        for tree, target in train_dataloader:
            prediction = model(tree)

            loss = loss_function(prediction, target)
            loss_accum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)
            optimizer.step()

        loss_accum /= len(train_dataloader)
        model.eval()
        for tree, target in test_dataloader:
            prediction = model(tree)

            test_loss = loss_function(prediction, target)
            test_loss += loss.item()

        print("Epoch", epoch, "training loss:", loss_accum, "test loss:", test_loss)
        
    return model