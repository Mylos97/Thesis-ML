import torch
from helper import make_dataloader
from itertools import product

def train(model, loss_function, data):
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [0.8, 0.1, 0.1])
    lrs = [0.0001, 0.001]
    batch_sizes = [32, 64]
    gradient_norms = [1.0, 3.0, 5.0]
    best_loss = float('inf')
    epochs = 100
    best_model = None

    for lr, batch_size, gradient_norm in list(product(lrs, batch_sizes, gradient_norms)):
        val_loss = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_dataloader = make_dataloader(x=train_dataset, batch_size=8)
        val_dataloader = make_dataloader(x=val_dataset, batch_size=8)
        test_dataloader = make_dataloader(x=test_dataset, batch_size=8)
        print(f"Starting training model epochs:{epochs} lr:{lr} batch size:{batch_size} optimizer:{optimizer.__class__.__name__} gradient norm:{gradient_norm}")
        model = train_model(model, loss_function, epochs, optimizer, gradient_norm, train_dataloader, test_dataloader)        

        model.eval()
        for tree, target in val_dataloader:
            prediction = model(tree)
            loss = loss_function(prediction, target)
            val_loss += loss.item()

        with torch.no_grad():
            if val_loss < best_loss:
                print(f'New best model found with validation loss:{val_loss}')
                best_loss = val_loss
                best_model = model

    return best_model, tree

def train_model(model, loss_function, epochs, optimizer, gradient_norm, train_dataloader, test_dataloader):
    best_model = None
    best_loss = 10000
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
            loss = loss_function(prediction, target)
            test_loss += loss.item()
        test_loss /= len(train_dataloader)
        if test_loss < best_loss:
            best_model = model
            best_loss = test_loss
        print("Epoch", epoch, "training loss:", loss_accum, "test loss:", test_loss)
        
    return best_model