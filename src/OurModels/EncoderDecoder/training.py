import torch
from helper import make_dataloader
from exporter import export_model
from dataloader import load_data
from model import TreeAutoEncoder

epochs = 100
batch_size = 32
lr = 0.0001
data = load_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(load_data, [0.8, 0.1, 0.1])
train_dataloader = make_dataloader(train_dataset, batch_size, device)
val_dataloader = make_dataloader(val_dataset, batch_size, device)
test_dataloader = make_dataloader(test_dataset, batch_size, device)

dim = None # need to find
model = TreeAutoEncoder(dim).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
bce_loss_fn = torch.nn.BCELoss() # We need to find loss function.

losses = []

for epoch in range(epochs):
    loss_accum = 0
    test_loss = 0
    
    model.train()
    for tree, target in train_dataloader:
        prediction = model(tree)

        loss = bce_loss_fn(prediction, target)
        loss_accum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_accum /= len(train_dataloader)
    losses.append(loss_accum)

    model.eval()
    for tree, target in test_dataloader:
        prediction = model(tree)

        test_loss = bce_loss_fn(prediction, target)
        test_loss += loss.item()

    print("Epoch", epoch, "training loss:", loss_accum, "test loss:", test_loss)
    
export_model(model, tree)