import torch
from PairWise.treenetwork import TreeConvolution256
from helper import make_dataloader, build_trees
from exporter import export_model
from dataloader import load_data

epochs = 100
batch_size = 32
lr = 0.0001
data = load_data()
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(load_data, [0.8, 0.1, 0.1])

train_dataloader = make_dataloader(train_dataset, batch_size)
val_dataloader = make_dataloader(val_dataset, batch_size)
test_dataloader = make_dataloader(test_dataset, batch_size)

model = TreeConvolution256(1) # Not correct we need to find the length of each tuple
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
bce_loss_fn = torch.nn.BCELoss() 

losses = []

for epoch in range(epochs):
    loss_accum = 0
    test_loss = 0
    model.train()
    for x, target in train_dataloader:
        tree = build_trees(x)

        pred = model(tree)

        loss = bce_loss_fn(pred, target)
        loss_accum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_accum /= len(train_dataloader)
    losses.append(loss_accum)

    model.eval()
    for x, target in test_dataloader:
        tree = build_trees(x)
        pred = model(tree)

        test_loss = bce_loss_fn(pred, target)
        test_loss += loss.item()


    print("Epoch", epoch, "training loss:", loss_accum, "test loss:", test_loss)
    
export_model(model, x)