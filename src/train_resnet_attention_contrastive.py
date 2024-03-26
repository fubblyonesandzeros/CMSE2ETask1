import torch
from torch import nn
from torch.utils.data import DataLoader
from models import ResNetWithChannelSpatialAttention
from data import PhotonElectronContrastiveDataset
from loss import ContrastiveLoss
from tqdm import tqdm


train = DataLoader(
    PhotonElectronContrastiveDataset("train"),
    batch_size=64,
    shuffle=True,
)

test = DataLoader(
    PhotonElectronContrastiveDataset("test"),
    batch_size=64,
    shuffle=False
)


net = ResNetWithChannelSpatialAttention(
    in_channels=2, 
    starting_channels=16,
    img_height=32,
    img_width=32,
    num_heads=4,
    embed_dim=32,
    depth=8,
    dropout=.2
)

device = "cuda" if torch.cuda.is_available() else "cpu"

net.to(device)
optimiser = torch.optim.Adam(net.parameters(), lr=3e-4)

lossfn = ContrastiveLoss()

train_loss_over_time = []
test_loss_over_time = []

best_test_loss = float("inf")

for epoch in range(50):
    running_train_loss = []
    running_test_loss = []

    net.train()
    for x1, x2, y in tqdm(train):
        net.zero_grad()

        x1 = x1.to(device)
        x2 = x2.to(device)

        y = y.to(device)

        p1 = net(x1)
        p2 = net(x2)

        loss = lossfn(p1, p2, y)
        loss.backward()

        optimiser.step()

        running_train_loss.append(loss.item())

    net.eval()
    with torch.no_grad():
        for x1, x2, y in tqdm(test):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            p1 = net(x1)
            p2 = net(x2)

            loss = lossfn(p1, p2, y)

            running_test_loss.append(loss.item())

    train_loss = sum(running_train_loss)/len(running_train_loss)
    test_loss = sum(running_test_loss)/len(running_test_loss)

    train_loss_over_time.append(train_loss)
    test_loss_over_time.append(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss

        checkpoint = {
            "model": net.state_dict(),
            "epoch": epoch+1,
            "optimiser": optimiser.state_dict(),
        }

        torch.save(checkpoint, "resnet_attn_contrastive_checkpoint.pt")

    performance_checkpoint = {
        "train_loss_over_time": train_loss_over_time,
        "test_loss_over_time": test_loss_over_time,
    }

    torch.save(performance_checkpoint, "resnet_attn_contrastive_performance_checkpoint.pt")