import torch
import tqdm
def train(x, test, device="cuda", weight_path="./weight.txt"):
    x = x.to(device)
    test = test.to(device)
    weight = torch.zeros((50, 50), device=device)
    for i in range(50):
        weight[i, i] = 1
    mask = torch.ones((50, 50), device=device)
    for i in range(50):
        for j in range(50):
            if j > i:
                mask[i, j] = 0
            else:
                mask[i, j] = 1
    weight = weight * mask
    weight.requires_grad_(True)
    optimizer = torch.optim.Adam([weight], lr=0.01)
    for epoch in tqdm.trange(201):
        pred_y = torch.matmul(weight * mask, x)[:, :-1]
        loss = (pred_y - x[:, 1:]).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 50 == 0:
            print("train loss: ", loss.item(), "epoch: ", epoch)
            with torch.no_grad():
                pred_y = torch.matmul(weight * mask, test)[:, :-1]
                loss = (pred_y - test[:, 1:]).pow(2).mean()
                print("test loss: ", loss.item(), "epoch: ", epoch)
            with open(f"{weight_path}", "w") as f:
                f.write(f"{weight.detach().cpu().numpy().tolist()}\n")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--weight_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

x = torch.load("./train/features.pt")
test = torch.load("./valid/features.pt")
train(x, test, device=args.device, weight_path=args.weight_path)

