import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import timedelta


class MiniModel(nn.Module):
    def __init__(self, n_ent=200, n_rel=10, dim=16):
        super().__init__()
        self.ent_emb = nn.Embedding(n_ent, dim)
        self.rel_emb = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, h, t, r):
        return torch.sum((self.ent_emb(h) + self.rel_emb(r)) * self.ent_emb(t), dim=1)


def train_mini():
    device = torch.device('cpu')
    print("Using CPU for stability")

    # 极小数据集
    n_ent, n_rel = 100, 5
    train_data = []
    for _ in range(500):
        train_data.append((
            np.random.randint(0, n_ent),
            np.random.randint(0, n_rel),
            np.random.randint(0, n_ent)
        ))

    model = MiniModel(n_ent, n_rel, dim=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining on {len(train_data)} triples")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    start_time = time.time()

    for epoch in range(5):
        epoch_loss = 0
        for i, (h, r, t) in enumerate(train_data):
            h_t = torch.tensor([h], dtype=torch.long).to(device)
            r_t = torch.tensor([r], dtype=torch.long).to(device)
            t_t = torch.tensor([t], dtype=torch.long).to(device)

            pos_score = model(h_t, t_t, r_t)
            neg_t = torch.tensor([np.random.randint(0, n_ent)], dtype=torch.long).to(device)
            neg_score = model(h_t, neg_t, r_t)

            loss = F.relu(1.0 + neg_score - pos_score).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

            # 显示进度
            if i % 50 == 0:
                elapsed = time.time() - start_time
                eta = (len(train_data) - i) * (elapsed / (i + 1))
                print(f"\rEpoch {epoch + 1} | Batch {i}/{len(train_data)} | "
                      f"Loss: {loss.item():.4f} | ETA: {timedelta(seconds=int(eta))}", end='')

        print(f"\nEpoch {epoch + 1} avg loss: {epoch_loss / len(train_data):.4f}")

    print(f"\nTotal time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    train_mini()