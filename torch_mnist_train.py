import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

EPOCHS = 1
BATCH_SIZE = 128
LR = 1e-3
NUM_CLASSES = 10
TRAIN_SIZE = 60000
VAL_SIZE = 10000

BW_WARMUP = 5
BW_TRIALS = 20
BW_N, BW_M = 500_000, 2000


NIC_MAP = {
    0: "mlx5_5", 1: "mlx5_6", 2: "mlx5_7", 3: "mlx5_8",
    4: "mlx5_9", 5: "mlx5_10", 6: "mlx5_11", 7: "mlx5_12",
}


def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank in NIC_MAP:
        os.environ["NCCL_IB_HCA"] = NIC_MAP[local_rank]
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank


def cleanup():
    dist.destroy_process_group()


def sync_all():
    torch.cuda.synchronize()
    dist.barrier()


def measure_bus_bandwidth(local_rank):
    """Run a timed allreduce to measure bus bandwidth, similar to nccl-tests."""
    n = dist.get_world_size()
    mat = torch.rand(BW_N, BW_M, dtype=torch.float32).cuda(local_rank)
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    sync_all()
    for _ in range(BW_WARMUP):
        dist.all_reduce(mat)
    sync_all()

    start_ev.record()
    for _ in range(BW_TRIALS):
        dist.all_reduce(mat)
    end_ev.record()
    sync_all()

    elapsed_s = start_ev.elapsed_time(end_ev) / 1000.0
    avg_s = elapsed_s / BW_TRIALS
    payload_bytes = BW_M * BW_N * 4
    algbw = payload_bytes / avg_s
    busbw = algbw * (2 * (n - 1) / n)

    if dist.get_rank() == 0:
        print(f"\n=== Allreduce bandwidth ({BW_TRIALS} trials, {n} ranks, "
              f"{payload_bytes / 1e9:.2f} GB payload) ===", flush=True)
        print(f"  algbw: {algbw / 1e9:.3f} GB/s  ({algbw * 8 / 1e9:.1f} Gb/s)", flush=True)
        print(f"  busbw: {busbw / 1e9:.3f} GB/s  ({busbw * 8 / 1e9:.1f} Gb/s)\n", flush=True)

    del mat
    torch.cuda.empty_cache()
    return algbw, busbw


def get_dataset():
    """Synthetic MNIST-shaped data (1x28x28 images, 10 classes).

    Avoids torchvision dependency and network downloads.
    """
    g = torch.Generator().manual_seed(42)
    train_x = torch.randn(TRAIN_SIZE, 1, 28, 28, generator=g)
    train_y = torch.randint(0, NUM_CLASSES, (TRAIN_SIZE,), generator=g)
    val_x = torch.randn(VAL_SIZE, 1, 28, 28, generator=g)
    val_y = torch.randint(0, NUM_CLASSES, (VAL_SIZE,), generator=g)
    return TensorDataset(train_x, train_y), TensorDataset(val_x, val_y)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main():
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Starting training: {world_size} ranks, {EPOCHS} epochs, batch_size={BATCH_SIZE}",
              flush=True)

    measure_bus_bandwidth(local_rank)

    train_set, val_set = get_dataset()
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = ConvNet().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X, y = X.cuda(local_rank), y.cuda(local_rank)
            loss = loss_fn(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if rank == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.cuda(local_rank), y.cuda(local_rank)
                    correct += (model(X).argmax(1) == y).sum().item()
                    total += y.size(0)
            acc = 100.0 * correct / total
            print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_acc={acc:.2f}%",
                  flush=True)

    if rank == 0:
        print("Training complete.", flush=True)

    cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        raise
