"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import math

# -----------------------------
# Learning Rate Scheduler (Warmup + Cosine Decay)
# -----------------------------
def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """
    Learning rate schedule used in ViT/BERT:
    - Linear warmup
    - Cosine decay to zero
    """
    # Warmup
    if step < warmup_steps:
        return step / warmup_steps

    # Cosine Decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


# Global tracking for scheduler
global_step = 0


# -----------------------------
# Training Step
# -----------------------------
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:

    global global_step
    model.train()

    train_loss = 0
    train_acc = 0

    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=False,
    )

    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)

        # Forward
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (global norm = 1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update the scheduler *every step*
        scheduler.step()
        global_step += 1

        preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        acc = (preds == y).float().mean().item()
        train_acc += acc

        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({
            "step": f"{batch+1}/{len(dataloader)}",
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}",
            "lr": f"{current_lr:.6f}"
        })

    return train_loss / len(dataloader), train_acc / len(dataloader)


# -----------------------------
# Testing Step
# -----------------------------
def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:

    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)
            test_loss += loss.item()

            preds = logits.argmax(dim=1)
            test_acc += (preds == y).float().mean().item()

    return test_loss / len(dataloader), test_acc / len(dataloader)


# -----------------------------
# Training Loop
# -----------------------------
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List]:


    results = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    model.to(device)

    for epoch in (range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            total_epochs=epochs,

        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
