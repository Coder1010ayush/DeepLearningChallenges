import torch
import os
import json
import logging
import argparse
from preprocessing import resizeImage
from dataloader import VNetDataClass
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import BaseModel
import tqdm as tqm
from model import ModelArgs
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGGNet model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default='cpu',
                        choices=['cpu', 'cuda'], help="Device to use for training.")
    return parser.parse_args()


def getConfig(model, lr=0.0001, schedular_type="step"):
    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    loss_fn = nn.NLLLoss()

    if schedular_type == "step":
        from torch.optim.lr_scheduler import StepLR
        schedular = StepLR(optimizer, step_size=20, gamma=0.04)
    else:
        schedular = None

    return loss_fn, optimizer, schedular


def train(model, epochs=20, batch_size=16, device='cpu'):
    vbc = VNetDataClass(folder_path="vggnet/sports_images/train")
    train_loader = DataLoader(vbc, batch_size=batch_size, shuffle=True)
    loss_fn, optimizer, schedular = getConfig(model=model)

    model.to(device)

    for epoch in range(epochs):
        logging.info("INFO: Model training starting ...............")
        logging.info("INFO: Epoch is %d", epoch)
        total_loss = 0.0
        start = time.time()
        for step, (image, label) in enumerate(tqm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()

            out = model(image)
            loss = loss_fn(out, label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if schedular is not None:
            schedular.step()

        logging.info("INFO: Total training loss is %.4f", total_loss / len(train_loader))
        validate(model=model, device=device)
        logging.info("INFO: Model is validated ......................")

        end = time.time()
        logging.info("INFO: Total time taken in training is %.2f seconds", end - start)


def validate(model, device='cpu'):
    start = time.time()
    logging.info("INFO: Model evaluation starting ...............")

    vbc = VNetDataClass(folder_path="vggnet/sports_images/test")
    val_loader = DataLoader(vbc, batch_size=16, shuffle=False)
    loss_fn = nn.NLLLoss()

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for step, (image, label) in enumerate(tqm.tqdm(val_loader, desc="Validating")):
            image, label = image.to(device), label.to(device)

            out = model(image)
            loss = loss_fn(out, label)
            total_loss += loss.item()

    end = time.time()
    average_loss = total_loss / len(val_loader)
    logging.info("INFO: Total validation loss is %.4f", average_loss)
    logging.info("INFO: Total time in evaluation is %.2f seconds", end - start)


def singleImageinference():
    pass


def inference(model, data_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, label in data_loader:
            image, label = image.to(device), label.to(device)
            out = model(image)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    logging.info("INFO: Accuracy: %.4f", accuracy)
    logging.info("INFO: F1 Score: %.4f", f1)
    logging.info("INFO: Precision: %.4f", precision)
    logging.info("INFO: Recall: %.4f", recall)


if __name__ == "__main__":
    args = parse_args()

    model = BaseModel(input_size=2187, args=ModelArgs)
    train(model=model, epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    logging.info("INFO: Model is trained .....................")
    torch.save(model.state_dict(), "vggnet.pth")
    # vbc = VNetDataClass(folder_path="vggnet/sports_images/test")
    # val_loader = DataLoader(vbc, batch_size=16, shuffle=False)
    # model.load_state_dict(torch.load("vggnet/model_arch/vggnet.pth"))
    # inference(model, val_loader)
