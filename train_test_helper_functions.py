import torch
from torch import nn
from torch.utils.data import DataLoader , random_split
import torch.nn.functional as F
import numpy as np


def train_step(model, train_data, loss_function, optimizer, device):
    running_loss = 0
    for X, y in train_data:
        X, y = X.to(device), y.to(device)

        # Predict
        y_pred = model(X)

        # Loss calculation
        loss = loss_function(y_pred, y)
        running_loss += loss.item() * X.size(0)

        # Optimize params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss /= len(train_data.dataset)
    return running_loss

def validation_step(model, validation_data, loss_function, device):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for X_val, y_val in validation_data:
            X_val, y_val = X_val.to(device), y_val.to(device)

            y_pred = model(X_val)
            loss = loss_function(y_pred, y_val)
            running_loss += loss.item() * X_val.size(0)

    running_loss /= len(validation_data.dataset)
    return running_loss

def fit(model, train_data, loss_function, optimizer, epochs, device, validation_data=None, earlyStopping=False, patience=3):
    train_loss_epoch = []
    val_loss_epoch = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        avg_train_loss = train_step(model, train_data, loss_function, optimizer, device)
        train_loss_epoch.append(avg_train_loss)
        verbose_string = f"Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.5f}"

        if validation_data:
            avg_val_loss = validation_step(model, validation_data, loss_function, device)
            val_loss_epoch.append(avg_val_loss)
            verbose_string += f" | Validation Loss: {avg_val_loss:.5f}"

            if earlyStopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(verbose_string)
                        print("Early stopping")
                        break

        print(verbose_string)

    return train_loss_epoch, val_loss_epoch

def evaluate(model, test_data, loss_function, device):
    model.eval()
    correct, total = 0, 0
    running_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_data:
            X_test, y_test = X_test.to(device), y_test.to(device)

            output = model(X_test)

            loss = loss_function(output, y_test)
            running_loss += loss.item() * X_test.size(0)

            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, dim=1)

            if y_test.ndim == 2:
                y_test = torch.argmax(y_test, dim=1)

            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

    total_loss = running_loss / len(test_data.dataset)
    accuracy = correct / total
    return accuracy, total_loss
