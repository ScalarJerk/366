
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ResNet_CBAM import CBAMClassifier
from datasets.dataset_loader import GestureDataset
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, save_path='/content/drive/MyDrive/research_project/model/ResNet_CBAM_classifier.pth'):
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = running_corrects.double() / total_train

        print(f'Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f}')

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_epoch_loss = val_loss / total_val
        val_epoch_acc = val_corrects.double() / total_val

        print(f'Epoch {epoch}/{num_epochs} - Val Loss: {val_epoch_loss:.4f} - Val Acc: {val_epoch_acc:.4f}')

        # Deep copy the model if validation accuracy improves
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at epoch {epoch} with Val Acc: {best_val_acc:.4f}')

    print('Training complete.')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')

if __name__ == "__main__":
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_CLASSES = 2  # Binary classification based on Correct Label (1 or 2)

    # --- Paths ---
    train_csv = "/content/drive/MyDrive/research_project/train.csv"
    val_csv = "/content/drive/MyDrive/research_project/val.csv"

    # --- Create Dataset and DataLoader ---
    train_dataset = GestureDataset(csv_file=train_csv, transform=None)  # Transforms handled in GestureDataset
    val_dataset = GestureDataset(csv_file=val_csv, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Initialize Model ---
    model = CBAMClassifier(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # --- Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Create output directory if not exists ---
    os.makedirs("../model", exist_ok=True)

    # --- Train the Model ---
    train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE, num_epochs=EPOCHS, save_path='/content/drive/MyDrive/research_project/model/ResNet_CBAM_classifier.pth')
