# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch import optim
import copy
import numpy as np

# %%
print(torch.__version__)

# %%
print(torch.cuda.is_available())  # It will be true if gpu is usable
print(torch.cuda.get_device_name(0))

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# %%
# Transform for training and testing
# The images are resized to 150x150 pixels and converted to tensors
transform_train = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])


# %%
# Dataset preparation

def get_data_loaders(batch_size, num_workers=2):
    train_dataset = datasets.ImageFolder("food11/train", transform=transform_train)
    val_dataset = datasets.ImageFolder("food11/validation", transform=transform_test)
    test_dataset = datasets.ImageFolder("food11/test", transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# %%
# Model definition
# The model is a convolutional neural network with 5 convolutional blocks, each followed by batch normalization and ReLU activation functions.
# After the convolutional layers, there is a dropout layer to reduce overfitting, followed by an adaptive average pooling layer and two fully connected layers.
# The final output layer has 11 units, corresponding to the 11 classes in the dataset.
# The model uses ReLU activation functions and dropout for regularization.
# The model is designed to classify images into 11 different classes, which are food categories in this case.

class CNN(nn.Module):
    def __init__(self, num_classes=11):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 150 -> 75
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 75 -> 37
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 37 -> 18
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 18 -> 9
            
        )

        self.dropout = nn.Dropout(0.5)  # Dropout layer after conv layers for reducing overfitting


        # Adaptive Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),             # 512 * 1 * 1 = 512
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.dropout(x)          # Dropout again
        x = self.gap(x)
        x = self.fc(x)
        return x

# %%
def train_model(model, train_loader, val_loader, epochs, learning_rate, save_path, device):
    model.train()  # select training mode

    criterion = nn.CrossEntropyLoss()  # loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Adam optimizer, more advanced but leads to overfitting. 

    # SGD optimizer, more basic but less overfitting
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)


    train_acc_list = []
    val_acc_list = []
    best_val_acc = 0

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # load images and labels to GPU

            optimizer.zero_grad()  # resetting gradients to zero
            outputs = model(images)  # model prediction
            loss = criterion(outputs, labels)  # calculate loss
            loss.backward()  # backpropagation
            optimizer.step()  # update weights

            # calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train
        train_acc_list.append(train_acc)

        # validation accuracy
        correct_val = 0
        total_val = 0
        model.eval()  # evaluation mode
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # load images and labels to GPU
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val
        val_acc_list.append(val_acc)

        # Record the best accuracy model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)  # Record the best model

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return train_acc_list, val_acc_list, best_val_acc


# %%
# Main function to run the training and evaluation

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


# %%
# Testing and evaluation
def test_model(model_path="model.pth", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # just take the test loader
    _, _, test_loader =  get_data_loaders(batch_size, num_workers=2)

    model = CNN(num_classes=11)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Test Accuracy: {acc:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[
        'apple_pie', 'cheesecake', 'chicken_curry', 'french_fries',
        'fried_rice', 'hamburger', 'hot_dog', 'ice_cream',
        'omelette', 'pizza', 'sushi'
    ]))


# %%
# Main function to run the training and evaluation
# def main():
def main(): 

    learning_rates = [0.001, 0.005, 0.01]
    batch_sizes = [32, 64]
    num_epochs = 50

    results = []

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_size in batch_sizes:
        for lr in learning_rates:
            print(f"\n Training with batch size={batch_size}, learning rate={lr}")

            # load data
            train_loader, val_loader, _ = get_data_loaders(batch_size, num_workers=2)

            # create model
            model = CNN(num_classes=11)
            model.to(device)

            # start training
            save_path = f"model_bs{batch_size}_lr{lr}.pth"
            train_acc_list, val_acc_list, best_val_acc = train_model(
                model,
                train_loader,
                val_loader,
                epochs=num_epochs,
                learning_rate=lr,
                save_path=save_path,
                device=device
            )

            # keep track of results
            results.append({
                "batch_size": batch_size,
                "learning_rate": lr,
                "train_acc_list": train_acc_list,
                "val_acc_list": val_acc_list,
                "best_val_acc": best_val_acc
            })

if __name__ == "__main__":
    main()


# %%
def plot_results(epochs, train_acc, val_acc, title):
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
test_model(model_path="model_bs64_lr0.005.pth", batch_size=64)


# %%
# Residual Block for ResNet-like architecture
# The residual block is a building block for ResNet architectures. It allows gradients to flow through the network more easily, which can help with training deeper networks.
# It consists of two convolutional layers with batch normalization and ReLU activation, along with a skip connection that adds the input to the output of the block.

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output channels are different, we need to adjust the dimensions using a 1x1 convolution
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# %%
# Residual CNN Model
# The ResidualCNN class is a convolutional neural network that uses residual blocks to improve training and performance.
# It consists of several convolutional layers followed by residual blocks, batch normalization, ReLU activation, and max pooling layers.
# The final layers include an adaptive average pooling layer and two fully connected layers with dropout for regularization.

class ResidualCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(ResidualCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res_block1 = ResidualBlock(64, 64)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res_block2 = ResidualBlock(128, 128)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.res_block1(x)
        x = self.layer3(x)
        x = self.res_block2(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


# %%
def RunResidualCNN():
    learning_rates = [0.001, 0.005, 0.01]
    batch_sizes = [32, 64]
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    print(f"\n Training model: RESIDUAL CNN\n")

    for batch_size in batch_sizes:
        for lr in learning_rates:
            print(f"\n Batch Size={batch_size},  Learning Rate={lr}")

            # Data loaders
            train_loader, val_loader, _ = get_data_loaders(batch_size, num_workers=2)

            # Model creation
            model = ResidualCNN(num_classes=11)
            model.to(device)

            save_path = f"residual_bs{batch_size}_lr{lr}.pth"

            # Training
            train_acc, val_acc, best_val_acc = train_model(
                model,
                train_loader,
                val_loader,
                epochs=num_epochs,
                learning_rate=lr,
                save_path=save_path,
                device=device
            )

            # Store results
            results.append({
                "model": "residual",
                "batch_size": batch_size,
                "learning_rate": lr,
                "train_acc_list": train_acc,
                "val_acc_list": val_acc,
                "best_val_acc": best_val_acc
            })



# %%
RunResidualCNN()

# %%
# Testing and evaluation for ResidualCNN
def test_residual(model_path="residual_bs16_lr0.005.pth", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # just take the test loader
    _, _, test_loader = get_data_loaders(batch_size, num_workers=2)

    # Use ResidualCNN model instead of CNN
    model = ResidualCNN(num_classes=11)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Test Accuracy: {acc:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[
        'apple_pie', 'cheesecake', 'chicken_curry', 'french_fries',
        'fried_rice', 'hamburger', 'hot_dog', 'ice_cream',
        'omelette', 'pizza', 'sushi'
    ]))


# %%
# Running the best model accoring to the validation accuracy
test_residual(model_path="residual_bs32_lr0.001.pth", batch_size=32)


# %% [markdown]
# PART 2 - TRANSFER LEARNING

# %%
def RunTransferLearning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rates = [0.001, 0.005, 0.01] # Learning rates to test
    batch_sizes = [32, 64] # Batch sizes to test
    num_epochs = 50  # 50 epochs for training

    results = []
    # 2 cases: fc_only and fc_and_last2conv
    # fc_only: only the fully connected layer is trained
    # fc_and_last2conv: the last two convolutional layers are also trained
    # The rest of the layers are frozen (not trained)
    for case in ["fc_only", "fc_and_last2conv"]:
        print(f"\nTraining Case: {case.upper()}")

        for batch_size in batch_sizes:
            for lr in learning_rates:
                print(f"\nBatch Size={batch_size},  Learning Rate={lr}")

                # Load data
                train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size, num_workers=2)

                # Load pre-trained model
                model = models.mobilenet_v2(pretrained=True)

                # Freeze all layers first
                for param in model.parameters():
                    param.requires_grad = False

                # Unfreeze depending on case
                if case == "fc_and_last2conv":
                    for param in model.features[-1].parameters():
                        param.requires_grad = True
                    for param in model.features[-2].parameters():
                        param.requires_grad = True

                # Replace FC layer
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, 11)

                # FC layer requires grad
                for param in model.classifier[1].parameters():
                    param.requires_grad = True

                model = model.to(device)

                # Optimizer
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                criterion = nn.CrossEntropyLoss()

                best_val_acc = 0
                train_acc_list = []
                val_acc_list = []

                # Training loop
                for epoch in range(num_epochs):
                    model.train()
                    correct_train = 0
                    total_train = 0

                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        _, predicted = torch.max(outputs, 1)
                        total_train += labels.size(0)
                        correct_train += (predicted == labels).sum().item()

                    train_acc = correct_train / total_train
                    train_acc_list.append(train_acc)

                    # Validation
                    val_acc = evaluate_model(model, val_loader, device)
                    val_acc_list.append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), f"{case}_bs{batch_size}_lr{lr}.pth")

                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

                # Test accuracy + confusion matrix
                test_acc, y_true, y_pred = test_model(model, test_loader, device)
                print(f"Test Accuracy: {test_acc:.4f}")

                # Confusion matrix plot
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix ({case}, BS={batch_size}, LR={lr})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.show()

                # Store results
                results.append({
                    "case": case,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "train_acc_list": train_acc_list,
                    "val_acc_list": val_acc_list,
                    "test_acc": test_acc
                })

    return results


# %%
#  Evaluation helper
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total



# %%
#  Test helper
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds


# %%
results = RunTransferLearning()



