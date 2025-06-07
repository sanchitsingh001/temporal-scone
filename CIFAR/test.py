import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.wrn_ssnd import WideResNet  # adjust if your WRN model is in a different path

# # ----------------------------
# # Hyperparameters
# # ----------------------------
# batch_size = 128
# learning_rate = 0.1
# epochs = 100
# seed = 42
# save_dir = 'snapshots/pretrained'
# save_name = 'cinic10_wrn_pretrained_epoch_{}.pt'

# # ----------------------------
# # Dataset (CINIC-10)
# # ----------------------------
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
# ])

# # Replace this path with the actual root of CINIC-10
# cinic_root = '../data/CINIC/'  # should contain train/, valid/, test/ folders

# train_set = datasets.ImageFolder(os.path.join(cinic_root, 'train'), transform=transform)
# val_set = datasets.ImageFolder(os.path.join(cinic_root, 'valid'), transform=transform)

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# # ----------------------------
# # Model
# # ----------------------------
# num_classes = 10
# net = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = net.to(device)

# # ----------------------------
# # Training setup
# # ----------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)

# # ----------------------------
# # Training loop
# # ----------------------------
# os.makedirs(save_dir, exist_ok=True)

# for epoch in range(epochs):
#     net.train()
#     total_loss = 0.0
#     correct = 0
#     total = 0

#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * targets.size(0)
#         _, predicted = outputs.max(1)
#         correct += predicted.eq(targets).sum().item()
#         total += targets.size(0)

#     scheduler.step()

#     acc = 100. * correct / total
#     avg_loss = total_loss / total
#     print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')

#     # Save model checkpoint
#     torch.save(net.state_dict(), os.path.join(save_dir, save_name.format(epoch)))
#     print(f"Checkpoint saved: {save_name.format(epoch)}")

# print("Training complete.")
# ----------------------------
# Load test set
# ----------------------------

# ----------------------------
# Settings
# ----------------------------
model_name = "snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt"
cinic_root = "../data/CINIC/"  # path containing test/ folder
batch_size = 128

# ----------------------------
# Transforms
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ----------------------------
# Load test set
# ----------------------------
test_set = datasets.ImageFolder(os.path.join(cinic_root, 'test'), transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ----------------------------
# Load model
# ----------------------------
num_classes = 10
net = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load(model_name, map_location=device))
net = net.to(device)
net.eval()

# ----------------------------
# Evaluate on test set
# ----------------------------
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

test_acc = 100. * correct / total
print(f"Test Accuracy (epoch 99): {test_acc:.2f}%")
