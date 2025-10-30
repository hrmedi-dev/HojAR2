import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
data_dir = 'dataset'   # carpeta donde están tus imágenes
batch_size = 16
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Entrenando en: {device}")

# -------------------------------
# TRANSFORMACIONES
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# DATASET Y DATALOADER
# -------------------------------
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Clases detectadas: {dataset.classes}")

# -------------------------------
# MODELO BASE (ResNet50)
# -------------------------------
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # congelamos las capas base

num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# -------------------------------
# OPTIMIZADOR Y PÉRDIDA
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# -------------------------------
# ENTRENAMIENTO
# -------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Época [{epoch+1}/{num_epochs}] - Pérdida: {running_loss:.3f} - Precisión: {acc:.2f}%")

# -------------------------------
# GUARDAR MODELO
# -------------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': dataset.classes
}, 'model/leaf_model.pth')

print("✅ Modelo entrenado y guardado en model/leaf_model.pth")
