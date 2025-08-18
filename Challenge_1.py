import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_clean = datasets.ImageFolder(
    "/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET/train/clean",
    transform=transform
)
train_noisy = datasets.ImageFolder(
    "/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET/train/noisy",
    transform=transform
)

class PairedDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean
    def __len__(self):
        return len(self.noisy)
    def __getitem__(self, idx):
        return self.noisy[idx][0], self.clean[idx][0]

paired_dataset = PairedDataset(train_noisy, train_clean)
train_loader = DataLoader(paired_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

for epoch in range(10):
    running_loss = 0.0
    for noisy_imgs, clean_imgs in train_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
        outputs = autoencoder(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Autoencoder Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")

torch.save(autoencoder.state_dict(), "denoiser.pth")

test_path = "/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET/test/noisy"
denoised_path = "Denoised_Images/"
os.makedirs(denoised_path, exist_ok=True)

autoencoder.eval()
for img_name in os.listdir(test_path):
    img = Image.open(os.path.join(test_path, img_name)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # prepare input
    with torch.no_grad():
        out = autoencoder(x)
    save_image(out, os.path.join(denoised_path, img_name))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 128 -> 64

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 64 -> 32

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 32 -> 16

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

train_dataset = datasets.ImageFolder(
    "/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET/train/clean",
    transform=transform
)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader_cls = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader_cls = DataLoader(val_subset, batch_size=32, shuffle=False)

model = SimpleCNN(num_classes=5).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader_cls:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion_cls(outputs, labels)
        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    val_correct, val_total = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader_cls:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_acc = 100. * val_correct / val_total

    print(f"Classifier Epoch [{epoch+1}/10], "
          f"Loss: {running_loss/len(train_loader_cls):.4f}, "
          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), "classifier.pth")

test_imgs = os.listdir(denoised_path)
preds = []

model.eval()
with torch.no_grad():
    for img_name in test_imgs:
        img = Image.open(os.path.join(denoised_path, img_name)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)  # prepare input
        output = model(x)
        pred_class = torch.argmax(output, 1).item() + 1  # map to 1–5
        preds.append([img_name, pred_class])

df = pd.DataFrame(preds, columns=["Images", "Predicted_Classes"])
df.to_csv("test_labels.csv", index=False)

psnr_before, psnr_after, ssim_before, ssim_after = [], [], [], []

autoencoder.eval()
for (nimg, _), (cimg, _) in zip(train_noisy, train_clean):
    n = np.array(transforms.ToPILImage()(nimg))
    c = np.array(transforms.ToPILImage()(cimg))

    with torch.no_grad():
        d = autoencoder(nimg.unsqueeze(0).to(device)).cpu().squeeze(0)
    d = d.permute(1, 2, 0).numpy()
    d = (d * 255).astype(np.uint8)

    psnr_before.append(psnr(c, n))
    psnr_after.append(psnr(c, d))
    ssim_before.append(ssim(c, n, channel_axis=2))
    ssim_after.append(ssim(c, d, channel_axis=2))

print("PSNR Before:", np.mean(psnr_before), "After:", np.mean(psnr_after))
print("SSIM Before:", np.mean(ssim_before), "After:", np.mean(ssim_after))


