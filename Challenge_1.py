import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms,models
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr,structural_similarity as ssim
from PIL import Image


class denoisingautoencoder(nn.Module):
    def __init__(self):
        super(denoisingautoencoder,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
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
    transforms.Resize((128,128)),   
    transforms.ToTensor()           
])
train_clean = datasets.ImageFolder("/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET/train/clean", transform=transform)
train_noisy = datasets.ImageFolder("/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET/train/noisy", transform=transform)

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
autoencoder = denoisingautoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

for epoch in range(10): # adjust epochs
    for noisy_imgs, clean_imgs in train_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
        outputs = autoencoder(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

torch.save(autoencoder.state_dict(), "denoiser.pth")

test_path = "/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET/test/noisy"
denoised_path = "Denoised_Images/"
os.makedirs(denoised_path, exist_ok=True)

autoencoder.eval()
for img_name in os.listdir(test_path):
    img = Image.open(os.path.join(test_path, img_name)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = autoencoder(x)
    save_image(out, os.path.join(denoised_path, img_name))

train_dataset = datasets.ImageFolder("/kaggle/input/180-dc-ml-sig-recruitment/REC_DATASET/train/clean", transform=transform)
train_loader_cls = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes
model = model.to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5): # adjust epochs
    for imgs, labels in train_loader_cls:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion_cls(outputs, labels)
        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()
    print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "classifier.pth")

test_imgs = os.listdir(denoised_path)
preds = []

model.eval()
with torch.no_grad():
    for img_name in test_imgs:
        img = Image.open(os.path.join(denoised_path, img_name)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        output = model(x)
        pred_class = torch.argmax(output, 1).item() + 1  # mapping 1–5
        preds.append([img_name, pred_class])

df = pd.DataFrame(preds, columns=["Images", "Predicted_Classes"])
df.to_csv("test_labels.csv", index=False)

clean_path = "train/clean/"
noisy_path = "train/noisy/"


psnr_before, psnr_after, ssim_before, ssim_after = [], [], [], []
for (nimg, _), (cimg, _) in zip(train_noisy, train_clean):
    n = np.array(transforms.ToPILImage()(nimg))
    c = np.array(transforms.ToPILImage()(cimg))
    d = autoencoder(nimg.unsqueeze(0).to(device)).detach().cpu().squeeze(0).permute(1,2,0).numpy()
    d = (d*255).astype(np.uint8)

    psnr_before.append(psnr(c, n))
    psnr_after.append(psnr(c, d))
    ssim_before.append(ssim(c, n, channel_axis=2))
    ssim_after.append(ssim(c, d, channel_axis=2))

print("PSNR Before:", np.mean(psnr_before), "After:", np.mean(psnr_after))
print("SSIM Before:", np.mean(ssim_before), "After:", np.mean(ssim_after))

