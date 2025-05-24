import torch
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import math


transform=transforms.Compose([
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=1)], p=0.3),
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

train_data=datasets.ImageFolder(root='', transform=transform)# root files for training and testing dataset
test_data=datasets.ImageFolder(root='', transform=transform)

device="cuda" if torch.cuda.is_available() else "cpu"
num_classes=len(train_data.classes)

class Vit(nn.Module):
    def __init__(self, img_size=128, patch_size=8, depth=6, num_classes=len(train_data.classes),
                in_channels=1, dim=128, mlp=512, nheads=4, dropout=0.1):
        super().__init__()
        assert(img_size%patch_size==0), "image size should be divisible by patch size"
        self.patch_size=patch_size
        self.dim=dim
        num_patches=(img_size//patch_size)**2
        self.patch_embed=nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token=nn.Parameter(torch.randn(1,1,dim))
        self.pos_embed=nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.dropout=nn.Dropout(dropout)
        encoder_layer=nn.TransformerEncoderLayer(d_model=dim, nhead=nheads, dim_feedforward=mlp,
                                                      dropout=dropout, activation='gelu', batch_first=True)
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm=nn.LayerNorm(dim)
        self.mlp_head=nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B=x.size(0)
        x=self.patch_embed(x)
        x=x.flatten(2).transpose(1,2)
        cls_token=self.cls_token.expand(B, -1, -1)
        x=torch.cat((cls_token,x), dim=1)
        x=x+self.pos_embed
        x=self.dropout(x)
        x=self.transformer(x)
        cls_out=x[:,0]
        avg_out=x[:,1:].mean(dim=1)
        x=self.norm(cls_out+avg_out)
        return self.mlp_head(x)

if __name__ == "__main__":

    train_loader=torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
    test_Loader=torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
    model=Vit().to(device)
    optimizer=torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss()
    saving_path="brain-tumour.pth"
    start_epoch=0
    epochs=100
    
    torch.cuda.empty_cache()
    if os.path.exists(saving_path):
        checkpoint = torch.load(saving_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch=checkpoint["epoch"]+1

    print("training initiated")
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', ncols=100, colour='blue')
        for batch_idx, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            loop.set_postfix({"loss": loss.item(), "batch": f"{batch_idx+1}/{len(train_loader)}"})
        print(f"Epoch: {epoch+1} completed with a loss of {train_loss/len(train_loader):.4f}")
        torch.save({
            'epoch':epoch,
            'optimizer':optimizer.state_dict(),
            'model':model.state_dict()
        }, saving_path)

    print("Training done, time for testing")
    img_path=""#image path for testing a singular image
    transformen=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    img=Image.open(img_path)
    img=transform(img)
    img=img.unsqueeze(0)
    img=img.to(device)

    with torch.inference_mode():
        output=model(img)
        _,preds=torch.max(output,1)
    print(f"disease: {train_data.classes[preds.item()]}")

    correct = 0
    total = 0
    with torch.inference_mode():
        model.eval()
        testing_loss = 0
        loop = tqdm(enumerate(test_Loader), total=len(test_Loader), desc='Testing', ncols=100, colour='green')
        for batch_idx, (images, labels) in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            testing_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix({"loss": loss.item(), "batch": f"{batch_idx+1}/{len(test_Loader)}"})

    print(f"Testing done\nAccuracy: {correct * 100 / total:.2f}%\nLoss: {testing_loss / len(test_Loader):.4f}")
