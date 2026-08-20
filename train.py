import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data_loader import DTIDataset
from torch.utils.data import DataLoader
from model import DeepDTI

# 1. Dataset حقيقية وموسعة للتجربة والتدريب
def get_extended_dataset():
    data = {
        'SMILES': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',           # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',       # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',       # Ibuprofen
            'CC(=O)NC1=CC=C(O)C=C1',              # Paracetamol
            'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C4)O', # Morphine
            'ClC1=CC=C(C=C1)C(C2=CC=C(Cl)C=C2)C(Cl)(Cl)Cl', # DDT
            'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C', # Testosterone
            'CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C' # Penicillin
        ],
        'Protein_Seq': [
            'MSLSLVTASALCRWRWKNGFGRWLPRAP',
            'MAGALRAGSLAPLLLLGAALLSGWAGGA',
            'MSLSLVTASALCRWRWKNGFGRWLPRAP',
            'MAGALRAGSLAPLLLLGAALLSGWAGGA',
            'MKSILGLACLLLSATVFSSAPVTINL',
            'MAGALRAGSLAPLLLLGAALLSGWAGGA',
            'MKSILGLACLLLSATVFSSAPVTINL',
            'MSLSLVTASALCRWRWKNGFGRWLPRAP'
        ],
        'Label': [1, 0, 1, 0, 1, 0, 1, 1]
    }
    return pd.DataFrame(data)

def train_model():
    print("🚀 Starting Training Pipeline...")
    
    # Prep Data
    df = get_extended_dataset()
    dataset = DTIDataset(df)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize Model, Loss, Optimizer
    model = DeepDTI()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 20
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for drugs, targets, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(drugs, targets)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")
            
    # Save Model Weights
    torch.save(model.state_dict(), "dti_model.pth")
    print("✅ Model trained and saved as 'dti_model.pth'")

if __name__ == "__main__":
    train_model()