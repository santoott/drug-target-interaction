import streamlit as st
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import torch.nn.functional as F

from model import DeepDTI
from data_loader import smiles_to_fingerprint, protein_to_aac

# Page Setup
st.set_page_config(
    page_title="BioAI: Drug-Target Interaction Predictor",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Drug-Target Interaction (DTI) Predictor")
st.markdown("""
Predict binding interaction between **Small Molecule Drugs** (SMILES) and **Target Proteins** (Amino Acid Sequences) using Deep Neural Networks.
""")

st.sidebar.header("⚙️ Configuration")
model_path = st.sidebar.text_input("Model Weights", "dti_model.pth")

# Load Model
@st.cache_resource
def load_trained_model(path):
    model = DeepDTI()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

try:
    model = load_trained_model(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Input UI Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("💊 Drug Molecule Input")
    smiles_input = st.text_input(
        "SMILES String", 
        "CC(=O)OC1=CC=CC=C1C(=O)O"  # Default: Aspirin
    )
    
    # Render Molecule 2D Structure
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is not None:
        img = Draw.MolToImage(mol, size=(300, 200))
        st.image(img, caption="2D Chemical Structure")
    else:
        st.warning("Invalid SMILES String!")

with col2:
    st.subheader("🎯 Target Protein Input")
    protein_input = st.text_area(
        "Amino Acid Sequence", 
        "MSLSLVTASALCRWRWKNGFGRWLPRAP",
        height=180
    )

st.markdown("---")

# Prediction Trigger
if st.button("🚀 Predict Interaction", type="primary"):
    if mol is None:
        st.error("Please enter a valid SMILES string.")
    elif not protein_input.strip():
        st.error("Please enter a valid Protein sequence.")
    else:
        # Preprocess Inputs
        d_fp = smiles_to_fingerprint(smiles_input)
        p_aac = protein_to_aac(protein_input.strip().upper())
        
        if d_fp is None or p_aac is None:
            st.error("Error processing input features.")
        else:
            d_tensor = torch.tensor(d_fp).unsqueeze(0)
            p_tensor = torch.tensor(p_aac).unsqueeze(0)
            
            # Forward Pass
            with torch.no_grad():
                logit = model(d_tensor, p_tensor)
                prob = torch.sigmoid(logit).item()
            
            # Display Results
            st.subheader("📊 Prediction Results")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prob >= 0.5:
                    st.success(f"### 🎯 Strong Interaction Predicted!")
                else:
                    st.error(f"### ❌ No Interaction Predicted")
            
            with res_col2:
                st.metric(label="Binding Probability", value=f"{prob * 100:.2f}%")
                st.progress(prob)