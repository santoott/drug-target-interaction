cat << 'EOF' > README.md
# 🧬 Drug-Target Interaction (DTI) Predictor

A Deep Learning system built using **PyTorch**, **RDKit**, and **Streamlit** to predict the binding affinity and interaction between small-molecule drugs and target protein sequences.

## 🌟 Key Features
- **Molecular Feature Extraction**: Converts SMILES strings to 2048-bit Morgan Fingerprints via RDKit.
- **Protein Feature Representation**: Computes normalized Amino Acid Composition (AAC) vectors.
- **Deep Hybrid Network**: Two-stream architecture merging drug and protein representations for binary interaction classification.
- **Interactive UI**: Built with Streamlit to display real-time predictions and 2D chemical structure visualizations.

## 🛠️ Tech Stack
- Python 3
- PyTorch
- RDKit
- Streamlit
- Pandas & NumPy

## 🚀 Quick Start
```bash
git clone [https://github.com/santoott/drug-target-interaction.git](https://github.com/santoott/drug-target-interaction.git)
cd drug-target-interaction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py