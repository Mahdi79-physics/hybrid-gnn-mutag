# 🧠 Hybrid GIN + GATv2 Graph Classification

This repository implements a **Hybrid Graph Neural Network** combining **GIN** (Graph Isomorphism Network) and **GATv2** (Graph Attention Network v2) for **graph classification** on the **MUTAG dataset**.

The model also computes **subgraph Lovász numbers** for enhanced graph-level representations.

---

## 📊 Figures & Results

*(Figures will be generated automatically during training and saved in the `figures/` folder.)*

- Train & validation loss per fold  
- Train & validation accuracy per fold

---

## 🧪 Scripts

- `scripts/train_mutag.py` → Train the hybrid GNN with **10-fold cross-validation** and generate performance figures

---

## 🧠 Theory / Model

- **GINEncoder**: Graph Isomorphism Network with dropout & batch normalization  
- **GATv2Encoder**: Graph Attention Network v2  
- **Hybrid Model**: Concatenates GIN and GATv2 node embeddings, pools to graph-level representation, adds subgraph Lovász number, then classifies  

**Loss:** Binary cross-entropy with Lovász regularization

---

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
