# 🧠 Hybrid GIN + GATv2 Graph Classification

This repository implements a **Hybrid Graph Neural Network** combining **GIN** (Graph Isomorphism Network) and **GATv2** (Graph Attention Network v2) for **graph classification** on the **MUTAG dataset**.

The model also computes **subgraph Lovász numbers** for enhanced graph-level representations.

---

### 📊 Training Results

**10-Fold Cross-Validation**

| Metric | Value |
|--------|-------|
| Avg Train Accuracy | 0.9019 |
| Avg Validation Accuracy | 0.8675 |
| Avg Train Loss | 0.2255 |
| Avg Validation Loss | 0.3519 |
## 🧪 Scripts

- `scripts/train_mutag.py` → Train the hybrid GNN with **10-fold cross-validation**

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
