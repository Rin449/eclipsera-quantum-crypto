# ECLIPSERA™ - Quantum Neural Cryptosystem v9

[![GitHub stars](https://img.shields.io/github/stars/ivan4154_4/Eclipsera?style=social)](https://github.com/Rin449/eclipsera-quantum-crypto)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

> **Warning: NOT a post-quantum cryptosystem**  
> This is an **experimental neural cryptosystem** using adversarial training (Alice-Bob vs Eve).  
> It is **NOT resistant to Shor's or Grover's algorithms** and should **NOT be used for real-world security**.

---

## What is Eclipsera™?

A **GUI-based neural encryption tool** that demonstrates:
- **Adversarial training** in cryptography: Alice & Bob cooperate, Eve tries to eavesdrop.
- **High entropy ciphertext** (~15.9 bits/16-bit block).
- **Low QBER** (~1.5%) for legitimate receiver (Bob).
- **Eve limited to ~40% accuracy** via dropout, noise injection, and penalty terms.

Built with **PyTorch + CustomTkinter**, inspired by **quantum neural networks (QNN)** and **chaos theory** — but **runs on classical hardware**.

---

## Features

| Feature | Description |
|-------|-----------|
| **GUI Encrypt/Decrypt** | Full UTF-8, emoji, Vietnamese support |
| **Base64 + JSON Export** | Copy-paste ready for API integration |
| **Adversarial Security** | Eve trained to fail (≤40% accuracy) |
| **High Entropy** | Ciphertext near-uniform (≥15.9/16 bits) |
| **Fast Inference** | <1s per 1KB on CPU |

---

## Security Reality Check

| Claim | Reality |
|------|--------|
| "Post-quantum secure" | **False** – No mathematical proof, no lattice/Hash-based crypto |
| "Resists Shor's algorithm" | **False** – Runs on classical NN, breakable by key recovery |
| "Quantum-inspired" | **True** – Uses Hadamard-like layers, phase gates, chaos |
| "Eve can't decrypt" | **True in model** – But only because Eve is **weak by design** |

> **Use Case**: Educational, research, red-teaming AI crypto, demo of GAN-like training in security.

---

## Architecture (Inspired, Not Quantum)

```text
Alice:  msg + key + basis → [ResBlocks + Hadamard + Chaos] → cipher
Bob:   cipher + key + basis → [Transformer-like] → msg
Eve:   cipher only → [Weak net + 80% dropout] → guess (fails)

## 📦 Installation
```bash
git clone https://github.com/Rin449/eclipsera-quantum-crypto.git
cd eclipsera-quantum-crypto
pip install -r requirements.txt
python eclipsera.py
