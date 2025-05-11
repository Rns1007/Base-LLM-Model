# GPT Language Model from Scratch (PyTorch)

This project implements a simplified GPT-style transformer model using PyTorch. It includes tokenization, dataset preparation, a transformer architecture with multi-head self-attention, and a text generation function.

## 📚 Project Overview

- Tokenization using OpenAI's `tiktoken` GPT-2 tokenizer.
- Dataset creation using a sliding window approach.
- Transformer architecture including:
  - Multi-head self-attention with causal masking
  - Layer normalization and GELU activation
  - Feed-forward neural networks
- Text generation with greedy decoding
- Configurable model architecture (based on GPT-2 124M structure)

## 🚀 Features

- Custom dataset preparation (`GPTDatasetV1`)
- Configurable model via a `GPT_CONFIG_124M` dictionary
- Minimal external dependencies
- GPU-ready PyTorch implementation
- Fully self-contained and customizable

## 🛠 Installation

Make sure you have Python 3.8+ and install the following dependencies:

```bash
pip install torch tiktoken
```
## 📦 File Structure

project/

│

├── Project_1_MAT496.py      # Main implementation and execution notebook

├── README.md                   # Project description

├── Datasets

## 📖 Usage

### 1. Prepare Text
You can load your custom text or dataset and pass it to the dataloader:

dataloader = create_dataloader_v1(txt, batch_size=4, max_length=256)

### 2. Initialize the Model

model = GPTModel(GPT_CONFIG_124M)

### 3. Text Generation
Generate text from a prompt index tensor:

generated = generate_text_simple(model, input_ids, max_new_tokens=50, context_size=1024)

## 📌Configuration

Model configuration is defined in the dictionary:

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}









