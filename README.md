# GPT Model Implementation from Scratch (PyTorch)

An end-to-end implementation of a Generative Pre-trained Transformer (GPT) model from scratch using PyTorch. This project is structured as a comprehensive educational resource that builds the entire transformer architecture, sets up a training environment, and includes a utility to load official pre-trained OpenAI GPT-2 weights for immediate coherent text generation.

## Features

This repository covers the complete implementation of a language model following the GPT-2 architecture:

### Model Components
* **Multi-Head Self-Attention** (`MultiHeadAttention` class): Implements **causal (masked) self-attention**, essential for generative models.
* **Layer Normalization** (`LayerNorm` class): A custom implementation for stabilizing the training process, typically used in pre-normalization layers.
* **Feed-Forward Network** (`FeedForward` class): Includes the **GELU** activation function.
* **Transformer Block** (`TransformerBlock` class): Encapsulates the attention and feed-forward layers with **shortcut connections** (residual connections).

### Full Architecture
* **GPT Model** (`GPTModel` class): The main model, which integrates token embeddings, positional embeddings, a sequence of `TransformerBlock`s, and an output head.
* **Configuration**: Uses `GPT_CONFIG_124M` to define a model matching the specifications of the small, **124 million parameter GPT-2** model.

### Training & Inference
* **Data Pipeline**: Custom `GPTDatasetV1` and `create_dataloader_v1` for efficient batch processing of long text sequences.
* **Loss Calculation**: Utility functions for calculating training and validation set losses (`calculate_loss_v1`).
* **Text Generation**: Implements both **Greedy Decoding** (`generate_text_simple`) and advanced **Sampling** (`generate`) with support for **Top-K** filtering and **Temperature** scaling.

### Pre-trained Weights
* **OpenAI Weight Loading**: Includes a sophisticated utility function (`load_weights_into_gpt`) to correctly map and load the official **pre-trained GPT-2 124M weights** from OpenAI into the custom PyTorch model, allowing for high-quality text generation immediately after loading.

---

## Requirements and Setup

### Prerequisites

To run the Jupyter Notebook and the code, you need **Python 3.10+**.

### Installation

Clone the repository and install the necessary libraries. The notebook requires specific versions of libraries, including a dependency on TensorFlow for the pre-trained weight conversion utility.

```bash
# Clone the repository
git clone <your-repo-link>
cd gpt-from-scratch

# Create a virtual environment (optional, but recommended)
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

# Install required packages
# These packages are necessary for the implementation and the pre-trained weight loading utility
pip install torch tiktoken numpy tqdm requests tensorflow>=2.15.0

---

## Usage

The entire implementation and walkthrough are contained within a single Jupyter Notebook.

### 1. Run the Notebook

Start the Jupyter environment and open the main file:

```bash
jupyter notebook

Open the `Entire_Chatgpt.ipynb` file. Running the cells sequentially will:

* Load the sample training text (`the-verdict.txt`).
* Define and instantiate all model component classes.
* Test the training pipeline.
* Demonstrate text generation.

### 2. Loading Pre-trained GPT-2 Weights

The key highlight is loading the official GPT-2 weights, which allows the custom-built model to generate coherent text immediately.

* Navigate to **"STEP 10: LOADING PRETRAINED WEIGHTS FROM OPENAI"** in the notebook.
* Run the cells to download the checkpoint files for the GPT-2 Small (124M) model.
* Execute the `load_weights_into_gpt` function to correctly map the downloaded weights to your `GPTModel` instance.
