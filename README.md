# nanoPaliGemma

## 📖 Overview

This repository contains a faithful, **from-scratch implementation** of Google's **PaliGemma** Vision-Language Model (VLM) using PyTorch.

PaliGemma is a lightweight open VLM inspired by PaLI-3. It combines a **SigLIP** vision encoder with the **Gemma** language model. This project reconstructs the model architecture layer-by-layer, including the multi-modal projector and the attention mechanisms, offering a transparent look into how modern VLMs process visual and textual tokens simultaneously.

### Key Features

- **Pure PyTorch:** No abstraction layers from high-level libraries; all model components (Encoder, Projector, Decoder) are explicitly defined.
- **Modular Design:** Clean separation between the Vision Transformer (SigLIP) and the LLM (Gemma).
- **Inference Ready:** Simple scripts to load weights and run inference on custom images.

---

## 🛠️ Installation

Follow these steps to set up the environment and prepare the model for inference.

### 1\. Clone the Repository

```bash
git clone https://github.com/sahilX7/nanoPaliGemma.git
cd nanoPaliGemma
```

### 2\. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\\Scripts\\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3\. Install Dependencies

Install the required PyTorch and utility libraries.

```bash
pip install -r requirements.txt
```

> Note: Ensure you have a compatible version of torch installed with CUDA support if you intend to run this on a GPU.
> 

---

## 🚀 Usage

### 1\. Prepare Data

Download any image you wish to test and place it in the `test_images` folder

### 2\. Configure Inference

Open `main.py` and modify the configuration constants at the top of the file to match your desired input.

```python
# main.py

IMAGE_PATH = "test_images/dog.jpg"
MAX_TOKENS_TO_GENERATE = 1
```

### 3\. Run the Model

Execute the main script to generate a caption or answer based on the image and prompt.

```bash
python main.py

```

### 4\. Expected Output

You will see an `Ask anything` prompt in your terminal. Enter your text prompt about the image to generate a response. The model will process the image embeddings, concatenate them with your text prompt, and autoregressively generate a response.

```
Ask anything:
describe this image

Device in use: cpu
Loading model...
Running inference...

--------------------------------------------------
USER: describe this image
ASSISTANT: Dog raising its paw
```

---

## 📂 Project Structure

```
nanoPaliGemma/
├── paligemma/
│   ├── core/
│   │   └── paligemma.py           # Defines the core PaliGemma architecture
│   ├── inference/
│   │   └── inference.py           # Contains the inference loop and generation logic
│   ├── models/
│   │   ├── gemma.py               # Gemma language model implementation
│   │   └── siglip.py              # SigLIP vision encoder implementation
│   ├── processor/
│   │   └── paligemma_processor.py # Handles input data processing 
│   └── utils/
│       └── utils.py               # Utilities to load PaliGemma weights into the architecture
├── test_images/                   # Directory for input images
├── venv/                          # Virtual environment
├── weights/                       # Directory for model weights
├── main.py                        # Entry point for the application
└── requirements.txt               # Python dependencies
└── README.md               # Project documentation

```

---

## 🧠 Model Architecture Details

This implementation focuses on the three critical components of PaliGemma:

1. **Vision Encoder (SigLIP):** Contrasted Signal-Language Image Pre-training. It maps the image into a sequence of "soft" tokens.
2. **Projector:** A linear projection layer that aligns the dimensions of the visual tokens with the embedding space of the language model.
3. **Language Model (Gemma):** A decoder-only Transformer that takes the concatenated sequence of (Visual Tokens + Text Tokens) and generates the output text.

$\text{Output} = \text{Gemma}(\text{Projector}(\text{SigLIP}(\text{Image})) + \text{Tokenizer}(\text{Prompt}))$

---

## 📜 Acknowledgements

- Based on the paper: *PaliGemma: A Versatile 3B VLM for Transfer* by Google DeepMind.
- Architecture references from the official [PaliGemma](https://github.com/huggingface/transformers/tree/main/src/transformers/models/paligemma), [SigLIP](https://github.com/huggingface/transformers/tree/main/src/transformers/models/siglip) and [Gemma](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma) implementations.