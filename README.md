# 🤖 Transformer Parameter Calculator

A beautiful, modern web-based calculator for transformer model parameters and memory requirements. Features a stunning gradient UI with accurate parameter calculations for GPT, BERT, T5, and custom transformer architectures.

## ✨ Features

- **🎨 Beautiful Gradient UI**: Modern web interface with smooth gradients and responsive design
- **🔧 Accurate Calculations**: Corrected parameter counting formulas for all transformer types
- **⚡ Real-time Updates**: Auto-calculation as you type with debouncing
- **📱 Mobile Friendly**: Responsive design that works on all devices
- **🎯 Model Presets**: Quick presets for popular models (GPT-2, BERT, T5)
- **💾 Memory Estimation**: Precise memory calculations with different precision options
- **📊 Detailed Breakdown**: Parameter distribution across attention, feed-forward, and embeddings
- **🔍 Tooltips**: Helpful explanations for each parameter

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ 
- pip

### Installation & Running

1. **Clone/Navigate to the project:**
   ```bash
   cd transformer_param_calc
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:5000`

## 🧮 What's Been Fixed

The original Tkinter version had several calculation errors that have been corrected:

### ❌ Original Issues:
- Incorrect attention parameter formula
- Missing bias terms in calculations  
- Wrong layer normalization parameter counts
- Confused positional encoding parameters
- Incorrect encoder-decoder cross-attention calculations

### ✅ Corrected Logic:
- **Attention Parameters**: `4 * d_model * d_model + 4 * d_model` (Q, K, V, Output projections + biases)
- **Feed-Forward**: `d_model * d_ff + d_ff + d_ff * d_model + d_model` (2 linear layers with biases)
- **Layer Normalization**: `2 * 2 * d_model` per layer (2 layer norms, each with scale & shift)
- **Embeddings**: `vocab_size * d_model + context_length * d_model` (token + position embeddings)
- **Cross-Attention**: Properly calculated for encoder-decoder models

## 🎯 Supported Model Types

- **Decoder-only** (GPT-style): Autoregressive language models
- **Encoder-only** (BERT-style): Bidirectional encoders for classification
- **Encoder-Decoder** (T5-style): Sequence-to-sequence models

## 📊 Calculation Details

The calculator provides:
- **Total Parameters**: Complete parameter count
- **Memory Usage**: With different precision options (32/16/8/4-bit)
- **Parameter Breakdown**: Attention vs Feed-Forward vs Embeddings
- **FLOPs Estimation**: Approximate floating-point operations per token

## 🎨 UI Features

- **Gradient Backgrounds**: Beautiful color transitions
- **Glass Morphism**: Translucent panels with backdrop blur
- **Smooth Animations**: Hover effects and transitions
- **Responsive Grid**: Adapts to screen size
- **Interactive Tooltips**: Hover for parameter explanations
- **Real-time Validation**: Input validation with error messages

## 🔧 Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Styling**: Pure CSS with gradients and animations
- **No Dependencies**: Runs with just Flask, no complex setup

## 📱 Mobile Support

The interface is fully responsive and works great on:
- Desktop computers
- Tablets
- Mobile phones
- All modern browsers

## 🤝 Contributing

Feel free to submit issues or pull requests to improve the calculator!

## 📄 License

MIT License - feel free to use and modify as needed. 