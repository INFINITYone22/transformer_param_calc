from flask import Flask, render_template, request, jsonify
import math
import json

app = Flask(__name__)

def calculate_transformer_params(d_model, num_layers, num_heads, d_ff, vocab_size, 
                               context_length, model_type, precision_bits, attn_precision_bits,
                               precision_type="standard", attn_precision_type="standard"):
    """
    Calculate transformer parameters with enhanced analysis
    """
    try:
        # Validate inputs
        if d_model % num_heads != 0:
            return {"error": "d_model must be divisible by num_heads"}
        
        d_k = d_model // num_heads
        
        # Embedding parameters
        # Token embeddings: vocab_size * d_model
        # Position embeddings: context_length * d_model (for learned positional encoding)
        embed_params = vocab_size * d_model + context_length * d_model
        
        # Per-layer parameters
        # Self-attention: Q, K, V projections + output projection
        # Q, K, V: each is d_model * d_model (including all heads)
        # Output projection: d_model * d_model
        # Bias terms: 4 * d_model (for Q, K, V, and output)
        self_attn_params = 4 * d_model * d_model + 4 * d_model
        
        # Feed-forward network
        # First linear: d_model * d_ff + d_ff (weights + bias)
        # Second linear: d_ff * d_model + d_model (weights + bias)
        ff_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
        
        # Layer normalization (2 per layer: pre-attention and pre-ff)
        # Each layer norm has 2 * d_model parameters (scale and shift)
        layer_norm_params = 2 * 2 * d_model
        
        if model_type == "Decoder-only":
            # Decoder layers: self-attention + feed-forward + layer norms
            layer_params = num_layers * (self_attn_params + ff_params + layer_norm_params)
            # Final layer norm
            final_norm_params = 2 * d_model
            # Output projection (often tied with input embeddings, but counted separately)
            output_params = d_model * vocab_size
            
            total_params = embed_params + layer_params + final_norm_params + output_params
            attn_params = num_layers * self_attn_params
            
        elif model_type == "Encoder-only":
            # Encoder layers: self-attention + feed-forward + layer norms
            layer_params = num_layers * (self_attn_params + ff_params + layer_norm_params)
            # Final layer norm
            final_norm_params = 2 * d_model
            # Classification head (typically d_model * num_classes, assuming small num_classes)
            output_params = d_model * 1000  # Assuming 1000 classes
            
            total_params = embed_params + layer_params + final_norm_params + output_params
            attn_params = num_layers * self_attn_params
            
        else:  # Encoder-Decoder
            # Encoder layers
            encoder_layer_params = num_layers * (self_attn_params + ff_params + layer_norm_params)
            
            # Decoder layers: self-attention + cross-attention + feed-forward + layer norms
            # Cross-attention has same parameter count as self-attention
            cross_attn_params = self_attn_params
            # 3 layer norms per decoder layer (pre-self-attn, pre-cross-attn, pre-ff)
            decoder_layer_norm_params = 3 * 2 * d_model
            decoder_layer_params = num_layers * (self_attn_params + cross_attn_params + ff_params + decoder_layer_norm_params)
            
            # Final layer norm
            final_norm_params = 2 * d_model
            # Output projection
            output_params = d_model * vocab_size
            
            total_params = embed_params + encoder_layer_params + decoder_layer_params + final_norm_params + output_params
            attn_params = num_layers * self_attn_params + num_layers * (self_attn_params + cross_attn_params)
        
        # Calculate memory usage
        non_attn_params = total_params - attn_params
        
        # Memory calculation with special handling for binary/ternary
        if precision_type == "binary":
            bytes_per_param = 1 / 8  # 1 bit per parameter
        elif precision_type == "ternary":
            bytes_per_param = 1.585 / 8  # log2(3) bits per parameter
        else:
            bytes_per_param = precision_bits / 8
            
        if attn_precision_type == "binary":
            attn_bytes_per_param = 1 / 8  # 1 bit per parameter
        elif attn_precision_type == "ternary":
            attn_bytes_per_param = 1.585 / 8  # log2(3) bits per parameter
        else:
            attn_bytes_per_param = attn_precision_bits / 8
        
        memory_bytes = (non_attn_params * bytes_per_param + attn_params * attn_bytes_per_param)
        memory_gb = memory_bytes / (1024 ** 3)
        memory_mb = memory_bytes / (1024 ** 2)
        
        # Enhanced calculations
        flops_per_token = 6 * total_params  # Approximate FLOPs per token
        
        # Training time estimation (rough estimates)
        # Based on typical training speeds and hardware
        training_flops_per_token = 2 * flops_per_token  # Forward + backward pass
        
        # Cost analysis (rough estimates for cloud computing)
        # A100 80GB: ~$3/hour, ~312 TFLOPS
        # V100 32GB: ~$2.5/hour, ~125 TFLOPS
        a100_tflops = 312e12
        v100_tflops = 125e12
        a100_cost_per_hour = 3.0
        v100_cost_per_hour = 2.5
        
        # Memory efficiency analysis
        memory_efficiency = {
            "parameters_per_gb": total_params / max(memory_gb, 0.001),
            "memory_breakdown": {
                "weights": memory_gb,
                "gradients": memory_gb,  # Same as weights for training
                "optimizer_states": memory_gb * 2,  # Adam optimizer states
                "activations": estimate_activation_memory(d_model, num_layers, context_length) / (1024**3)
            }
        }
        
        total_training_memory = sum(memory_efficiency["memory_breakdown"].values())
        
        # Hardware recommendations
        hardware_recommendations = get_hardware_recommendations(memory_gb, total_training_memory)
        
        return {
            "total_params": total_params,
            "attn_params": attn_params,
            "non_attn_params": non_attn_params,
            "embed_params": embed_params,
            "memory_gb": memory_gb,
            "memory_mb": memory_mb,
            "flops_per_token": flops_per_token,
            "params_formatted": f"{total_params:,}",
            "memory_formatted": f"{memory_gb:.2f} GB" if memory_gb >= 1 else f"{memory_mb:.1f} MB",
            "precision_info": {
                "weight_type": precision_type,
                "weight_bits": precision_bits if precision_type == "standard" else (1 if precision_type == "binary" else 1.585),
                "attention_type": attn_precision_type,
                "attention_bits": attn_precision_bits if attn_precision_type == "standard" else (1 if attn_precision_type == "binary" else 1.585)
            },
            "training_analysis": {
                "training_flops_per_token": training_flops_per_token,
                "total_training_memory_gb": total_training_memory,
                "memory_efficiency": memory_efficiency,
                "hardware_recommendations": hardware_recommendations
            },
            "cost_analysis": {
                "a100_training_cost_per_hour": a100_cost_per_hour,
                "v100_training_cost_per_hour": v100_cost_per_hour,
                "estimated_tokens_per_hour_a100": a100_tflops / (training_flops_per_token / 1e12),
                "estimated_tokens_per_hour_v100": v100_tflops / (training_flops_per_token / 1e12)
            },
            "model_analysis": {
                "attention_percentage": (attn_params / total_params) * 100,
                "feedforward_percentage": ((non_attn_params - embed_params) / total_params) * 100,
                "embedding_percentage": (embed_params / total_params) * 100,
                "parameters_per_layer": total_params / num_layers if num_layers > 0 else 0,
                "attention_head_dimension": d_k,
                "total_attention_heads": num_heads * num_layers
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

def estimate_activation_memory(d_model, num_layers, context_length, batch_size=1):
    """
    Estimate activation memory requirements
    """
    # Rough estimate of activation memory per token
    # Attention activations: sequence_length * d_model * num_layers
    # Feed-forward activations: sequence_length * d_ff * num_layers (approx 4 * d_model)
    attention_activations = context_length * d_model * num_layers * 4  # 4 bytes per float32
    ff_activations = context_length * d_model * 4 * num_layers * 4  # d_ff ≈ 4 * d_model
    
    total_activations = (attention_activations + ff_activations) * batch_size
    return total_activations

def get_hardware_recommendations(inference_memory_gb, training_memory_gb):
    """
    Provide hardware recommendations based on memory requirements
    """
    recommendations = {
        "inference": [],
        "training": []
    }
    
    # Inference recommendations
    if inference_memory_gb <= 4:
        recommendations["inference"].extend(["RTX 3060 (12GB)", "RTX 4060 Ti (16GB)", "RTX 3070 (8GB)"])
    elif inference_memory_gb <= 8:
        recommendations["inference"].extend(["RTX 3080 (10GB)", "RTX 4070 (12GB)", "RTX 3070 Ti (8GB)"])
    elif inference_memory_gb <= 12:
        recommendations["inference"].extend(["RTX 3080 Ti (12GB)", "RTX 4070 Ti (12GB)", "RTX 3060 (12GB)"])
    elif inference_memory_gb <= 16:
        recommendations["inference"].extend(["RTX 4080 (16GB)", "RTX 4060 Ti (16GB)"])
    elif inference_memory_gb <= 24:
        recommendations["inference"].extend(["RTX 3090 (24GB)", "RTX 4090 (24GB)", "RTX A5000 (24GB)"])
    elif inference_memory_gb <= 48:
        recommendations["inference"].extend(["RTX A6000 (48GB)", "A40 (48GB)"])
    else:
        recommendations["inference"].extend(["A100 (80GB)", "H100 (80GB)", "Multiple GPUs required"])
    
    # Training recommendations
    if training_memory_gb <= 16:
        recommendations["training"].extend(["RTX 4080 (16GB)", "RTX 4060 Ti (16GB)"])
    elif training_memory_gb <= 24:
        recommendations["training"].extend(["RTX 3090 (24GB)", "RTX 4090 (24GB)"])
    elif training_memory_gb <= 48:
        recommendations["training"].extend(["RTX A6000 (48GB)", "A40 (48GB)"])
    elif training_memory_gb <= 80:
        recommendations["training"].extend(["A100 (80GB)", "H100 (80GB)"])
    else:
        recommendations["training"].extend(["Multiple A100s", "Multiple H100s", "Distributed training required"])
    
    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    
    try:
        # Handle precision bits conversion safely
        precision_bits = float(data.get('precision_bits', 32))
        attn_precision_bits = float(data.get('attn_precision_bits', 32))
        
        result = calculate_transformer_params(
            d_model=int(data['d_model']),
            num_layers=int(data['num_layers']),
            num_heads=int(data['num_heads']),
            d_ff=int(data['d_ff']),
            vocab_size=int(data['vocab_size']),
            context_length=int(data['context_length']),
            model_type=data['model_type'],
            precision_bits=precision_bits,
            attn_precision_bits=attn_precision_bits,
            precision_type=data.get('precision_type', 'standard'),
            attn_precision_type=data.get('attn_precision_type', 'standard')
        )
        
        return jsonify(result)
        
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": f"Invalid input parameters: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Calculation error: {str(e)}"}), 500

@app.route('/presets')
def get_presets():
    presets = {
        "GPT-2 Small": {
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "d_ff": 3072,
            "vocab_size": 50257,
            "context_length": 1024,
            "model_type": "Decoder-only"
        },
        "GPT-2 Medium": {
            "d_model": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "d_ff": 4096,
            "vocab_size": 50257,
            "context_length": 1024,
            "model_type": "Decoder-only"
        },
        "GPT-2 Large": {
            "d_model": 1280,
            "num_layers": 36,
            "num_heads": 20,
            "d_ff": 5120,
            "vocab_size": 50257,
            "context_length": 1024,
            "model_type": "Decoder-only"
        },
        "GPT-3.5": {
            "d_model": 4096,
            "num_layers": 96,
            "num_heads": 32,
            "d_ff": 16384,
            "vocab_size": 50257,
            "context_length": 4096,
            "model_type": "Decoder-only"
        },
        "BERT Base": {
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "d_ff": 3072,
            "vocab_size": 30522,
            "context_length": 512,
            "model_type": "Encoder-only"
        },
        "BERT Large": {
            "d_model": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "d_ff": 4096,
            "vocab_size": 30522,
            "context_length": 512,
            "model_type": "Encoder-only"
        },
        "T5 Small": {
            "d_model": 512,
            "num_layers": 6,
            "num_heads": 8,
            "d_ff": 2048,
            "vocab_size": 32128,
            "context_length": 512,
            "model_type": "Encoder-Decoder"
        },
        "T5 Base": {
            "d_model": 768,
            "num_layers": 12,
            "num_heads": 12,
            "d_ff": 3072,
            "vocab_size": 32128,
            "context_length": 512,
            "model_type": "Encoder-Decoder"
        },
        "LLaMA 7B": {
            "d_model": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "d_ff": 11008,
            "vocab_size": 32000,
            "context_length": 2048,
            "model_type": "Decoder-only"
        },
        "LLaMA 13B": {
            "d_model": 5120,
            "num_layers": 40,
            "num_heads": 40,
            "d_ff": 13824,
            "vocab_size": 32000,
            "context_length": 2048,
            "model_type": "Decoder-only"
        },
        "Mistral 7B": {
            "d_model": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "d_ff": 14336,
            "vocab_size": 32000,
            "context_length": 8192,
            "model_type": "Decoder-only"
        },
        "Gemma 2B": {
            "d_model": 2048,
            "num_layers": 18,
            "num_heads": 8,
            "d_ff": 16384,
            "vocab_size": 256000,
            "context_length": 8192,
            "model_type": "Decoder-only"
        },
        "Gemma 7B": {
            "d_model": 3072,
            "num_layers": 28,
            "num_heads": 16,
            "d_ff": 24576,
            "vocab_size": 256000,
            "context_length": 8192,
            "model_type": "Decoder-only"
        }
    }
    return jsonify(presets)

@app.route('/compare', methods=['POST'])
def compare_models():
    """
    Compare multiple model configurations
    """
    data = request.json
    models = data.get('models', [])
    
    if not models:
        return jsonify({"error": "No models provided for comparison"}), 400
    
    comparison_results = []
    
    for model_config in models:
        try:
            result = calculate_transformer_params(
                d_model=int(model_config['d_model']),
                num_layers=int(model_config['num_layers']),
                num_heads=int(model_config['num_heads']),
                d_ff=int(model_config['d_ff']),
                vocab_size=int(model_config['vocab_size']),
                context_length=int(model_config['context_length']),
                model_type=model_config['model_type'],
                precision_bits=float(model_config.get('precision_bits', 32)),
                attn_precision_bits=float(model_config.get('attn_precision_bits', 32)),
                precision_type=model_config.get('precision_type', 'standard'),
                attn_precision_type=model_config.get('attn_precision_type', 'standard')
            )
            
            if 'error' not in result:
                result['name'] = model_config.get('name', f"Model {len(comparison_results) + 1}")
                comparison_results.append(result)
            
        except Exception as e:
            continue
    
    return jsonify({
        "comparison": comparison_results,
        "summary": generate_comparison_summary(comparison_results)
    })

def generate_comparison_summary(results):
    """
    Generate a summary of model comparisons
    """
    if not results:
        return {}
    
    params = [r['total_params'] for r in results]
    memories = [r['memory_gb'] for r in results]
    
    return {
        "total_models": len(results),
        "parameter_range": {
            "min": min(params),
            "max": max(params),
            "ratio": max(params) / min(params) if min(params) > 0 else 0
        },
        "memory_range": {
            "min": min(memories),
            "max": max(memories),
            "ratio": max(memories) / min(memories) if min(memories) > 0 else 0
        },
        "largest_model": max(results, key=lambda x: x['total_params'])['name'],
        "smallest_model": min(results, key=lambda x: x['total_params'])['name']
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 