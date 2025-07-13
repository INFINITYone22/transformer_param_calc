import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import math
from datetime import datetime

class DarkTheme:
    # Dark theme color palette
    BG_PRIMARY = "#0f0f23"
    BG_SECONDARY = "#1a1a2e"
    BG_CARD = "#16213e"
    ACCENT_PRIMARY = "#6366f1"
    ACCENT_SECONDARY = "#8b5cf6"
    TEXT_PRIMARY = "#f8fafc"
    TEXT_SECONDARY = "#cbd5e1"
    TEXT_MUTED = "#64748b"
    BORDER = "#334155"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"

class TransformerCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.current_results = None
        
    def setup_window(self):
        self.root.title("Transformer Parameter Calculator Pro")
        self.root.geometry("1000x700")
        self.root.configure(bg=DarkTheme.BG_PRIMARY)
        self.root.resizable(True, True)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"1000x700+{x}+{y}")
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure dark theme styles
        style.configure('Title.TLabel', 
                       font=('Inter', 18, 'bold'),
                       foreground=DarkTheme.TEXT_PRIMARY,
                       background=DarkTheme.BG_PRIMARY)
        
        style.configure('Heading.TLabel',
                       font=('Inter', 12, 'bold'),
                       foreground=DarkTheme.ACCENT_PRIMARY,
                       background=DarkTheme.BG_CARD)
        
        style.configure('Dark.TLabel',
                       font=('Inter', 10),
                       foreground=DarkTheme.TEXT_SECONDARY,
                       background=DarkTheme.BG_CARD)
        
        style.configure('Dark.TEntry',
                       font=('Inter', 10),
                       foreground=DarkTheme.TEXT_PRIMARY,
                       fieldbackground=DarkTheme.BG_SECONDARY,
                       bordercolor=DarkTheme.BORDER,
                       insertcolor=DarkTheme.TEXT_PRIMARY)
        
        style.configure('Dark.TCombobox',
                       font=('Inter', 10),
                       foreground=DarkTheme.TEXT_PRIMARY,
                       fieldbackground=DarkTheme.BG_SECONDARY,
                       bordercolor=DarkTheme.BORDER)
        
        style.configure('Dark.TButton',
                       font=('Inter', 10, 'bold'),
                       foreground=DarkTheme.TEXT_PRIMARY,
                       background=DarkTheme.ACCENT_PRIMARY,
                       bordercolor=DarkTheme.ACCENT_PRIMARY,
                       focuscolor='none')
        
        style.map('Dark.TButton',
                 background=[('active', DarkTheme.ACCENT_SECONDARY),
                           ('pressed', DarkTheme.ACCENT_SECONDARY)])
        
        style.configure('Success.TButton',
                       font=('Inter', 10, 'bold'),
                       foreground=DarkTheme.TEXT_PRIMARY,
                       background=DarkTheme.SUCCESS,
                       bordercolor=DarkTheme.SUCCESS,
                       focuscolor='none')
        
        style.configure('Dark.TFrame',
                       background=DarkTheme.BG_CARD,
                       bordercolor=DarkTheme.BORDER,
                       relief='solid',
                       borderwidth=1)
        
        style.configure('Card.TFrame',
                       background=DarkTheme.BG_CARD,
                       bordercolor=DarkTheme.BORDER,
                       relief='solid',
                       borderwidth=1)
        
        style.configure('Dark.TNotebook',
                       background=DarkTheme.BG_SECONDARY,
                       bordercolor=DarkTheme.BORDER)
        
        style.configure('Dark.TNotebook.Tab',
                       background=DarkTheme.BG_SECONDARY,
                       foreground=DarkTheme.TEXT_SECONDARY,
                       padding=[20, 8])
        
        style.map('Dark.TNotebook.Tab',
                 background=[('selected', DarkTheme.ACCENT_PRIMARY)],
                 foreground=[('selected', DarkTheme.TEXT_PRIMARY)])
        
    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=DarkTheme.BG_PRIMARY)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = ttk.Label(main_frame, text="🤖 Transformer Parameter Calculator Pro", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame, style='Dark.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Configuration tab
        self.config_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.config_frame, text="Configuration")
        self.create_config_tab()
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.results_frame, text="Results")
        self.create_results_tab()
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.analysis_frame, text="Analysis")
        self.create_analysis_tab()
        
    def create_config_tab(self):
        # Create scrollable frame
        canvas = tk.Canvas(self.config_frame, bg=DarkTheme.BG_CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Dark.TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Presets section
        presets_frame = ttk.LabelFrame(scrollable_frame, text="Quick Presets", style='Card.TFrame')
        presets_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        presets_inner = ttk.Frame(presets_frame, style='Dark.TFrame')
        presets_inner.pack(fill='x', padx=10, pady=10)
        
        self.create_preset_buttons(presets_inner)
        
        # Model configuration section
        config_frame = ttk.LabelFrame(scrollable_frame, text="Model Configuration", style='Card.TFrame')
        config_frame.pack(fill='x', padx=20, pady=10)
        
        config_inner = ttk.Frame(config_frame, style='Dark.TFrame')
        config_inner.pack(fill='x', padx=10, pady=10)
        
        self.create_config_inputs(config_inner)
        
        # Precision settings section
        precision_frame = ttk.LabelFrame(scrollable_frame, text="Precision Settings", style='Card.TFrame')
        precision_frame.pack(fill='x', padx=20, pady=10)
        
        precision_inner = ttk.Frame(precision_frame, style='Dark.TFrame')
        precision_inner.pack(fill='x', padx=10, pady=10)
        
        self.create_precision_inputs(precision_inner)
        
        # Action buttons
        action_frame = ttk.Frame(scrollable_frame, style='Dark.TFrame')
        action_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Button(action_frame, text="🎯 Auto Configure", command=self.auto_configure,
                  style='Success.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(action_frame, text="🧮 Calculate Parameters", command=self.calculate_params,
                  style='Dark.TButton').pack(side='left', padx=10)
        ttk.Button(action_frame, text="💾 Save Config", command=self.save_config,
                  style='Dark.TButton').pack(side='left', padx=10)
        ttk.Button(action_frame, text="📁 Load Config", command=self.load_config,
                  style='Dark.TButton').pack(side='left', padx=10)
        
    def create_preset_buttons(self, parent):
        presets = [
            "GPT-2 Small", "GPT-2 Medium", "GPT-2 Large", "GPT-3.5",
            "BERT Base", "BERT Large", "T5 Small", "T5 Base",
            "LLaMA 7B", "LLaMA 13B", "Mistral 7B", "Gemma 7B"
        ]
        
        for i, preset in enumerate(presets):
            row = i // 4
            col = i % 4
            btn = ttk.Button(parent, text=preset, 
                           command=lambda p=preset: self.load_preset(p),
                           style='Dark.TButton')
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
        
        # Configure grid weights
        for i in range(4):
            parent.grid_columnconfigure(i, weight=1)
    
    def create_config_inputs(self, parent):
        # Create input variables
        self.d_model = tk.StringVar(value="768")
        self.num_layers = tk.StringVar(value="12")
        self.num_heads = tk.StringVar(value="12")
        self.d_ff = tk.StringVar(value="3072")
        self.vocab_size = tk.StringVar(value="50257")
        self.context_length = tk.StringVar(value="1024")
        self.model_type = tk.StringVar(value="Decoder-only")
        
        # Grid layout for inputs
        inputs = [
            ("Model Dimension (d_model):", self.d_model),
            ("Number of Layers:", self.num_layers),
            ("Number of Heads:", self.num_heads),
            ("Feed-Forward Dimension:", self.d_ff),
            ("Vocabulary Size:", self.vocab_size),
            ("Context Length:", self.context_length),
        ]
        
        for i, (label_text, var) in enumerate(inputs):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(parent, text=label_text, style='Dark.TLabel').grid(
                row=row, column=col, sticky='w', padx=(0, 10), pady=5)
            ttk.Entry(parent, textvariable=var, style='Dark.TEntry', width=15).grid(
                row=row, column=col+1, sticky='w', padx=(0, 20), pady=5)
        
        # Model type dropdown
        ttk.Label(parent, text="Model Type:", style='Dark.TLabel').grid(
            row=3, column=0, sticky='w', padx=(0, 10), pady=5)
        model_combo = ttk.Combobox(parent, textvariable=self.model_type, 
                                  values=["Decoder-only", "Encoder-only", "Encoder-Decoder"],
                                  style='Dark.TCombobox', state='readonly', width=15)
        model_combo.grid(row=3, column=1, sticky='w', padx=(0, 20), pady=5)
        
    def create_precision_inputs(self, parent):
        self.precision_type = tk.StringVar(value="standard")
        self.precision_bits = tk.StringVar(value="32")
        self.attn_precision_type = tk.StringVar(value="standard")
        self.attn_precision_bits = tk.StringVar(value="32")
        
        # Weight precision
        ttk.Label(parent, text="Weight Precision Type:", style='Dark.TLabel').grid(
            row=0, column=0, sticky='w', padx=(0, 10), pady=5)
        weight_precision_combo = ttk.Combobox(parent, textvariable=self.precision_type,
                                            values=["standard", "binary", "ternary"],
                                            style='Dark.TCombobox', state='readonly', width=15)
        weight_precision_combo.grid(row=0, column=1, sticky='w', padx=(0, 20), pady=5)
        
        ttk.Label(parent, text="Weight Bits:", style='Dark.TLabel').grid(
            row=0, column=2, sticky='w', padx=(0, 10), pady=5)
        ttk.Entry(parent, textvariable=self.precision_bits, style='Dark.TEntry', width=15).grid(
            row=0, column=3, sticky='w', pady=5)
        
        # Attention precision
        ttk.Label(parent, text="Attention Precision Type:", style='Dark.TLabel').grid(
            row=1, column=0, sticky='w', padx=(0, 10), pady=5)
        attn_precision_combo = ttk.Combobox(parent, textvariable=self.attn_precision_type,
                                          values=["standard", "binary", "ternary"],
                                          style='Dark.TCombobox', state='readonly', width=15)
        attn_precision_combo.grid(row=1, column=1, sticky='w', padx=(0, 20), pady=5)
        
        ttk.Label(parent, text="Attention Bits:", style='Dark.TLabel').grid(
            row=1, column=2, sticky='w', padx=(0, 10), pady=5)
        ttk.Entry(parent, textvariable=self.attn_precision_bits, style='Dark.TEntry', width=15).grid(
            row=1, column=3, sticky='w', pady=5)
        
    def create_results_tab(self):
        # Results display
        results_container = ttk.Frame(self.results_frame, style='Dark.TFrame')
        results_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            results_container,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=('Consolas', 11),
            bg=DarkTheme.BG_SECONDARY,
            fg=DarkTheme.TEXT_PRIMARY,
            insertbackground=DarkTheme.TEXT_PRIMARY,
            selectbackground=DarkTheme.ACCENT_PRIMARY,
            borderwidth=1,
            relief='solid'
        )
        self.results_text.pack(fill='both', expand=True)
        
        # Export buttons
        export_frame = ttk.Frame(results_container, style='Dark.TFrame')
        export_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(export_frame, text="📄 Export JSON", command=lambda: self.export_results('json'),
                  style='Dark.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(export_frame, text="📊 Export CSV", command=lambda: self.export_results('csv'),
                  style='Dark.TButton').pack(side='left', padx=10)
        ttk.Button(export_frame, text="📝 Export Report", command=lambda: self.export_results('txt'),
                  style='Dark.TButton').pack(side='left', padx=10)
        
    def create_analysis_tab(self):
        # Analysis display
        analysis_container = ttk.Frame(self.analysis_frame, style='Dark.TFrame')
        analysis_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Analysis text area
        self.analysis_text = scrolledtext.ScrolledText(
            analysis_container,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=('Consolas', 11),
            bg=DarkTheme.BG_SECONDARY,
            fg=DarkTheme.TEXT_PRIMARY,
            insertbackground=DarkTheme.TEXT_PRIMARY,
            selectbackground=DarkTheme.ACCENT_PRIMARY,
            borderwidth=1,
            relief='solid'
        )
        self.analysis_text.pack(fill='both', expand=True)
        
    def load_preset(self, preset_name):
        presets = {
            "GPT-2 Small": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072, "vocab_size": 50257, "context_length": 1024, "model_type": "Decoder-only"},
            "GPT-2 Medium": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096, "vocab_size": 50257, "context_length": 1024, "model_type": "Decoder-only"},
            "GPT-2 Large": {"d_model": 1280, "num_layers": 36, "num_heads": 20, "d_ff": 5120, "vocab_size": 50257, "context_length": 1024, "model_type": "Decoder-only"},
            "GPT-3.5": {"d_model": 4096, "num_layers": 96, "num_heads": 32, "d_ff": 16384, "vocab_size": 50257, "context_length": 4096, "model_type": "Decoder-only"},
            "BERT Base": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072, "vocab_size": 30522, "context_length": 512, "model_type": "Encoder-only"},
            "BERT Large": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096, "vocab_size": 30522, "context_length": 512, "model_type": "Encoder-only"},
            "T5 Small": {"d_model": 512, "num_layers": 6, "num_heads": 8, "d_ff": 2048, "vocab_size": 32128, "context_length": 512, "model_type": "Encoder-Decoder"},
            "T5 Base": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072, "vocab_size": 32128, "context_length": 512, "model_type": "Encoder-Decoder"},
            "LLaMA 7B": {"d_model": 4096, "num_layers": 32, "num_heads": 32, "d_ff": 11008, "vocab_size": 32000, "context_length": 2048, "model_type": "Decoder-only"},
            "LLaMA 13B": {"d_model": 5120, "num_layers": 40, "num_heads": 40, "d_ff": 13824, "vocab_size": 32000, "context_length": 2048, "model_type": "Decoder-only"},
            "Mistral 7B": {"d_model": 4096, "num_layers": 32, "num_heads": 32, "d_ff": 14336, "vocab_size": 32000, "context_length": 8192, "model_type": "Decoder-only"},
            "Gemma 7B": {"d_model": 3072, "num_layers": 28, "num_heads": 16, "d_ff": 24576, "vocab_size": 256000, "context_length": 8192, "model_type": "Decoder-only"}
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            self.d_model.set(str(preset["d_model"]))
            self.num_layers.set(str(preset["num_layers"]))
            self.num_heads.set(str(preset["num_heads"]))
            self.d_ff.set(str(preset["d_ff"]))
            self.vocab_size.set(str(preset["vocab_size"]))
            self.context_length.set(str(preset["context_length"]))
            self.model_type.set(preset["model_type"])
            
            # Auto-calculate after loading preset
            self.calculate_params()
            
    def auto_configure(self):
        try:
            d_model = int(self.d_model.get())
            if d_model < 64:
                messagebox.showerror("Error", "Please enter a valid model dimension (d_model >= 64)")
                return
            
            # Calculate recommended values
            recommended_heads = max(1, d_model // 64)
            recommended_d_ff = 4 * d_model
            
            self.num_heads.set(str(recommended_heads))
            self.d_ff.set(str(recommended_d_ff))
            self.vocab_size.set("50257")
            self.context_length.set("1024")
            self.precision_type.set("standard")
            self.precision_bits.set("32")
            self.attn_precision_type.set("standard")
            self.attn_precision_bits.set("32")
            
            messagebox.showinfo("Success", "Configuration auto-generated successfully!")
            self.calculate_params()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid model dimension first")
            
    def calculate_params(self):
        try:
            # Get input values
            d_model = int(self.d_model.get())
            num_layers = int(self.num_layers.get())
            num_heads = int(self.num_heads.get())
            d_ff = int(self.d_ff.get())
            vocab_size = int(self.vocab_size.get())
            context_length = int(self.context_length.get())
            model_type = self.model_type.get()
            precision_bits = float(self.precision_bits.get())
            attn_precision_bits = float(self.attn_precision_bits.get())
            precision_type = self.precision_type.get()
            attn_precision_type = self.attn_precision_type.get()
            
            # Calculate parameters using the same logic as the web app
            result = self.calculate_transformer_params(
                d_model, num_layers, num_heads, d_ff, vocab_size,
                context_length, model_type, precision_bits, attn_precision_bits,
                precision_type, attn_precision_type
            )
            
            if "error" in result:
                messagebox.showerror("Error", result["error"])
                return
            
            self.current_results = result
            self.display_results(result)
            self.display_analysis(result)
            
            # Switch to results tab
            self.notebook.select(1)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Calculation error: {str(e)}")
            
    def calculate_transformer_params(self, d_model, num_layers, num_heads, d_ff, vocab_size, 
                                   context_length, model_type, precision_bits, attn_precision_bits,
                                   precision_type="standard", attn_precision_type="standard"):
        """Enhanced parameter calculation with detailed analysis"""
        try:
            if d_model % num_heads != 0:
                return {"error": "d_model must be divisible by num_heads"}
            
            d_k = d_model // num_heads
            
            # Calculate parameters using the same logic as the web app
            embed_params = vocab_size * d_model + context_length * d_model
            self_attn_params = 4 * d_model * d_model + 4 * d_model
            ff_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
            layer_norm_params = 2 * 2 * d_model
            
            if model_type == "Decoder-only":
                layer_params = num_layers * (self_attn_params + ff_params + layer_norm_params)
                final_norm_params = 2 * d_model
                output_params = d_model * vocab_size
                total_params = embed_params + layer_params + final_norm_params + output_params
                attn_params = num_layers * self_attn_params
                
            elif model_type == "Encoder-only":
                layer_params = num_layers * (self_attn_params + ff_params + layer_norm_params)
                final_norm_params = 2 * d_model
                output_params = d_model * 1000
                total_params = embed_params + layer_params + final_norm_params + output_params
                attn_params = num_layers * self_attn_params
                
            else:  # Encoder-Decoder
                encoder_layer_params = num_layers * (self_attn_params + ff_params + layer_norm_params)
                cross_attn_params = self_attn_params
                decoder_layer_norm_params = 3 * 2 * d_model
                decoder_layer_params = num_layers * (self_attn_params + cross_attn_params + ff_params + decoder_layer_norm_params)
                final_norm_params = 2 * d_model
                output_params = d_model * vocab_size
                total_params = embed_params + encoder_layer_params + decoder_layer_params + final_norm_params + output_params
                attn_params = num_layers * self_attn_params + num_layers * (self_attn_params + cross_attn_params)
            
            non_attn_params = total_params - attn_params
            
            # Memory calculation
            if precision_type == "binary":
                bytes_per_param = 1 / 8
            elif precision_type == "ternary":
                bytes_per_param = 1.585 / 8
            else:
                bytes_per_param = precision_bits / 8
                
            if attn_precision_type == "binary":
                attn_bytes_per_param = 1 / 8
            elif attn_precision_type == "ternary":
                attn_bytes_per_param = 1.585 / 8
            else:
                attn_bytes_per_param = attn_precision_bits / 8
            
            memory_bytes = (non_attn_params * bytes_per_param + attn_params * attn_bytes_per_param)
            memory_gb = memory_bytes / (1024 ** 3)
            memory_mb = memory_bytes / (1024 ** 2)
            
            flops_per_token = 6 * total_params
            
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
            
    def display_results(self, result):
        self.results_text.delete(1.0, tk.END)
        
        report = f"""
🤖 TRANSFORMER PARAMETER CALCULATOR RESULTS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════

📊 MAIN RESULTS:
• Total Parameters: {result['params_formatted']}
• Memory Required: {result['memory_formatted']}
• Attention Parameters: {result['attn_params']:,}
• Non-Attention Parameters: {result['non_attn_params']:,}
• Embedding Parameters: {result['embed_params']:,}
• FLOPs per Token: {result['flops_per_token']:,}

═══════════════════════════════════════════════════════════════

📈 PARAMETER BREAKDOWN:
• Attention Layers: {result['model_analysis']['attention_percentage']:.1f}%
• Feed-Forward Layers: {result['model_analysis']['feedforward_percentage']:.1f}%
• Embeddings: {result['model_analysis']['embedding_percentage']:.1f}%
• Parameters per Layer: {result['model_analysis']['parameters_per_layer']:,.0f}
• Attention Head Dimension: {result['model_analysis']['attention_head_dimension']}
• Total Attention Heads: {result['model_analysis']['total_attention_heads']}

═══════════════════════════════════════════════════════════════

💾 MEMORY ANALYSIS:
• Model Weights: {result['memory_gb']:.2f} GB
• Estimated Training Memory: {result['memory_gb'] * 4:.2f} GB
  (includes gradients, optimizer states, and activations)

═══════════════════════════════════════════════════════════════
        """
        
        self.results_text.insert(tk.END, report)
        
    def display_analysis(self, result):
        self.analysis_text.delete(1.0, tk.END)
        
        # Generate hardware recommendations
        memory_gb = result['memory_gb']
        training_memory = memory_gb * 4
        
        inference_gpus = []
        training_gpus = []
        
        if memory_gb <= 4:
            inference_gpus = ["RTX 3060 (12GB)", "RTX 4060 Ti (16GB)", "RTX 3070 (8GB)"]
        elif memory_gb <= 8:
            inference_gpus = ["RTX 3080 (10GB)", "RTX 4070 (12GB)", "RTX 3070 Ti (8GB)"]
        elif memory_gb <= 12:
            inference_gpus = ["RTX 3080 Ti (12GB)", "RTX 4070 Ti (12GB)", "RTX 3060 (12GB)"]
        elif memory_gb <= 16:
            inference_gpus = ["RTX 4080 (16GB)", "RTX 4060 Ti (16GB)"]
        elif memory_gb <= 24:
            inference_gpus = ["RTX 3090 (24GB)", "RTX 4090 (24GB)", "RTX A5000 (24GB)"]
        elif memory_gb <= 48:
            inference_gpus = ["RTX A6000 (48GB)", "A40 (48GB)"]
        else:
            inference_gpus = ["A100 (80GB)", "H100 (80GB)", "Multiple GPUs required"]
        
        if training_memory <= 16:
            training_gpus = ["RTX 4080 (16GB)", "RTX 4060 Ti (16GB)"]
        elif training_memory <= 24:
            training_gpus = ["RTX 3090 (24GB)", "RTX 4090 (24GB)"]
        elif training_memory <= 48:
            training_gpus = ["RTX A6000 (48GB)", "A40 (48GB)"]
        elif training_memory <= 80:
            training_gpus = ["A100 (80GB)", "H100 (80GB)"]
        else:
            training_gpus = ["Multiple A100s", "Multiple H100s", "Distributed training required"]
        
        analysis = f"""
🔍 DETAILED ANALYSIS & RECOMMENDATIONS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════

🖥️ HARDWARE RECOMMENDATIONS:

💡 For Inference ({memory_gb:.2f} GB required):
{chr(10).join(f"  • {gpu}" for gpu in inference_gpus[:3])}

🏋️ For Training ({training_memory:.2f} GB required):
{chr(10).join(f"  • {gpu}" for gpu in training_gpus[:3])}

═══════════════════════════════════════════════════════════════

⚡ PERFORMANCE ESTIMATES:

• Model Size Category: {"Small" if result['total_params'] < 1e8 else "Medium" if result['total_params'] < 1e10 else "Large"}
• Inference Speed: {"Fast" if memory_gb < 8 else "Medium" if memory_gb < 24 else "Slow"}
• Training Feasibility: {"Easy" if training_memory < 24 else "Moderate" if training_memory < 80 else "Challenging"}

═══════════════════════════════════════════════════════════════

📊 SCALING INSIGHTS:

• Parameters scale roughly as: d_model² × num_layers
• Memory scales linearly with parameters
• Training time scales super-linearly with parameters
• Attention complexity: O(sequence_length²)

═══════════════════════════════════════════════════════════════

💰 COST CONSIDERATIONS:

• Cloud Training Cost (A100): ~$3-5/hour
• Estimated Training Time: {"Hours" if result['total_params'] < 1e8 else "Days" if result['total_params'] < 1e10 else "Weeks"}
• Recommended Approach: {"Single GPU" if training_memory < 80 else "Multi-GPU/Distributed"}

═══════════════════════════════════════════════════════════════

🎯 OPTIMIZATION SUGGESTIONS:

• Consider gradient checkpointing to reduce memory
• Use mixed precision training (FP16/BF16)
• Implement gradient accumulation for larger effective batch sizes
• Consider parameter sharing techniques for efficiency

═══════════════════════════════════════════════════════════════
        """
        
        self.analysis_text.insert(tk.END, analysis)
        
    def save_config(self):
        if not self.current_results:
            messagebox.showwarning("Warning", "No configuration to save. Please calculate parameters first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Configuration"
        )
        
        if filename:
            config = {
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "d_model": int(self.d_model.get()),
                    "num_layers": int(self.num_layers.get()),
                    "num_heads": int(self.num_heads.get()),
                    "d_ff": int(self.d_ff.get()),
                    "vocab_size": int(self.vocab_size.get()),
                    "context_length": int(self.context_length.get()),
                    "model_type": self.model_type.get(),
                    "precision_type": self.precision_type.get(),
                    "precision_bits": float(self.precision_bits.get()),
                    "attn_precision_type": self.attn_precision_type.get(),
                    "attn_precision_bits": float(self.attn_precision_bits.get())
                },
                "results": self.current_results
            }
            
            try:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
                
    def load_config(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Configuration"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                if "configuration" in config:
                    cfg = config["configuration"]
                    self.d_model.set(str(cfg["d_model"]))
                    self.num_layers.set(str(cfg["num_layers"]))
                    self.num_heads.set(str(cfg["num_heads"]))
                    self.d_ff.set(str(cfg["d_ff"]))
                    self.vocab_size.set(str(cfg["vocab_size"]))
                    self.context_length.set(str(cfg["context_length"]))
                    self.model_type.set(cfg["model_type"])
                    self.precision_type.set(cfg.get("precision_type", "standard"))
                    self.precision_bits.set(str(cfg.get("precision_bits", 32)))
                    self.attn_precision_type.set(cfg.get("attn_precision_type", "standard"))
                    self.attn_precision_bits.set(str(cfg.get("attn_precision_bits", 32)))
                    
                    messagebox.showinfo("Success", f"Configuration loaded from {filename}")
                    self.calculate_params()
                else:
                    messagebox.showerror("Error", "Invalid configuration file format")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
                
    def export_results(self, format_type):
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export. Please calculate parameters first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=f".{format_type}",
            filetypes=[(f"{format_type.upper()} files", f"*.{format_type}"), ("All files", "*.*")],
            title=f"Export Results as {format_type.upper()}"
        )
        
        if filename:
            try:
                if format_type == 'json':
                    with open(filename, 'w') as f:
                        json.dump(self.current_results, f, indent=2)
                elif format_type == 'csv':
                    with open(filename, 'w') as f:
                        f.write("Metric,Value\n")
                        f.write(f"Total Parameters,{self.current_results['total_params']}\n")
                        f.write(f"Memory (GB),{self.current_results['memory_gb']}\n")
                        f.write(f"Attention Parameters,{self.current_results['attn_params']}\n")
                        f.write(f"Non-Attention Parameters,{self.current_results['non_attn_params']}\n")
                        f.write(f"Embedding Parameters,{self.current_results['embed_params']}\n")
                        f.write(f"FLOPs per Token,{self.current_results['flops_per_token']}\n")
                else:  # txt
                    with open(filename, 'w') as f:
                        f.write(self.results_text.get(1.0, tk.END))
                        
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

def main():
    root = tk.Tk()
    app = TransformerCalculatorApp(root)
    root.mainloop() 

if __name__ == "__main__":
    main() 