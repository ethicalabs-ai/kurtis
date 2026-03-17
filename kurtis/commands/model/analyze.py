import click
import json
import os
import torch
from rich.console import Console
from rich.table import Table
from kurtis.model import load_model_and_tokenizer
from kurtis.train import train_model

@click.command(name="analyze")
@click.option("--analysis-json", default="dataset_analysis.json", help="Path to dataset analysis JSON.")
@click.option("--output-dir", default="./model_analysis", help="Output directory.")
@click.option("--preprocessed-dataset-path", default="./processed_dataset", help="Path to preprocessed dataset.")
@click.pass_context
def command(ctx, analysis_json, output_dir, preprocessed_dataset_path):
    """Analyze model architecture and verify hyperparams with a 2-step TTT run."""
    config = ctx.obj["CONFIG"]
    console = Console()
    
    if not os.path.exists(analysis_json):
        console.print(f"[red]Error: {analysis_json} not found. Run 'kurtis dataset analyze' first.[/red]")
        return
        
    with open(analysis_json, "r") as f:
        data_stats = json.load(f)
        
    console.print("[bold blue]Analyzing model architecture and layers...[/bold blue]")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Layer Analysis
    table = Table(title="Model Layer Analysis")
    table.add_column("Layer Name", style="cyan")
    table.add_column("Parameters", style="magenta")
    table.add_column("LoRA Target", style="green")
    
    total_params = 0
    trainable_params = 0
    
    target_modules = config.LORA_CONFIG.target_modules
    
    # Group layers for summary
    layer_types = {}
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        is_target = any(m in name for m in target_modules)
        
        parts = name.split(".")
        layer_base = ".".join(parts[:3]) if len(parts) > 3 else parts[0]
        layer_types[layer_base] = layer_types.get(layer_base, 0) + param.numel()
        
    # Show summary of layer types/blocks
    for layer, count in list(layer_types.items())[:20]:
        is_target = any(m in layer for m in target_modules)
        table.add_row(layer, f"{count:,}", str(is_target))
        
    if len(layer_types) > 20:
        table.add_row("...", "...", "...")
        
    table.add_row("Total Parameters", f"{total_params:,}", "N/A")
    
    console.print(table)
    
    # TTT Run (1 step)
    console.print("\n[bold yellow]Performing 1-step TTT (Test-Time Training) verification...[/bold yellow]")
    
    # Prepare TTT config
    ttt_config = config.TRAINING_CONFIG.copy()
    ttt_config.update({
        "max_steps": 1,
        "logging_steps": 1,
        "eval_steps": 1,
        "eval_subset_size": 2, # Very small eval for verification
        "lr": data_stats["suggested_hyperparams"]["lr"],
        "warmup_ratio": data_stats["suggested_hyperparams"]["warmup_ratio"]
    })
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run a tiny training session
    # We pass the preprocessed dataset path if it exists
    train_model(
        model,
        tokenizer,
        training_config=ttt_config,
        lora_config=config.LORA_CONFIG,
        output_dir=output_dir,
        model_output="ttt_run",
        push=False,
        preprocessed_dataset_path=preprocessed_dataset_path
    )
    
    console.print("\n[bold green]Model analysis and TTT verification complete.[/bold green]")
