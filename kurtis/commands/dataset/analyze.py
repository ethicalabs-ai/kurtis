import click
import json
import os
import math
from rich.console import Console
from rich.table import Table
from kurtis.dataset import load_datasets_from_yaml
from kurtis.model import load_tokenizer_only

@click.command(name="analyze")
@click.option("--dataset-config", default="datasets.yaml", help="Path to YAML dataset config.")
@click.option("--preprocessed-path", default="", help="Path to preprocessed dataset on disk.")
@click.option("--output-json", default="dataset_analysis.json", help="Path to save analysis results.")
@click.option("--sample-size", default=1000, help="Number of samples to analyze for token stats.")
@click.pass_context
def command(ctx, dataset_config, preprocessed_path, output_json, sample_size):
    """Analyze dataset distribution and suggest hyperparameters."""
    config = ctx.obj["CONFIG"]
    console = Console()
    
    dataset = None
    if preprocessed_path and os.path.exists(preprocessed_path):
        from datasets import load_from_disk
        console.print(f"[bold blue]Loading preprocessed dataset from {preprocessed_path}...[/bold blue]")
        dataset = load_from_disk(preprocessed_path)
        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
    elif os.path.exists(dataset_config):
        console.print(f"[bold blue]Loading dataset from {dataset_config}...[/bold blue]")
        dataset = load_datasets_from_yaml(dataset_config)
    else:
        console.print(f"[red]Error: Neither {preprocessed_path} nor {dataset_config} found.[/red]")
        return

    # Stats
    total_samples = len(dataset)
    domains = {}
    sources = {}
    
    # Tokenizer for token stats
    tokenizer = load_tokenizer_only(config, config.TRANSFORMERS_MODEL_PRETRAINED)
    
    token_counts = []
    subset_size = min(total_samples, sample_size)
    
    console.print(f"Randomly sampling {subset_size} samples from {total_samples:,} for analysis...")
    
    import random
    # Use fixed seed for reproducibility in analysis
    random.seed(42)
    indices = random.sample(range(total_samples), subset_size)
    
    # Calculate stats
    with click.progressbar(indices, label="Analyzing samples") as bar:
        for idx in bar:
            example = dataset[idx]
            # Use 'text' field if available (preprocessed), otherwise question + answer
            if "text" in example:
                text = str(example["text"])
            else:
                text = str(example.get("question", "")) + str(example.get("answer", ""))
                
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_counts.append(len(tokens))
            
            domain = example.get("dataset_domain", "unknown")
            source = example.get("dataset_name", "unknown")
            
            domains[domain] = domains.get(domain, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    
    # Total estimated tokens
    est_total_tokens = avg_tokens * total_samples
    
    # Hyperparams suggestion logic (Heuristics)
    # Learning Rate scaling: scale based on total tokens
    # Values adapted from standard LLM scaling laws and Echo-DSRN defaults
    if est_total_tokens > 1e9: # 1B+ tokens
        suggested_lr = 2e-5
        suggested_epochs = 1
    elif est_total_tokens > 1e8: # 100M+ tokens
        suggested_lr = 5e-5
        suggested_epochs = 2
    else:
        suggested_lr = 1e-4
        suggested_epochs = 3
        
    # Table Output
    table = Table(title="Dataset Distribution Analysis")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Samples", f"{total_samples:,}")
    table.add_row("Avg Tokens/Sample", f"{avg_tokens:.2f}")
    table.add_row("Max Tokens Detected", str(max_tokens))
    table.add_row("Estimated Total Tokens", f"{est_total_tokens:,.0f}")
    
    console.print(table)
    
    # Domain Table
    d_table = Table(title="Domain Distribution (Sampled)")
    d_table.add_column("Domain", style="cyan")
    d_table.add_column("Count", style="magenta")
    d_table.add_column("Percentage", style="green")
    
    # Also prepare markdown string for easy copying
    md_table = "| Domain | Count | Percentage |\n| :--- | :--- | :--- |\n"

    for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
        p = (count/subset_size)*100
        d_table.add_row(domain, str(count), f"{p:.1f}%")
        md_table += f"| {domain} | {count} | {p:.1f}% |\n"
        
    console.print(d_table)
    
    console.print("\n[bold blue]Markdown Table (for documentation):[/bold blue]")
    console.print(md_table)
    
    # Suggested Hyperparams
    h_table = Table(title="Suggested Hyperparameters")
    h_table.add_column("Parameter", style="cyan")
    h_table.add_column("Value", style="magenta")
    h_table.add_row("Learning Rate", f"{suggested_lr:.1e}")
    h_table.add_row("Epochs", str(suggested_epochs))
    h_table.add_row("Warmup Ratio", "0.1")
    h_table.add_row("Weight Decay", "0.02")
    
    console.print(h_table)
    
    # Save to JSON
    results = {
        "total_samples": total_samples,
        "avg_tokens": avg_tokens,
        "max_tokens": max_tokens,
        "est_total_tokens": est_total_tokens,
        "suggested_hyperparams": {
            "lr": suggested_lr,
            "epochs": suggested_epochs,
            "warmup_ratio": 0.1,
            "weight_decay": 0.02
        },
        "domains": domains
    }
    
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    console.print(f"\n[green]Analysis results saved to {output_json}[/green]")
