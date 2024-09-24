import gc
import importlib

import click
import pyfiglet
import torch
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text


def print_kurtis_title():
    console = Console()
    title = pyfiglet.figlet_format("Kurtis", font="banner3-D")
    model_details = Text("Kurtis v0.1.0")

    panel = Panel.fit(
        model_details,
        border_style="bright_green",
        padding=(1, 4),
    )
    console.print("\n")
    console.print(title, style=Style(color="green"), justify="center")
    console.print(panel, justify="center")
    console.print("\n")


def load_config(config_module="kurtis.config.default"):
    """
    Load a custom configuration module.

    Args:
        config_module (str): The name of the configuration module to load.
                             Defaults to 'kurtis.config.default'.

    Returns:
        module: The loaded configuration module.
    """
    try:
        # Dynamically import the specified module
        config = importlib.import_module(config_module)
        return config
    except ImportError as e:
        click.echo(f"Error loading configuration module '{config_module}': {e}")
        return None


# https://github.com/pytorch/pytorch/issues/83015
# def get_device():
#     return (
#         "mps"
#         if torch.backends.mps.is_available()
#         else "cuda" if torch.cuda.is_available() else "cpu"
#     )


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_unused_memory():
    """
    Frees unused memory in PyTorch, managing both CPU and GPU environments safely.
    """
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except AttributeError:
            pass
    else:
        pass
    gc.collect()
