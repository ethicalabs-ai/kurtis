# kurtis/utils.py

import gc
import importlib
from types import ModuleType
from typing import Optional, Tuple

import click
import pyfiglet
import torch
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from nltk.tokenize import sent_tokenize
from transformers import PreTrainedTokenizer


def get_kurtis_title() -> Tuple[str, str]:
    """
    Returns the ASCII art title and panel text for Kurtis.

    Returns:
        tuple: (title_art (str), panel_text (str))
    """
    title_art: str = pyfiglet.figlet_format("Kurtis", font="banner3-D")
    panel_text: str = "Kurtis v0.1.0"
    return title_art, panel_text


def print_kurtis_title(console: Optional[Console] = None) -> None:
    """
    Prints the Kurtis title and panel using Rich.

    Args:
        console (Console, optional): A Rich Console instance. If not provided,
                                     a new Console will be created.
    """
    if console is None:
        console = Console()
    title_art, panel_text = get_kurtis_title()
    panel = Panel.fit(
        panel_text,
        border_style="bright_green",
        padding=(1, 4),
    )
    console.print("\n")
    console.print(title_art, style=Style(color="green"), justify="center")
    console.print(panel, justify="center")
    console.print("\n")


def load_config(config_module: str = "kurtis.config.default") -> Optional[ModuleType]:
    """
    Load a custom configuration module.

    Args:
        config_module (str): The name of the configuration module to load.
                             Defaults to 'kurtis.config.default'.

    Returns:
        ModuleType or None: The loaded configuration module, or None if an error occurs.
    """
    try:
        # Dynamically import the specified module
        config: ModuleType = importlib.import_module(config_module)
        return config
    except ImportError as e:
        click.echo(f"Error loading configuration module '{config_module}': {e}")
        return None


def get_device() -> str:
    """
    Returns the device to use based on CUDA availability.

    Returns:
        str: "cuda" if CUDA is available, otherwise "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_unused_memory() -> None:
    """
    Frees unused memory in PyTorch, handling both GPU and CPU environments.
    """
    if torch.cuda.is_available() and torch.backends.cuda.is_built():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except AttributeError:
            pass
    gc.collect()


# Intelligent sentence splitting and truncating
def clean_and_truncate(
    text: str, max_length: int, tokenizer: PreTrainedTokenizer
) -> str:
    text = text.strip()
    sentences = sent_tokenize(text)  # Split the text into sentences
    truncated_text = ""
    total_length = 0

    for sentence in sentences:
        # Encode the sentence to get its token length
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))

        # Check if adding this sentence would exceed the max_length
        if total_length + sentence_length > max_length:
            break  # Stop if adding the sentence would exceed the token limit

        # Add the sentence to the truncated text
        truncated_text += sentence + " "
        total_length += sentence_length

    return truncated_text.strip()  # Return the truncated text without extra spaces
