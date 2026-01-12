# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------

import os

import click

from kurtis.train_dpo import clean_dpo_dataset, generate_dpo_dataset


def handle_generate_dpo(debug=False):
    """
    Generate and clean a DPO dataset.
    """
    base_model = "microsoft/Phi-3.5-mini-instruct"
    source_path = os.path.join("datasets", "kurtis_mental_health", "kurtis.parquet")
    target_path = os.path.join("datasets", "kurtis_mental_health_dpo")
    clean_path = os.path.join("datasets", "kurtis_mental_health_dpo_clean")

    generate_dpo_dataset(
        base_model,
        source_path,
        target_path,
        debug=debug,
    )
    clean_dpo_dataset(
        target_path,
        clean_path,
        debug=debug,
    )


@click.command(name="dpo")
@click.pass_context
def command(ctx, max_seq_length, push):
    debug = ctx.obj["DEBUG"]
    handle_generate_dpo(debug=debug)
