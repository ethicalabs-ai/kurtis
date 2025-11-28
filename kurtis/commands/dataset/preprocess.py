# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------

import click


from kurtis.preprocess import preprocessing_main


@click.command(name="preprocess")
@click.option("--max-seq-length", default=1024, help="Max sequence length.")
@click.option("--push", is_flag=True, help="Push dataset to Hugging Face.")
@click.option("--dataset-config", help="Path to YAML dataset config.")
@click.pass_context
def command(ctx, max_seq_length, push, dataset_config):
    debug = ctx.obj["DEBUG"]
    config = ctx.obj["CONFIG"]
    preprocessing_main(
        config,
        max_length=max_seq_length,
        push=push,
        debug=debug,
        dataset_config_path=dataset_config,
    )
