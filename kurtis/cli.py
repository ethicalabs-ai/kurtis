# --------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             It does not contain AI-generated content or automated
#             code synthesis. Any imperfections are responsibly owned
#             by its human creators.
# --------------------------------------------------------------------
# From: https://stackoverflow.com/questions/50751040/include-submodules-on-click

import click
import importlib
import pkgutil
import os.path

from kurtis.utils import load_config, print_kurtis_title


def get_commands_from_pkg(pkg) -> dict:
    pkg_obj = importlib.import_module(pkg)

    pkg_path = os.path.dirname(pkg_obj.__file__)

    commands = {}
    for module in pkgutil.iter_modules([pkg_path]):
        module_obj = importlib.import_module(f"{pkg}.{module.name}")
        if not module.ispkg:
            commands[module_obj.command.name] = module_obj.command

        else:
            commands[module.name.replace("_", "-")] = click.Group(
                context_settings={"help_option_names": ["-h", "--help"]},
                help=module_obj.__doc__,
                commands=get_commands_from_pkg(f"{pkg}.{module.name}"),
            )

    return commands


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Kurtis Toolkit",
    commands=get_commands_from_pkg("kurtis.commands"),
)
@click.pass_context
@click.option(
    "--config-module",
    "-c",
    default="kurtis.config.default",
    help="Kurtis python config module.",
)
@click.option("--debug/--no-debug", default=False)
def cli(ctx, config_module, debug):
    ctx.ensure_object(dict)
    print_kurtis_title()
    click.echo(f"Debug mode is {'on' if debug else 'off'}")
    ctx.obj["DEBUG"] = debug
    ctx.obj["CONFIG"] = load_config(config_module)


if __name__ == "__main__":
    cli()
