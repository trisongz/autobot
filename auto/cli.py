
import json
import sys
import os
import typer
from typing import List
import time
from auto.utils.logger import get_logger
from auto.utils.core import run_command
from auto.utils.check_imports import check_imports

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = get_logger()

cli = typer.Typer()
monitor_app = typer.Typer()
cli.add_typer(monitor_app, name='monitor')
sess_app = typer.Typer()
cli.add_typer(sess_app, name='sess')
logging_app = typer.Typer()
cli.add_typer(logging_app, name='logging')
training_app = typer.Typer()
cli.add_typer(logging_app, name='train')

@sess_app.command('new')
def sess_new(name: str = typer.Argument("train")):
    _conda_exe = os.getenv('CONDA_EXE').replace('bin/conda', 'etc/profile.d/conda.sh')
    _conda_env = os.getenv('CONDA_DEFAULT_ENV', None)
    command = f'tmux new -d -s {name}'
    os.system(command)
    if _conda_env:
        command = f'tmux send-keys -t {name}.0 "source {_conda_exe} && conda deactivate && conda activate {_conda_env} && clear && cd {os.getcwd()}" ENTER'
        os.system(command)
    os.system(f'tmux a -t {name}')
    typer.echo(f'Created new tmux session called {name}. Use "autobot sess attach {name}" to enter the session or "autobot sess resume" to enter the last created session.')

@sess_app.command('attach')
def sess_attach(name: str = typer.Argument("train")):
    command = f'tmux a -t {name}'
    os.system(command)

@sess_app.command('kill')
def sess_kill(name: str = typer.Argument("train")):
    typer.echo(f'Ending Session: {name}')
    command = f'tmux kill-session -t {name}'
    os.system(command)

@sess_app.command('resume')
def sess_resume():
    command = f'tmux attach-session'
    os.system(command)

@sess_app.command('list')
def sess_list():
    command = f'tmux ls'
    ls = run_command(command)
    typer.echo(f'Sessions: {ls}')


@cli.command('setup')
def setup_auto():
    chk_libs = ['tensorflow', 'torch', 'transformers']
    typer.echo(f'Setting Up Libraries and Checking Installed')
    installed_libs = check_imports()
    for lib in chk_libs:
        _is_installed = f'{lib} - {installed_libs[lib]} is installed' if installed_libs[lib] else f'{lib} is not not installed'
        typer.echo(_is_installed)
        if typer.confirm(f"Update {lib}?"):
            os.system(f'pip install -q --upgrade {lib}')


if __name__ == "__main__":
    cli()