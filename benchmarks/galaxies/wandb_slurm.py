import wandb
import subprocess
import click
import yaml
import os
import json

os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY')

@click.command()
@click.argument("config_yaml")
@click.argument("train_file")
@click.argument("project_name")
def run(config_yaml, train_file, project_name):

    wandb.init(project=project_name)
    
    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict['program'] = train_file

    print(config_dict)

    sweep_id = wandb.sweep(config_dict, project=project_name)
    wandb.agent(sweep_id)


if __name__ == '__main__':
    run()