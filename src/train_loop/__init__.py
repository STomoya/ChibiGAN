"""Training loops
"""

from src import utils
from src.train_loop import chibigan, harmonizer


def launch():
    """launch training loop."""
    config = utils.get_hydra_config('config', 'config.yaml')
    if config.config.train_method == 'harmonizer':
        train_fn = harmonizer.train
    elif config.config.train_method == 'chibigan':
        train_fn = chibigan.train
    else:
        raise Exception(f'Unrecognized training method: {config.config.training_method}')

    train_fn(config)
