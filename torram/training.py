import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Type, cast

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from torram.utils.config import read_config
from torram.utils.dataset_utils import train_test_split
from torram.utils.ops import to_device_dict


@dataclass
class OptimizerConfig:
    num_epochs: int
    lr: float
    accumulate_grad_batches: int = 1


@dataclass
class LoggingConfig:
    num_ckpt_iterations: int
    num_test_iterations: int
    log_project: str


@dataclass
class DataConfig:
    num_workers: int
    batch_size: int
    test_split: float


class TrainingConfig(Protocol):
    data: DataConfig
    optimizer: OptimizerConfig
    logging: LoggingConfig


class TrainerSchema(Protocol):

    def __init__(self, config: Any):
        raise NotImplementedError

    def get_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step and return a dictionary of losses."""
        raise NotImplementedError

    def visualize(self, batch: Dict[str, torch.Tensor], n: int = 4) -> Dict[str, Any]:
        """Visualize a batch of data and return an image array."""
        raise NotImplementedError

    def evaluate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate the model on a batch and return a dictionary of metrics."""
        raise NotImplementedError

    def parameters(self):
        """Return the model trainable parameters."""
        raise NotImplementedError

    def train(self):
        """Set the model to training mode."""
        raise NotImplementedError

    def eval(self):
        """Set the model to evaluation mode."""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Return the model state dictionary."""
        raise NotImplementedError

    def to(self, device: torch.device):
        """Move the model to the specified device."""
        raise NotImplementedError


class DatasetSchema(Protocol):

    def __init__(self, config: Any):
        raise NotImplementedError


def train(
    config_schema: Type[TrainingConfig],
    trainer_class: Type[TrainerSchema],
    dataset_class: Type[DatasetSchema],
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--debug", action="store_true")
    args, args_unknown = parser.parse_known_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)
    config = read_config(args.config, config_schema, args_unknown)
    config = cast(TrainingConfig, config)
    logger.info(f"Using config: \n{OmegaConf.to_yaml(config)}")

    # Initialize the trainer/model and its default config parameters based on the model type.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    trainer = trainer_class(config)
    trainer.to(device)

    # Initialize WandB for logging.
    if args.disable_wandb:
        wandb.init(mode="disabled")
    else:
        config_dict = OmegaConf.to_container(config, resolve=True)
        config_dict = cast(dict, config_dict)
        wandb.init(project=config.logging.log_project, config=config_dict)

    # Setup dataset and dataloader.
    dataset = dataset_class(getattr(config, "dataset", None))
    dataset_train, dataset_test = train_test_split(dataset, test_ratio=config.data.test_split)
    dataloader_train = DataLoader(
        dataset_train,
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
        shuffle=True,
    )
    dataloader_test = DataLoader(
        dataset_test,
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
        shuffle=False,
    )

    # Setup the optimizer.
    optimizer = trainer.get_optimizer()

    # Print the number of trainable and total parameters in the model.
    num_params = sum(p.numel() for p in trainer.parameters())
    num_params_trainable = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params} parameters, {num_params_trainable} trainable.")

    # Run the training loop.
    trainer.train()
    global_step = 0
    for epoch in range(config.optimizer.num_epochs):
        total_loss = 0.0
        for batch in dataloader_train:
            batch = to_device_dict(batch, device=device)
            loss_dict = trainer.compute_loss(batch)
            loss = sum(loss_dict.values())
            loss = cast(torch.Tensor, loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            global_step += 1
            wandb.log(
                {
                    "loss/train/step": loss.item(),
                    "training/learning_rate": config.optimizer.lr,
                    **{f"loss/train/{k}": v.item() for k, v in loss_dict.items()},
                },
                step=global_step,
            )

            if global_step % config.logging.num_test_iterations == 0:
                logger.info(f"Epoch {epoch}, Step {global_step}, Train Loss: {loss.item()}")
                trainer.eval()

                metrics_train_dict = trainer.evaluate(batch)
                vis_train = trainer.visualize(batch, n=4)

                loss_val_dict = defaultdict(float)
                metrics_val_dict = defaultdict(float)
                for batch in dataloader_test:
                    batch = to_device_dict(batch, device=device)

                    loss_dict = trainer.compute_loss(batch)
                    loss = sum(loss_dict.values())
                    loss = cast(torch.Tensor, loss)
                    for k, v in loss_dict.items():
                        loss_val_dict[k] += v.item()

                    metrics_dict = trainer.evaluate(batch)
                    for k, v in metrics_dict.items():
                        metrics_val_dict[k] += v

                loss_val_dict["epoch"] = sum(loss_val_dict.values())
                num_test_samples = len(dataloader_test)
                loss_val_dict = {k: v / num_test_samples for k, v in loss_val_dict.items()}
                metrics_val_dict = {k: v / num_test_samples for k, v in metrics_val_dict.items()}
                wandb.log(
                    {
                        **{f"metrics/train/{k}": v for k, v in metrics_train_dict.items()},
                        **{f"metrics/val/{k}": v for k, v in metrics_val_dict.items()},
                        **{f"loss/val/{k}": v for k, v in loss_val_dict.items()},
                        **{f"visualization/train/{k}": v for k, v in vis_train.items()},
                    },
                    step=global_step,
                )
                trainer.train()

            if global_step % config.logging.num_ckpt_iterations == 0 and not args.disable_wandb:
                assert wandb.run is not None, "WandB run must be initialized to save checkpoints."
                logger.info(f"Saving checkpoint at step {global_step}")
                state = {
                    "model": trainer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                }
                torch.save(state, os.path.join(wandb.run.dir, f"checkpoint_{global_step}.pt"))

        avg_loss = total_loss / len(dataloader_train)
        wandb.log({"training/epoch": epoch, "loss/train/epoch": avg_loss}, step=global_step)
