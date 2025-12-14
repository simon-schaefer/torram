import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sized, Type, Union, cast

import torch
import torch.utils.data
import wandb
from omegaconf import OmegaConf

from torram.utils.config import read_config
from torram.utils.dataset_utils import (
    CustomSchemeDataLoader,
    DataConfig,
    DatasetSchema,
    ExtendedDatasetSchema,
    get_train_test_split_w_config,
)
from torram.utils.ops import to_device_dict
from torram.utils.wandb_utils import load_checkpoint_from_wandb


@dataclass
class OptimizerConfig:
    num_epochs: int
    lr: float
    accumulate_grad_batches: int = 1
    clip_grad_norm: Optional[float] = None


@dataclass
class LoggingConfig:
    num_ckpt_iterations: int
    num_test_iterations: int
    log_project: str


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

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the model state dictionary."""
        raise NotImplementedError

    def to(self, device: torch.device):
        """Move the model to the specified device."""
        raise NotImplementedError


def train(
    config_schema: Type[TrainingConfig],
    trainer_class: Type[TrainerSchema],
    dataset_class: Union[
        Type[DatasetSchema],
        Type[ExtendedDatasetSchema],
        Callable[[Any], torch.utils.data.Dataset],
    ],
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+")
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training.",
    )
    parser.add_argument("--debug", action="store_true")
    args, args_unknown = parser.parse_known_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    config = read_config(args.config, config_schema, args_unknown)
    config = cast(TrainingConfig, config)
    logger.info(f"Using config: \n{OmegaConf.to_yaml(config)}")

    # Initialize the trainer/model and its default config parameters based on the model type.
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
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
    dataset = cast(torch.utils.data.Dataset, dataset)
    dataset_train, datasets_test = get_train_test_split_w_config(dataset, config.data)
    assert isinstance(dataset_train, Sized) and len(dataset_train) > 0, "Training dataset is empty."
    assert len(datasets_test) > 0, "No testing datasets found."
    assert all((isinstance(ds, Sized) and len(ds)) > 0 for ds in datasets_test.values())

    dataloader_train = CustomSchemeDataLoader(
        dataset_train,
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
        shuffle=True,
    )
    dataloaders_test = {
        key: CustomSchemeDataLoader(
            ds,
            num_workers=config.data.num_workers,
            batch_size=config.data.batch_size,
            shuffle=False,
            drop_last=False,
        )
        for key, ds in datasets_test.items()
    }
    dataset_test_sizes = {key: len(ds) for key, ds in datasets_test.items()}
    logger.info(f"Training samples: {len(dataset_train)}, Testing samples: {dataset_test_sizes}")

    # Setup the optimizer.
    optimizer = trainer.get_optimizer()

    # If a checkpoint is provided, load it.
    global_step = 0
    if args.checkpoint is not None:
        checkpoint = load_checkpoint_from_wandb(args.checkpoint, device=device)
        trainer.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint.get("global_step", 0)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Print the number of trainable and total parameters in the model.
    num_params = sum(p.numel() for p in trainer.parameters())
    num_params_trainable = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params} parameters, {num_params_trainable} trainable.")

    # Run the training loop.
    trainer.train()
    accumulate_steps = config.optimizer.accumulate_grad_batches
    for epoch in range(config.optimizer.num_epochs):
        total_loss = 0.0
        for batch in dataloader_train:
            logger.debug(f"Training step {global_step}")
            batch = to_device_dict(batch, device=device)
            loss_dict = trainer.compute_loss(batch)
            loss = sum(loss_dict.values())
            loss = cast(torch.Tensor, loss)

            scaled_loss = loss / config.optimizer.accumulate_grad_batches
            scaled_loss.backward()
            total_loss += loss.item()

            # Gradient accumulation and optimizer step.
            accumulate_steps -= 1
            if accumulate_steps == 0:
                if config.optimizer.clip_grad_norm is not None:
                    assert config.optimizer.clip_grad_norm > 0.0
                    grad_norm = config.optimizer.clip_grad_norm
                    torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=grad_norm)

                optimizer.step()
                global_step += 1
                wandb.log(
                    {
                        "loss/train/step": loss.item(),
                        "training/learning_rate": config.optimizer.lr,
                        **{f"loss/train/{k}": v.item() for k, v in loss_dict.items()},
                    },
                    step=global_step,
                )
                loss_log_dict = {k: round(v.item(), 4) for k, v in loss_dict.items()}
                logger.debug(f"Loss dict: {loss_log_dict}")

                accumulate_steps = config.optimizer.accumulate_grad_batches
                optimizer.zero_grad()

            # Evaluation and visualization on train / test datasets.
            if global_step % config.logging.num_test_iterations == 0 and global_step > 0:
                logger.info(f"Epoch {epoch}, Step {global_step}, Train Loss: {loss.item()}")
                trainer.eval()

                # Evaluate on training batch.
                metrics_train_dict = trainer.evaluate(batch)
                vis_train = trainer.visualize(batch, n=4)

                # Evaluate on test datasets.
                loss_val_dict, metrics_val_dict = {}, {}
                vis_test = {}
                for dataset_name, dataloader_test in dataloaders_test.items():
                    loss_val_ds = defaultdict(float)
                    metrics_val_ds = defaultdict(float)
                    num_samples = len(dataloader_test)
                    prefix = f"{dataset_name}/" if len(dataloaders_test) > 1 else ""

                    for bi, batch in enumerate(dataloader_test):
                        batch = to_device_dict(batch, device=device)
                        if bi == 0:
                            vis_batch = trainer.visualize(batch, n=4)
                            vis_batch = {f"{prefix}{k}": v for k, v in vis_batch.items()}
                            vis_test.update(vis_batch)

                        with torch.no_grad():
                            loss_dict = trainer.compute_loss(batch)
                        loss = sum(loss_dict.values())
                        loss = cast(torch.Tensor, loss)
                        for k, v in loss_dict.items():
                            loss_val_ds[f"{prefix}{k}"] += v.item()

                        metrics_dict = trainer.evaluate(batch)
                        for k, v in metrics_dict.items():
                            metrics_val_ds[f"{prefix}{k}"] += v

                    loss_val_ds["epoch"] = sum(loss_val_ds.values())
                    loss_val_dict.update({k: v / num_samples for k, v in loss_val_ds.items()})
                    metrics_val_dict.update({k: v / num_samples for k, v in metrics_val_ds.items()})

                logger.debug(f"Training Metrics dict: {metrics_train_dict}")
                logger.debug(f"Validation Loss dict: {loss_val_dict}")
                logger.info(f"Validation Metrics dict: {metrics_val_dict}")
                wandb.log(
                    {
                        **{f"metrics/train/{k}": v for k, v in metrics_train_dict.items()},
                        **{f"metrics/val/{k}": v for k, v in metrics_val_dict.items()},
                        **{f"loss/val/{k}": v for k, v in loss_val_dict.items()},
                        **{f"visualization/train/{k}": v for k, v in vis_train.items()},
                        **{f"visualization/val/{k}": v for k, v in vis_test.items()},
                    },
                    step=global_step,
                )
                trainer.train()

            # Save checkpoint of model and training state.
            if (
                global_step % config.logging.num_ckpt_iterations == 0
                and not args.disable_wandb
                and global_step > 0
            ):
                assert wandb.run is not None, "WandB run must be initialized to save checkpoints."
                logger.info(f"Saving checkpoint at step {global_step}")
                state = {
                    "model": trainer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": OmegaConf.to_container(config, resolve=True),
                }
                checkpoint_path = os.path.join(wandb.run.dir, f"checkpoint_{global_step:09d}.pt")
                torch.save(state, checkpoint_path)
                wandb.save(checkpoint_path)

            # Update data loader iterations.
            dataloader_train.set_iteration(global_step)
            for dataloader_test in dataloaders_test.values():
                dataloader_test.set_iteration(global_step)

        avg_loss = total_loss / len(dataloader_train)
        wandb.log({"training/epoch": epoch, "loss/train/epoch": avg_loss}, step=global_step)
