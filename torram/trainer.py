import collections
import logging
import os
import randomname
import torch
import torram
import torch.nn.functional
import torch.utils.data

from torch.utils.data import Dataset
from typing import Dict, Optional, Protocol


class ModelProtocol(Protocol):  # pragma: no cover

    def wbatch(self, batch):
        ...

    def compute_loss(self, batch, model_output) -> Dict[str, torch.Tensor]:
        ...

    def evaluate(self, batch, model_output, reduce_mean: bool) -> Dict[str, torch.Tensor]:
        ...

    # Optional methods
    # def log_values(self, batch, model_output, logger, global_step: int, prefix: str = ""):
    #     ...
    # def visualize(self, batch, model_output, logger, global_step: int, prefix: str = ""):
    #     ...


class Trainer:  # pragma: no cover

    def __init__(self, model: ModelProtocol, ds_train: Dataset, ds_test: Dataset, device: torch.device,
                 batch_size: int = 24, batch_size_test: int = 8, pin_memory: bool = True, num_workers: int = 4,
                 learning_rate: float = 0.0001, log_level: int = 30, log_directory: Optional[str] = None,
                 log_steps_train: int = 100, log_steps_val: int = 500, log_steps_test: int = 5000):
        """Generic neural network trainer based on the instructions implemented in the model.

        This trainer class implements the boilerplate required to train most models in PyTorch, such as
        data loading, back-propagation, validation/testing steps, model storing. All model-specific functions
        are implemented in the model (see ModelProtocol).

        >>> modelx = ...
        >>> dataset_train, dataset_test = ...
        >>> trainer = Trainer(modelx, dataset_train, dataset_test, device=device, ...)
        >>> trainer.fit()

        Args:
            model: model to train, see ModelProtocol.
            ds_train: training dataset, subclass of torch.utils.data.Dataset. also used for "validation".
            ds_test: testing dataset, subclass of torch.utils.data.Dataset.
            device: training device.
            batch_size: training batch size, see torch.utils.data.DataLoader.
            batch_size_test: testing batch size, see torch.utils.data.DataLoader.
            pin_memory: reserve memory on device, see torch.utils.data.DataLoader.
            num_workers: number of workers for training data loading, see torch.utils.data.DataLoader.
            learning_rate: (constant) learning rate.
            log_level: logging level (DEBUG = 10, INFO = 20, WARNING = 30).
            log_directory: path to where logs and model should be stored, logging to terminal when None.
            log_steps_train: number of global training steps between logging training data (loss, runtime, etc.).
            log_steps_val: number of global training steps between validation run.
            log_steps_test: number of global training steps between testing run.
        """
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Invalid input model, must be torch.nn.Module")
        self.model = model
        self.device = device

        self.train_data_loader = torch.utils.data.DataLoader(ds_train,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             num_workers=num_workers,
                                                             pin_memory=pin_memory,
                                                             collate_fn=getattr(ds_train, "collate_fn", None))
        self.test_data_loader = torch.utils.data.DataLoader(ds_test,
                                                            batch_size=batch_size_test,
                                                            shuffle=True,
                                                            pin_memory=pin_memory,
                                                            collate_fn=getattr(ds_test, "collate_fn", None))

        self.evaluator = torram.utility.EvaluatorFactory(model.evaluate, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.log_name = randomname.get_name()
        log_format = '[%(asctime)s.%(msecs)03d %(levelname)s] %(message)s'
        logging.basicConfig(format=log_format, datefmt='%H:%M:%S', level=log_level)
        self.log_steps = {"train": log_steps_train, "val": log_steps_val, "test": log_steps_test}
        if log_directory is not None:
            log_dir = os.path.join(log_directory, self.log_name)
            self.logger = torram.utility.logger.TensorboardY(log_dir)
        else:
            self.logger = torram.utility.logger.LogLogger()

    ###################################################################################################################
    # Training Pipeline ###############################################################################################
    ###################################################################################################################
    def fit(self, num_global_steps: int = 10000):
        global_step = 0
        epoch = 0
        batch_size = self.train_data_loader.batch_size

        loss_running = 0
        steps_cache = collections.defaultdict(int)
        loss_cache = collections.defaultdict(int)

        logging.debug("Initial test and visualization step for checking if everything works")
        self.test_step(global_step=global_step)
        self.model.train()
        while global_step < num_global_steps:
            self.logger.add_scalar("epoch", epoch, global_step=global_step)
            timer = torram.utility.Timer()

            for batch in self.train_data_loader:
                logging.debug(f"Starting batch with global step {global_step} (epoch = {epoch})")
                timer.log_dt("data-loading")
                batch = torram.utility.moving.move_batch(batch, device=self.device)
                timer.log_dt("moving-data")

                logging.debug("Model forward pass of loaded batch")
                model_output = self.forward_batch(batch)
                timer.log_dt("model-forward-pass")

                logging.debug("Computing and logging loss for training and validation")
                loss_dict = self.model.compute_loss(batch, model_output)
                loss = sum(loss_dict.values())
                loss_cache = {key: loss_cache[key] + value for key, value in loss_dict.items()}
                loss_running += loss
                timer.log_dt("compute-loss")
                logging.debug(f"... got loss values {loss_dict}")

                logging.debug("Executing model backward pass")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                timer.log_dt("model-backward-pass")

                if steps_cache["log"] - self.log_steps['train'] > 0:
                    value_count = steps_cache["log"] / batch_size
                    steps_cache["log"] = 0

                    logs_dt = timer.get_and_reset_logs()
                    logs_dt = {key: sum(values) / len(values) for key, values in logs_dt.items()}  # normalize
                    loss_cache = {key: value / value_count for key, value in loss_cache.items()}  # normalize
                    self.logger.add_scalar_dicts(f"timings", logs_dt, global_step=global_step)
                    self.logger.add_scalar_dicts("loss/train", loss_cache, global_step=global_step)
                    self.logger.add_scalar("loss/train", loss_running / value_count, global_step=global_step)

                    loss_running = 0
                    loss_cache = collections.defaultdict(int)
                    timer.log_dt("logging-log-data")

                if steps_cache["val"] - self.log_steps['val'] > 0:
                    steps_cache["val"] = 0
                    with torch.no_grad():
                        self.validation_step(model_output, batch, global_step=global_step)
                        timer.log_dt("evaluating-val-data")

                if steps_cache["test"] - self.log_steps['test'] > 0:
                    steps_cache["test"] = 0
                    with torch.no_grad():
                        self.test_step(global_step=global_step)
                        timer.log_dt("evaluating-test-data")
                        self.save_model(global_step=global_step)
                        timer.log_dt("saving-model")

                global_step += batch_size
                steps_cache = {key: value + batch_size for key, value in steps_cache.items()}
                self.logger.flush()
                timer.log_dt("finishing-batch")
            epoch += 1
        self.logger.close()

    def forward_batch(self, batch):
        return self.model.wbatch(batch)

    def validation_step(self, model_output, batch, global_step: int):
        self.model.eval()
        metrics_dict = self.evaluator(batch, model_output, reduce_mean=True)
        self.logger.add_scalar_dicts("metrics/train", metrics_dict, global_step=global_step)
        if hasattr(self.model, "log_values"):
            self.model.log_values(batch, model_output, self.logger, global_step=global_step)
        self.model.train()

    def test_step(self, global_step: int):
        self.model.eval()
        metrics_dict = self.evaluator.wloader(self.forward_batch, self.test_data_loader, until=10)
        self.logger.add_scalar_dicts("metrics/test", metrics_dict, global_step=global_step)

        vis_batch = next(self.test_data_loader.__iter__())  # should yield first batch
        vis_batch = torram.utility.moving.move_batch(vis_batch, device=self.device)
        with torch.no_grad():
            model_output = self.forward_batch(vis_batch)

        loss_dict = self.model.compute_loss(vis_batch, model_output)
        self.logger.add_scalar_dicts("loss/test", loss_dict, global_step=global_step)
        self.logger.add_scalar("loss/test", sum(loss_dict.values()), global_step=global_step)
        if hasattr(self.model, "visualize"):
            vis_prefix = f"{self.log_name}/"
            self.model.visualize(vis_batch, model_output, self.logger, global_step=global_step, prefix=vis_prefix)
        self.model.train()

    def save_model(self, global_step: int):
        checkpoint_path = os.path.join(self.logger.log_dir, "checkpoints", f"{global_step}.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
