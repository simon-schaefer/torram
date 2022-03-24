import collections
import logging
import os
import randomname
import torch
import torram
import torch.nn.functional
import torch.utils.data

from torch.utils.data import Dataset
from typing import Dict, Protocol


class ModelProtocol(Protocol):  # pragma: no cover

    def wbatch(self, batch):
        ...

    def compute_loss(self, batch, model_output) -> Dict[str, torch.Tensor]:
        ...

    def evaluate(self, batch, model_output, reduce_mean: bool) -> Dict[str, torch.Tensor]:
        ...

    def log_values(self, batch, model_output, logger, global_step: int, prefix: str = ""):
        ...

    def visualize(self, batch, model_output, logger, global_step: int, prefix: str = ""):
        ...


class Trainer:  # pragma: no cover

    def __init__(self, model: ModelProtocol, ds_train: Dataset, ds_test: Dataset, device: torch.device,
                 config: torram.utility.Config = torram.utility.Config.empty()):
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Invalid input model, must be torch.nn.Module")
        self.config = config
        self.model = model
        self.device = device

        self.train_data_loader = torch.utils.data.DataLoader(ds_train,
                                                             batch_size=config.get("data/batch_size/train", 24),
                                                             shuffle=True,
                                                             num_workers=config.get("data/num_workers", 4),
                                                             pin_memory=config.get("data/pin_memory", True))
        self.test_data_loader = torch.utils.data.DataLoader(ds_test,
                                                            batch_size=config.get("data/batch_size/test", 8),
                                                            shuffle=True,
                                                            pin_memory=config.get("data/pin_memory", True))

        self.evaluator = torram.utility.EvaluatorFactory(model.evaluate, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get("hp/learning_rate", 0.00001))

        self.log_name = randomname.get_name()
        log_level = config.get("logging/log_level", 30)  # DEBUG = 10, INFO = 20
        log_directory = config.get("logging/log_directory", required=True)
        log_format = '[%(asctime)s.%(msecs)03d %(levelname)s] %(message)s'
        logging.basicConfig(format=log_format, datefmt='%H:%M:%S', level=log_level)
        log_dir = os.path.join(log_directory, self.log_name)
        self.logger = torram.utility.TensorboardY(log_dir)

    ###################################################################################################################
    # Training Pipeline ###############################################################################################
    ###################################################################################################################
    def fit(self, num_global_steps: int):
        global_step = 0
        epoch = 0
        batch_size = self.train_data_loader.batch_size

        loss_running = 0
        steps_cache = collections.defaultdict(int)
        loss_cache = collections.defaultdict(int)

        while global_step < num_global_steps:
            self.logger.add_scalar("epoch", epoch, global_step=global_step)
            timer = torram.utility.Timer()

            for batch in self.train_data_loader:
                logging.debug(f"Starting batch with global step {global_step} (epoch = {epoch})")
                timer.log_dt("data-loading")
                batch = tuple(x.to(self.device) for x in batch)
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

                if steps_cache["log"] - self.config.get("logging/log_steps", 100) > 0:
                    value_count = steps_cache["log"] / batch_size
                    steps_cache["log"] = 0

                    logs_dt = timer.get_and_reset_logs()
                    logs_dt = {key: sum(values) / len(values) for key, values in logs_dt.items()}  # normalize
                    loss_cache = {key: value / value_count for key, value in loss_cache.items()}  # normalize
                    self.logger.add_scalar_dict(f"timings", logs_dt, global_step=global_step)
                    self.logger.add_scalar_dict("loss/train", loss_cache, global_step=global_step)
                    self.logger.add_scalar("loss/train", loss_running / value_count, global_step=global_step)

                    loss_running = 0
                    loss_cache = collections.defaultdict(int)
                    timer.log_dt("logging-log-data")

                if steps_cache["val"] - self.config.get("logging/val_steps", 500) > 0:
                    steps_cache["val"] = 0
                    with torch.no_grad():
                        self.validation_step(model_output, batch, global_step=global_step)
                        timer.log_dt("evaluating-val-data")

                if steps_cache["test"] - self.config.get("logging/test_steps", 5000) > 0:
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
        metrics_dict = self.evaluator(model_output, batch, reduce_mean=True)
        self.logger.add_scalar_dict("metrics/train", metrics_dict, global_step=global_step)
        if hasattr(self.model, "log_values"):
            self.model.log_values(batch, model_output, self.logger, global_step=global_step)
        self.model.train()

    def test_step(self, global_step: int):
        self.model.eval()
        metrics_dict = self.evaluator.wloader(self.forward_batch, self.test_data_loader, until=10)
        self.logger.add_scalar_dict("metrics/test", metrics_dict, global_step=global_step)

        vis_batch = next(self.test_data_loader.__iter__())  # should yield first batch
        vis_batch = tuple(x.to(self.device) for x in vis_batch)
        with torch.no_grad():
            model_output = self.forward_batch(vis_batch)

        loss_dict = self.model.compute_loss(vis_batch, model_output)
        self.logger.add_scalar("loss/test", sum(loss_dict.values()), global_step=global_step)
        if hasattr(self.model, "visualize"):
            vis_prefix = f"{self.log_name}/"
            self.model.visualize(vis_batch, model_output, self.logger, global_step=global_step, prefix=vis_prefix)
        self.model.train()

    def save_model(self, global_step: int):
        checkpoint_path = os.path.join(self.logger.log_dir, "checkpoints", f"{global_step}.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
