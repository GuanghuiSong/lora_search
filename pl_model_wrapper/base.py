import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, MeanMetric
import pytorch_lightning as pl
from datasets import DatasetInfo
from transformers import PreTrainedModel

from tools.log_shortcut_weights import (
    get_opt_layer_res_shortcut_svd,
    get_roberta_layer_res_shortcut_svd,
)


class PlWrapperBase(pl.LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        optimizer: str = None,
        learning_rate=5e-4,  # for building optimizer
        weight_decay=0.0,  # for building optimizer
        lr_scheduler: str = "none",  # for building lr scheduler
        eta_min=0.0,  # for building lr scheduler
        epochs=1,  # for building lr_scheduler
        dataset_info: DatasetInfo = None,  # for getting num_classes for calculating Accuracy
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.eta_min = eta_min
        self.epochs = epochs

        self.loss_fn = nn.CrossEntropyLoss()

        assert "label" in dataset_info.features.keys()
        self.num_classes = dataset_info.features["label"].num_classes

        # train step metrics are logged in every step
        self.acc_train = Accuracy("multiclass", num_classes=self.num_classes)

        # validation metrics are logged when epoch ends
        self.acc_val = Accuracy("multiclass", num_classes=self.num_classes)
        self.loss_val = MeanMetric()

        # test metrics are logged when epoch ends
        self.acc_test = Accuracy("multiclass", num_classes=self.num_classes)
        self.loss_test = MeanMetric()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)

        self.acc_train(y_pred, y)

        self.log("train_acc_step", self.acc_train, prog_bar=True)
        self.log("train_loss_step", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)

        self.acc_val(y_pred, y)
        self.loss_val(loss)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.acc_val, prog_bar=True)
        self.log("val_loss_epoch", self.loss_val, prog_bar=True)

        print("\n", self.model.__class__.__name__)
        if "Ags" not in self.model.__class__.__name__:
            return
        # Log shortcut weights' singular values and unevenness metrics
        if "OPT" in self.model.__class__.__name__:
            print(f"Epoch {self.current_epoch} getting singular values...")
            singular_uneven = get_opt_layer_res_shortcut_svd(self.model)
            print(singular_uneven)
        elif "Roberta" in self.model.__class__.__name__:
            singular_uneven = get_roberta_layer_res_shortcut_svd(self.model)
        else:
            # TODO: accommodate more models
            raise ValueError(
                f"Model {self.model.__class__.__name__} not supported for logging shortcut singular values"
            )
        self.log_dict(singular_uneven)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)

        self.acc_test(y_pred, y)
        self.loss_test(loss)
        return loss

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.acc_test, prog_bar=True)
        self.log("test_loss_epoch", self.loss_test, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        y_pred = self.forward(x)
        return {"batch_idx": batch_idx, "pred_y": y_pred}

    def configure_optimizers(
        self,
    ) -> (
        torch.optim.Optimizer
        | dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler]
    ):
        # TODO: add back LR scheduler
        scheduler = None
        # Use self.trainer.model.parameters() instead of self.parameters() to support FullyShared (Model paralleled) training
        if self.optimizer == "adamw":
            opt = torch.optim.AdamW(
                self.trainer.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            match self.lr_scheduler:
                case "none":
                    scheduler = None
                case "cosine_annealing":
                    scheduler = CosineAnnealingLR(
                        opt, T_max=self.epochs, eta_min=self.eta_min
                    )
                case _:
                    raise ValueError(f"Unsupported lr scheduler: {self.lr_scheduler}")
        elif self.optimizer == "adam":
            opt = torch.optim.Adam(
                self.trainer.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            match self.lr_scheduler:
                case "none":
                    scheduler = None
                case "cosine_annealing":
                    scheduler = CosineAnnealingLR(
                        opt, T_max=self.epochs, eta_min=self.eta_min
                    )
                case _:
                    raise ValueError(f"Unsupported lr scheduler: {self.lr_scheduler}")
        elif self.optimizer in ["sgd_no_warmup", "sgd"]:
            opt = torch.optim.SGD(
                self.trainer.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True,
            )
            if self.optimizer == "sgd":
                match self.lr_scheduler:
                    case "none":
                        scheduler = None
                    case "cosine_annealing":
                        scheduler = CosineAnnealingLR(
                            opt, T_max=self.epochs, eta_min=self.eta_min
                        )
                    case _:
                        raise ValueError(
                            f"Unsupported lr scheduler: {self.lr_scheduler}"
                        )
        else:
            raise ValueError(f"Unsupported optimizer name {self.optimizer}")

        if scheduler is None:
            return {"optimizer": opt}
        else:
            return {"optimizer": opt, "lr_scheduler": scheduler}
