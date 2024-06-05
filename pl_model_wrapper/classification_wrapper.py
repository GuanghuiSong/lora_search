import torch
from transformers import PreTrainedModel

from dataset import AgsDatasetInfo
from .base import PlWrapperBase
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

# Sentence classification
class NLPClassificationModelWrapper(PlWrapperBase):
    def __init__(
        self,
        model: PreTrainedModel,
        optimizer: str = None,
        learning_rate=1e-4,  # for building optimizer
        weight_decay=0.0,  # for building optimizer
        lr_scheduler: str = "none",  # for building lr scheduler
        eta_min=0.0,  # for building lr scheduler
        epochs=200,  # for building lr_scheduler
        dataset_info: AgsDatasetInfo = None,  # for getting num_classes for calculating Accuracy
    ):
        super().__init__(
            model,
            optimizer,
            learning_rate,
            weight_decay,
            lr_scheduler,
            eta_min,
            epochs,
            dataset_info,
        )

    def forward(
        self,
        input_ids,  # ids: tokenizer(token) -> an id in its word list
        attention_mask=None,  # to prevent applying attention to padding characters
        token_type_ids=None,  # when multiple sentences are concatenated into the input, this indicate which sentence each character belongs to
        labels=None,
    ):
        if isinstance(input_ids, list):
            input_ids = torch.stack(input_ids)
        if isinstance(attention_mask, list):
            attention_mask = torch.stack(attention_mask)
        if isinstance(token_type_ids, list):
            token_type_ids = torch.stack(token_type_ids)
        if isinstance(labels, list):
            labels = torch.stack(labels)

        if token_type_ids is not None:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        return outputs
    # def pearson_and_spearman(preds, labels):
    #     pearson_corr = pearsonr(preds, labels)[0]
    #     spearman_corr = spearmanr(preds, labels)[0]
    #     return {
    #         "pearson": pearson_corr,
    #         "spearmanr": spearman_corr,
    #         "corr": (pearson_corr + spearman_corr) / 2,
    #     }
    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)

        outputs = self.forward(x, attention_mask, token_type_ids, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        _pred_logits, pred_ids = torch.max(logits, dim=1)
        y = labels[0] if len(labels) == 1 else labels.squeeze()

        self.acc_train(pred_ids, y)
        self.f1_train(pred_ids, y)

        self.log("train_loss_step", loss, prog_bar=True)
        self.log("train_acc_step", self.acc_train, prog_bar=True)
        self.log("train_f1_step", self.f1_train, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)

        outputs = self.forward(x, attention_mask, token_type_ids, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        _pred_logits, pred_ids = torch.max(logits, dim=1)
        y = labels[0] if len(labels) == 1 else labels.squeeze()

        self.acc_val(pred_ids, y)
        self.f1_val(pred_ids, y)

        self.matthews_corrcoef_val(pred_ids, y)
        self.pearson_corrcoef_val(pred_ids.float(), y.float())
        self.spearman_Corrcoef_val(pred_ids.float(), y.float())
        self.loss_val(loss)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss_epoch", self.loss_val, prog_bar=True)
        self.log("val_acc_epoch", self.acc_val, prog_bar=True)
        self.log("val_f1_epoch", self.f1_val, prog_bar=True)
        self.log("val_f1acc_epoch", (self.f1_val.compute() + self.acc_val.compute())/2, prog_bar=True, sync_dist=True)
        self.log("matthews_corrcoef_val", self.matthews_corrcoef_val, prog_bar=True)
        self.log("pearson-spearman_val", (self.pearson_corrcoef_val.compute() + self.spearman_Corrcoef_val.compute())/2, prog_bar=True, sync_dist=True)
        # self.log("pearson_corrcoef_val", self.pearson_corrcoef_val, prog_bar=True)
        # self.log("spearman_Corrcoef_val", self.spearman_Corrcoef_val, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)

        outputs = self.forward(x, attention_mask, token_type_ids, labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        _pred_logits, pred_ids = torch.max(logits, dim=1)
        y = labels[0] if len(labels) == 1 else labels.squeeze()

        self.acc_test(pred_ids, y)
        self.f1_test(pred_ids, y)
        self.loss_test(loss)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_loss_epoch", self.loss_test, prog_bar=True)
        self.log("test_acc_epoch", self.acc_test, prog_bar=True)
        self.log("test_f1_epoch", self.f1_test, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_type_ids = batch.get("token_type_ids", None)

        outputs = self.forward(x, attention_mask, token_type_ids, labels)
        logits = outputs["logits"]
        _pred_logits, pred_ids = torch.max(logits, dim=1)

        return {"batch_idx": batch_idx, "outputs": outputs, "pred_ids": pred_ids}
