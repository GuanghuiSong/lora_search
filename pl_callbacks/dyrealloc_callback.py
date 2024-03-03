import math
from typing import Any, Optional

import numpy as np
import toml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.pl_dataset_module import AgsDataModule
from lora.lora_modules import LoraLinear
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForSequenceClassification,
    OPTLoraForQuestionAnswering,
    OPTLoraDecoderLayer,
)
from pl_model_wrapper.base import PlWrapperBase

LORA_NAME_HASH = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "out_proj": 3,
    "fc1": 4,
    "fc2": 5,
}
ALPHA_UB = 10


class DynamicLoraReallocationCallback(pl.Callback):
    def __init__(
        self,
        N: int | float,
        data_module: pl.LightningDataModule,
        task: str,
        metric_reduction_tolerance: float,
        turn_on_percentile: float = 0.25,
        limit_test_batches: Optional[int | float] = None,
        save_path: str = None,
    ):
        """
        :param N: every N steps conduct alpha testing and reallocate lora ranks
        :param data_module: for loading batches for alpha testing
        :param task: task type for determining threshold comparison
        :param metric_reduction_tolerance: for computing the threshold for alpha testing
        :param turn_on_percentile: percentage of lora modules to be activated by the reallocation
        :param limit_test_batches: number of batches used in alpha testing
        :param save_path: file path for saving reallocation history
        """
        super().__init__()

        self.data_module = data_module
        assert task in ["classification", "summarization", "causal_language_modeling"]
        self.task = task

        if type(N) is int:
            self.N: int = N
        elif type(N) is float:
            assert 0.0 < N <= 1.0, "N should be 0.0 < N <= 1.0"
            self.N: int = round(len(self.get_alpha_testing_dataloader()) * N)
        else:
            raise TypeError("N should be int or float between 0.0 and 1.0")

        if limit_test_batches is None:
            # Single-shot per epoch
            self.limit_test_batches = round(len(self.get_alpha_testing_dataloader()) / N)
        elif type(limit_test_batches) is int:
            self.limit_test_batches = limit_test_batches
        elif type(limit_test_batches) is float:
            self.limit_test_batches = round(len(self.get_alpha_testing_dataloader()) * limit_test_batches)
        else:
            raise TypeError(
                "limit_test_batches should be None (assumed single-shot) or int or float between 0.0 and 1.0"
            )

        self.metric_reduction_tolerance = metric_reduction_tolerance
        self.turn_on_percentile = turn_on_percentile

        self.reallocation_history = []
        self.history_save_path = save_path

    def get_alpha_testing_dataloader(self):
        return self._get_mixed_dataloader()

    def _get_train_dataloader(self) -> DataLoader:
        return self.data_module.train_dataloader()

    def _get_mixed_dataloader(self) -> DataLoader:
        # 1:1 mixed training set & validation set
        assert type(self.data_module) is AgsDataModule
        self.data_module: AgsDataModule
        if self.data_module.training_dataset is None:
            raise RuntimeError("The training dataset is not available.")
        if self.data_module.validation_dataset is None:
            raise RuntimeError("The validation dataset is not available.")

        train_idx = torch.randperm(len(self.data_module.training_dataset))
        validation_idx = torch.randperm(len(self.data_module.val_dataloader()))
        interleave_idx = torch.stack([train_idx, validation_idx], dim=1).view(-1)

        data_collator = None
        if self.data_module.dataset_info.data_collator_cls is not None:
            data_collator = self.data_module.dataset_info.data_collator_cls(
                tokenizer=self.data_module.tokenizer
            )

        return DataLoader(
            torch.utils.data.Subset(
                torch.utils.data.ConcatDataset(
                    [self.data_module.training_dataset, self.data_module.validation_dataset]
                ),
                indices=interleave_idx,
            ),
            batch_size=self.data_module.batch_size,
            shuffle=False,
            num_workers=self.data_module.num_workers,
            collate_fn=data_collator,
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: PlWrapperBase,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.N > 0:
            return

        original_limit_test_batches = trainer.limit_test_batches
        trainer.limit_test_batches = self.limit_test_batches

        dataloader = self.get_alpha_testing_dataloader()

        original_val_metrics = trainer.test(
            pl_module, dataloaders=dataloader, verbose=False
        )[0]

        def get_metric_name():
            match self.task:
                case "classification":
                    return "test_acc_epoch"
                case "summarization":
                    return "test_rouge_epoch"
                case "causal_language_modeling":
                    return "test_perplexity_epoch"
                case _:
                    raise ValueError(f"Unsupported task: {self.task}")

        def get_metric_threshold():
            original_metric = original_val_metrics[get_metric_name()]
            match self.task:
                case "classification":
                    # Accuracy
                    return (
                        original_metric
                        - original_metric * self.metric_reduction_tolerance
                    )
                case "summarization":
                    # Rouge score
                    return (
                        original_metric
                        - original_metric * self.metric_reduction_tolerance
                    )
                case "causal_language_modeling":
                    # Perplexity
                    return (
                        original_metric
                        + original_metric * self.metric_reduction_tolerance
                    )
                case _:
                    raise ValueError(f"Unsupported task: {self.task}")

        def check_exceed_threshold(val_metrics_dict):
            val_metric = val_metrics_dict[get_metric_name()]
            threshold = get_metric_threshold()
            match self.task:
                case "classification":
                    return val_metric < threshold
                case "summarization":
                    return val_metric < threshold
                case "causal_language_modeling":
                    return val_metric > threshold
                case _:
                    raise ValueError(f"Unsupported task: {self.task}")

        # Result format: {layer_idx: {proj: alpha}}
        res_val: dict[int, dict[str, float]] = {}

        with torch.no_grad():
            model = pl_module.model
            assert (
                type(model) is OPTLoraForCausalLM
                or type(model) is OPTLoraForSequenceClassification
                or type(model) is OPTLoraForQuestionAnswering
            )
            model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

            # Get alpha importance for each module
            for decoder_layer in reversed(model.model.decoder.layers):
                decoder_layer: OPTLoraDecoderLayer
                layer_id = decoder_layer.layer_id
                lora_modules: dict[str, LoraLinear] = {
                    "q_proj": decoder_layer.self_attn.q_proj,
                    "k_proj": decoder_layer.self_attn.k_proj,
                    "v_proj": decoder_layer.self_attn.v_proj,
                    "out_proj": decoder_layer.self_attn.out_proj,
                    "fc1": decoder_layer.fc1,
                    "fc2": decoder_layer.fc2,
                }

                for proj_name, lora in lora_modules.items():
                    if (
                        lora.active_adapter not in lora.lora_A.keys()
                        or lora.r[lora.active_adapter] == 0
                    ):
                        continue

                    # TODO: update printing
                    print(
                        f">>> Alpha testing layer {layer_id} projection {proj_name}",
                        end="\r",
                    )

                    lb, rb = (0, ALPHA_UB)
                    while lb < rb:
                        alpha = (lb + rb) // 2
                        if alpha == 0:
                            break
                        lora.importance_alpha = alpha / ALPHA_UB
                        val_metrics = trainer.test(
                            pl_module, dataloaders=dataloader, verbose=False
                        )[0]
                        if check_exceed_threshold(val_metrics):
                            lb = alpha + 1
                        else:
                            rb = alpha
                    alpha_res = rb

                    lora.importance_alpha = 1.0
                    if layer_id not in res_val:
                        res_val[layer_id] = {}
                    res_val[layer_id][proj_name] = alpha_res

            # Decide which modules to keep
            alpha_list = np.concatenate(
                [
                    [
                        (layer_id, LORA_NAME_HASH[proj_name], v)
                        for proj_name, v in d.items()
                    ]
                    for layer_id, d in res_val.items()
                ],
                axis=0,
            )
            alpha_list = alpha_list[alpha_list[:, 0].argsort(kind="stable")]
            original_lora_module_num = len(alpha_list)
            budget = math.floor(self.turn_on_percentile * original_lora_module_num)
            # prioritise later layers
            idx = alpha_list[:, 2].argsort(kind="stable")[-budget:]
            turn_on = alpha_list[idx, :2].tolist()

            self.reallocation_history.append(turn_on)

            # Turn on/off lora modules
            for decoder_layer in reversed(model.model.decoder.layers):
                decoder_layer: OPTLoraDecoderLayer
                layer_id = decoder_layer.layer_id
                lora_modules: dict[str, LoraLinear] = {
                    "q_proj": decoder_layer.self_attn.q_proj,
                    "k_proj": decoder_layer.self_attn.k_proj,
                    "v_proj": decoder_layer.self_attn.v_proj,
                    "out_proj": decoder_layer.self_attn.out_proj,
                    "fc1": decoder_layer.fc1,
                    "fc2": decoder_layer.fc2,
                }

                for proj_name, lora in lora_modules.items():
                    if (
                        lora.active_adapter not in lora.lora_A.keys()
                        or lora.r[lora.active_adapter] == 0
                    ):
                        continue
                    proj_hash = LORA_NAME_HASH[proj_name]
                    lora.disable_adapters = [layer_id, proj_hash] in turn_on

            self.save_reallocation_history()
        trainer.limit_test_batches = original_limit_test_batches

    def save_reallocation_history(self):
        # Calculate frequency each lora module has been turned on
        turned_on_freq: dict[str, int | dict[str, int]] = {
            "total_reallocation_number": len(self.reallocation_history)
        }
        for reallocation in self.reallocation_history:
            for lora_module in reallocation:
                layer_id, proj_hash = lora_module
                proj_name = list(LORA_NAME_HASH.keys())[proj_hash]
                if f"layer_{layer_id}" not in turned_on_freq:
                    turned_on_freq[f"layer_{layer_id}"] = {}
                if proj_name in turned_on_freq[f"layer_{layer_id}"]:
                    turned_on_freq[f"layer_{layer_id}"][proj_name] = 0
                else:
                    turned_on_freq[f"layer_{layer_id}"][proj_name] += 1

        with open(self.history_save_path, "w+") as fout:
            toml.dump(turned_on_freq, fout)