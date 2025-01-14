import logging
import math
import types

import toml
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AgsDatasetInfo
from dataset.pl_dataset_module import AgsDataModule
from lora.lora_modules import LoraLinear
from models.model_info import AgsModelInfo
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForQuestionAnswering,
    OPTLoraForSequenceClassification,
    OPTLoraDecoderLayer,
)
from pl_model_wrapper.base import PlWrapperBase

logger = logging.getLogger(__name__)


def snip_test(
    pl_model: PlWrapperBase,
    model_info: AgsModelInfo,  # dataclass of model's task type and name
    data_module: AgsDataModule,  # for preparing and loading datasets for pl trainer
    dataset_info: AgsDatasetInfo,  # dataclass including e.g. number of classes for the pl model wrapper
    task,  # to decide the pl model wrapper of which type should be used
    optimizer,  # optimizer for pl trainer
    learning_rate,  # lr for optimizer. lr_scheduler is default as CosineAnnealingLR
    weight_decay,  # weight_decay for optimizer
    lr_scheduler,  # for building lr scheduler
    eta_min,  # for building lr scheduler
    pl_trainer_args,  # args for pl trainer; include e.g. "max_epochs" for setting up lr_scheduler
    auto_requeue,  # for setting up SLURMEnvironment, environment for distributed launch
    save_path,  # path for saving checkpoints
    save_time,  # for result toml filename
    load_name,  # path to the saved checkpoint
    load_type,  # model checkpoint's type: ['pt', 'pl']
    resume_training,  # whether resume zero-proxy trained model from the checkpoint
    limit_test_batches,  # number of test batches limit for the zero-proxy test
    **kwargs,
):
    logger.warning("Running SNIP test")

    model = pl_model.model

    def get_unshuffled_train_dataloader(datamodule: AgsDataModule):
        if datamodule.training_dataset is None:
            raise RuntimeError("The training dataset is not available.")
        data_collator = None
        if datamodule.dataset_info.data_collator_cls is not None:
            data_collator = datamodule.dataset_info.data_collator_cls(
                tokenizer=datamodule.tokenizer
            )
        return DataLoader(
            datamodule.training_dataset,
            batch_size=datamodule.batch_size,
            shuffle=False,
            num_workers=datamodule.num_workers,
            collate_fn=data_collator,
        )

    data_module.prepare_data()
    data_module.setup()
    dataloader = get_unshuffled_train_dataloader(data_module)

    # SNIP
    assert (
        type(model) is OPTLoraForCausalLM
        or type(model) is OPTLoraForSequenceClassification
        or type(model) is OPTLoraForQuestionAnswering
    )
    model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

    for name, param in model.named_parameters():
        param.requires_grad = False

    # add weight masks
    def lora_forward(self, x):
        if self.active_adapter not in self.lora_A.keys():
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        elif self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
        else:
            self.unmerge()
            res = F.linear(
                x, self.weight if not self.fan_in_fan_out else self.weight.T, self.bias
            )
            res = (
                res
                + (
                    F.linear(
                        F.linear(
                            self.lora_dropout[self.active_adapter](x),
                            self.lora_A[self.active_adapter].weight
                            * self.weight_mask_A,
                        ),
                        self.lora_B[self.active_adapter].weight * self.weight_mask_B,
                    )
                )
                * self.scaling[self.active_adapter]
            )
        # res = res.to(input_dtype)
        return res

    for decoder_layer in reversed(model.model.decoder.layers):
        decoder_layer: OPTLoraDecoderLayer
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
                or lora.disable_adapters
                or lora.r[lora.active_adapter] == 0
            ):
                continue

            lora_A: nn.Linear = lora.lora_A[lora.active_adapter]
            lora_A.weight.requires_grad = False
            lora.weight_mask_A = nn.Parameter(torch.ones_like(lora_A.weight))
            lora_B: nn.Linear = lora.lora_B[lora.active_adapter]
            lora_B.weight.requires_grad = False
            lora.weight_mask_B = nn.Parameter(torch.ones_like(lora_B.weight))

            lora.forward = types.MethodType(lora_forward, lora)

    # compute gradients
    if type(limit_test_batches) is float:
        limit_batch_num = math.ceil(len(dataloader) * limit_test_batches)
        if limit_batch_num != len(dataloader) * limit_test_batches:
            logger.warning(
                "More data batches than the provided test ratio limit are used"
            )
    else:
        limit_batch_num = limit_test_batches
    pl_model.to("cuda")
    pl_model.zero_grad()
    msg = ""
    for i, batch in enumerate(dataloader):
        if i >= limit_batch_num:
            break
        print(" " * len(msg), end="\r")
        msg = f">>> Testing on training batch {i+1} / {limit_batch_num}"
        print(msg, end="\r")
        batch = data_module.transfer_batch_to_device(batch, torch.device("cuda"), 0)
        loss = pl_model.training_step(batch=batch, batch_idx=i)
        loss.backward()
    print()

    # calculate score of every lora module
    grads_abs = {
        "limit_test_num": limit_batch_num * dataloader.batch_size,
    }
    for decoder_layer in model.model.decoder.layers:
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
                or lora.disable_adapters
                or lora.r[lora.active_adapter] == 0
            ):
                continue

            grad_lora = (
                torch.sum(torch.abs(lora.weight_mask_A.grad))
                + torch.sum(torch.abs(lora.weight_mask_B.grad))
            ).item()

            if f"layer_{layer_id}" not in grads_abs:
                grads_abs[f"layer_{layer_id}"] = {}
            grads_abs[f"layer_{layer_id}"][proj_name] = grad_lora

    log_path = f"{save_path}/snip_{save_time}.toml"
    with open(log_path, "w+") as fout:
        toml.dump(grads_abs, fout)
    logger.info("Result saved as toml")

    logger.warning("SNIP test done")
