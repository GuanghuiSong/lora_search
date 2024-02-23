import copy
import logging
import os
import pickle
import time

import toml
import torch
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from lora.lora_modules import LoraLinear, mark_only_lora_as_trainable, update_lora_importance_alpha_require_grad
from models.model_info import AgsModelInfo
from models.modeling_opt_lora import (
    OPTLoraForCausalLM,
    OPTLoraForQuestionAnswering,
    OPTLoraForSequenceClassification, OPTLoraDecoderLayer,
)
from projectors.shortcut_modules import mark_ags_as_trainable
from tools.checkpoint_load import load_model_chkpt
import pl_model_wrapper
from tools.trainable_param_printer import print_trainable_parameters

logger = logging.getLogger(__name__)


def alpha_importance_test(
    model: torch.nn.Module | torch.fx.GraphModule,
    tokenizer,
    model_info: AgsModelInfo,  # dataclass of model's task type and name
    data_module: pl.LightningDataModule,  # for preparing and loading datasets for pl trainer
    dataset_info,  # dataclass including e.g. number of classes for the pl model wrapper
    task,  # to decide the pl model wrapper of which type should be used
    optimizer,  # optimizer for pl trainer
    learning_rate,  # lr for optimizer. lr_scheduler is default as CosineAnnealingLR
    weight_decay,  # weight_decay for optimizer
    lr_scheduler,  # for building lr scheduler
    eta_min,  # for building lr scheduler
    pl_trainer_args,  # args for pl trainer; include e.g. "max_epochs" for setting up lr_scheduler
    auto_requeue,  # for setting up SLURMEnvironment, environment for distributed launch
    save_path,  # path for saving checkpoints
    load_name,  # path to the saved checkpoint
    load_type,  # model checkpoint's type: ['pt', 'pl']
    resume_training,  # whether resume zero-proxy trained model from the checkpoint
):
    t = time.strftime("%H-%M")

    pl_validator_args = copy.deepcopy(pl_trainer_args)

    if resume_training:
        wrapper_pl_model: pl.LightningModule = pl_model_wrapper.get_model_wrapper(
            model_info, task
        )

        # load model from pl checkpoint
        if load_name is not None:
            model = load_model_chkpt(load_name, load_type=load_type, model=model)

        logger.warning(
            f"Restore model state from pl checkpoint {load_name}. Entered hyperparameter configuration ignored."
        )

        pl_model = wrapper_pl_model(
            model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,  # for building lr scheduler
            eta_min=eta_min,  # for building lr scheduler
            epochs=pl_trainer_args["max_epochs"],
            optimizer=optimizer,
        )
    else:
        pl_model = zero_proxy_train_lora(
            model,
            tokenizer,
            model_info,
            data_module,
            dataset_info,
            task,
            optimizer,
            learning_rate,
            weight_decay,
            lr_scheduler,
            eta_min,
            pl_trainer_args,
            auto_requeue,
            save_path,
            load_name,
            load_type,
        )

    # Run each test only on one minibatch
    trainer = pl.Trainer(**pl_validator_args, limit_val_batches=1)
    original_val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]

    def get_metric_name():
        match task:
            case "classification":
                return "val_acc_epoch"
            case "summarization":
                return "val_rouge_epoch"
            case "causal_language_modeling":
                return "val_perplexity_epoch"
            case _:
                raise ValueError(f"Unsupported task: {task}")

    def get_metric_threshold():
        original_metric = original_val_metrics[get_metric_name()]
        match task:
            case "classification":
                # Accuracy
                return original_metric - original_metric * 0.01
            case "summarization":
                # Rouge score
                return original_metric - original_metric * 0.01
            case "causal_language_modeling":
                # Perplexity
                return original_metric + original_metric * 0.01
            case _:
                raise ValueError(f"Unsupported task: {task}")

    def check_exceed_threshold(val_metrics_dict):
        val_metric = val_metrics_dict[get_metric_name()]
        threshold = get_metric_threshold()
        match task:
            case "classification":
                return val_metric <= threshold
            case "summarization":
                return val_metric <= threshold
            case "causal_language_modeling":
                return val_metric >= threshold
            case _:
                raise ValueError(f"Unsupported task: {task}")

    # Result format: {layer_idx: {proj: alpha}}
    res_val: dict[str, str | float | dict[str, float]] = {
        "task": task,
        "metric_name": get_metric_name(),
        "zero-proxy_metric": original_val_metrics[get_metric_name()],
    }

    # FOR RESUMING ALPHA TESTING
    resume_layer_id = -1
    resume_toml = ""

    with torch.no_grad():
        assert (
            type(model) is OPTLoraForCausalLM
            or type(model) is OPTLoraForSequenceClassification
            or type(model) is OPTLoraForQuestionAnswering
        )
        model: OPTLoraForCausalLM | OPTLoraForSequenceClassification | OPTLoraForQuestionAnswering

        # RESUME ALPHA TESTING
        if resume_layer_id > 0:
            with open(resume_toml, "r") as f:
                res_val = toml.load(f)
            logger.warning(f"Resuming alpha testing from layer {resume_layer_id} based on {resume_toml}")

        def save_toml(res: dict):
            log_path = f"{save_path}/alpha-importance_{t}.toml"
            with open(log_path, "w+") as fout:
                toml.dump(res, fout)
            logger.info("Result saved as toml")

        for decoder_layer in model.model.decoder.layers:
            decoder_layer: OPTLoraDecoderLayer
            layer_id = decoder_layer.layer_id

            # FOR RESUMING ALPHA TESTING
            if layer_id < resume_layer_id:
                continue

            lora_modules: dict[str, LoraLinear] = {
                "q_proj": decoder_layer.self_attn.q_proj,
                "k_proj": decoder_layer.self_attn.k_proj,
                "v_proj": decoder_layer.self_attn.v_proj,
                "out_proj": decoder_layer.self_attn.out_proj,
                "fc1": decoder_layer.fc1,
                "fc2": decoder_layer.fc2,
            }

            for proj_name, lora in lora_modules.items():
                adapter_name = lora.active_adapter
                alpha_res = 1.0
                logger.info(f">>> Testing layer {layer_id} projection {proj_name} <<<")

                alpha = 0.5
                lora.importance_alpha[adapter_name] = alpha
                val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]

                if check_exceed_threshold(val_metrics):
                    while alpha < 1.0:
                        alpha += 0.1
                        lora.importance_alpha[adapter_name] = alpha
                        val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
                        if not check_exceed_threshold(val_metrics):
                            alpha_res = alpha - 0.1
                            break
                else:
                    while alpha > 0.0:
                        alpha -= 0.1
                        lora.importance_alpha[adapter_name] = alpha
                        val_metrics = trainer.validate(pl_model, datamodule=data_module)[0]
                        if check_exceed_threshold(val_metrics):
                            alpha_res = alpha
                            break

                res_val[f"layer_{layer_id}"][proj_name] = alpha_res
                save_toml(res_val)

        save_toml(res_val)


def zero_proxy_train_lora(
    model: torch.nn.Module | torch.fx.GraphModule,
    tokenizer,
    model_info: AgsModelInfo,  # dataclass of model's task type and name
    data_module: pl.LightningDataModule,  # for preparing and loading datasets for pl trainer
    dataset_info,  # dataclass including e.g. number of classes for the pl model wrapper
    task,  # to decide the pl model wrapper of which type should be used
    optimizer,  # optimizer for pl trainer
    learning_rate,  # lr for optimizer. lr_scheduler is default as CosineAnnealingLR
    weight_decay,  # weight_decay for optimizer
    lr_scheduler,  # for building lr scheduler
    eta_min,  # for building lr scheduler
    pl_trainer_args,  # args for pl trainer; include e.g. "max_epochs" for setting up lr_scheduler
    auto_requeue,  # for setting up SLURMEnvironment, environment for distributed launch
    save_path,  # path for saving checkpoints
    load_name,  # path to the saved checkpoint
    load_type,  # model checkpoint's type: ['pt', 'pl']
):
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        latest_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path,
            filename="alpha-zero-proxy_last_chkpt",
            # save_last=True,
        )
        # Monitoring lr for the lr_scheduler
        lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        # TensorBoard logger
        tb_logger = TensorBoardLogger(save_dir=save_path, name="alpha-zero-proxy_logs")
        pl_trainer_args["callbacks"] = [
            latest_checkpoint_callback,
            lr_monitor_callback,
        ]
        pl_trainer_args["logger"] = [tb_logger]

    if auto_requeue is not None:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    pl_trainer_args["plugins"] = plugins

    wrapper_pl_model: pl.LightningModule = pl_model_wrapper.get_model_wrapper(
        model_info, task
    )

    # load model state from checkpoint
    if load_name is not None:
        model = load_model_chkpt(load_name, load_type=load_type, model=model)
        logger.warning(
            f"Restore model state from pl checkpoint {load_name}. Entered model hyperparameter configuration ignored."
        )

    pl_model: pl.LightningModule = wrapper_pl_model(
        model,
        dataset_info=dataset_info,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,  # for building lr scheduler
        eta_min=eta_min,  # for building lr scheduler
        epochs=pl_trainer_args["max_epochs"],
        optimizer=optimizer,
    )

    if model_info.is_lora:
        mark_only_lora_as_trainable(model, bias="none")
        update_lora_importance_alpha_require_grad(model, require_grad=False)

        if model_info.is_ags:
            mark_ags_as_trainable(model)
        print_trainable_parameters(model)

    # Zero-proxy training for LoRA modules
    trainer = pl.Trainer(**pl_trainer_args, limit_train_batches=0.01)

    trainer.fit(pl_model, datamodule=data_module)

    return pl_model