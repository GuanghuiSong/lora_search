from os import PathLike

import datasets
from datasets import DatasetInfo
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

from dataset import AgsDatasetInfo
from models.model_info import get_model_info, ModelSource, MANUAL_MODELS


def get_model(
    name: str,
    task: str,
    dataset_info: AgsDatasetInfo,
    pretrained: bool,
    checkpoint: str | PathLike = None,
    lora_config: dict = None,
    shortcut_config: dict = None,
) -> PreTrainedModel:
    model_info = get_model_info(name)
    model_kwargs = {
        "name": name,
        "task": task,
        "dataset_info": dataset_info,
        "pretrained": pretrained,
        "checkpoint": checkpoint,
    }
    if model_info.is_lora:
        model_kwargs["lora_config"] = lora_config

    if model_info.is_ags:
        model_kwargs["shortcut_config"] = shortcut_config

    match model_info.model_source:
        case ModelSource.HF_TRANSFORMERS:
            model = get_hf_model(**model_kwargs)
        case ModelSource.MANUAL:
            model = get_manual_model(**model_kwargs)
        case _:
            raise ValueError(f"Model source {model_info.model_source} not supported.")

    return model


def get_hf_model(
    name: str,
    task: str,
    dataset_info: AgsDatasetInfo,
    pretrained: bool,
    checkpoint: str | PathLike = None,
) -> PreTrainedModel:
    """
    Args:
        name: The name of the model.
        task: The task type.
        dataset_info: The dataset info.
        pretrained: Whether to load the pretrained model dict.
        checkpoint: The path to the checkpoint.

    ---
    A HuggingFace model checkpoint includes both config and model dict.
    - if `pretrained` is False, we will load the config from name/checkpoint and initialize the model randomly.
    - if `pretrained` is True, we will load the config and model dict from name/checkpoint.
    """

    model_info = get_model_info(name)

    match task:
        case "causal_language_modeling":
            if not model_info.causal_LM:
                raise ValueError(f"Task {task} is not supported for {name}")
            if pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
            else:
                config = AutoConfig.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
                model = AutoModelForCausalLM.from_config(config)
        case "classification":
            if not model_info.sequence_classification:
                raise ValueError(f"Task {task} is not supported for {name}")
            config = AutoConfig.from_pretrained(
                name if checkpoint is None else checkpoint,
                num_labels=dataset_info.num_classes,
            )
            if pretrained:
                model = AutoModelForSequenceClassification.from_pretrained(
                    name if checkpoint is None else checkpoint, config=config
                )
            else:
                model = AutoModelForSequenceClassification.from_config(config)
        case "summarization":
            if not model_info.seq2seqLM:
                raise ValueError(f"Task {task} is not supported for {name}")
            if pretrained:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
            else:
                config = AutoConfig.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
                model = AutoModelForSeq2SeqLM.from_config(config)
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model


def get_manual_model(
    name: str,
    task: str,
    dataset_info: AgsDatasetInfo,
    pretrained: bool,
    checkpoint: str | PathLike,
    lora_config: dict = None,
    shortcut_config: dict = None,
) -> PreTrainedModel:
    """
    Args:
        name: The name of the model.
        task: The task type.
        dataset_info: The dataset info.
        pretrained: Whether to load the model dict.
        checkpoint: The checkpoint path (For HuggingFace Models this means both config and model dict).
        lora_config: The LoRA config.
        shortcut_config: The AGS shortcut config.

    ---
    Arg `pretrained` and `checkpoint`:
    - if pretrained and checkpoint: load pretrained config and model dict
    - if (not pretrained) and checkpoint: load pretrained config only, e.g., num_hidden_layers, num_attention_heads, etc.
    - else: raise RuntimeError
    """
    model_info = get_model_info(name)

    match task:
        case "classification":
            assert (
                model_info.sequence_classification
            ), f"Task {task} is not supported for {name}"
            num_classes = dataset_info.num_classes
            if lora_config is not None and shortcut_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                    shortcut_config=shortcut_config,
                    num_labels=num_classes,
                )
            elif lora_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                    num_labels=num_classes,
                )
            elif shortcut_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    shortcut_config=shortcut_config,
                    num_labels=num_classes,
                )
            else:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    num_labels=num_classes,
                )
            model_cls = MANUAL_MODELS[name]["sequence_classification"]
        case "causal_language_modeling":
            assert model_info.causal_LM, f"Task {task} is not supported for {name}"
            if lora_config is not None and shortcut_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                    shortcut_config=shortcut_config,
                )
            elif lora_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                )
            elif shortcut_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    shortcut_config=shortcut_config,
                )
            else:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(checkpoint)
            model_cls = MANUAL_MODELS[name]["causal_LM"]
        case "summarization":
            assert model_info.seq2seqLM, f"Task {task} is not supported for {name}"
            if lora_config is not None and shortcut_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                    shortcut_config=shortcut_config,
                )
            elif lora_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    lora_config=lora_config,
                )
            elif shortcut_config is not None:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                    checkpoint,
                    shortcut_config=shortcut_config,
                )
            else:
                config = MANUAL_MODELS[name]["config_cls"].from_pretrained(checkpoint)
            model_cls = MANUAL_MODELS[name]["seq2seqLM"]
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")
    if pretrained:
        model = model_cls.from_pretrained(checkpoint, config=config)
    else:
        model = model_cls(config)

    return model
