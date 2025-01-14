import datasets
import pytorch_lightning as pl
from transformers import PreTrainedModel, PreTrainedTokenizer

from dataset import get_dataset_info, AgsDatasetInfo
from dataset.pl_dataset_module import AgsDataModule
from loading.config_load import load_config
from loading.tokenizer_loader import get_tokenizer
from models.model_info import get_model_info, AgsModelInfo
from loading.model_loader import get_model


def setup_model_and_dataset(
    args,
) -> tuple[
    PreTrainedModel,
    AgsModelInfo,
    PreTrainedTokenizer,
    pl.LightningDataModule,
    AgsDatasetInfo,
]:
    dataset_info = get_dataset_info(args.dataset)

    checkpoint = None
    if args.load_name is not None:  # and args.load_type == "hf":
        checkpoint = args.load_name

    tokenizer = get_tokenizer(
        args.model,
        args.backbone_model if args.backbone_model is not None else checkpoint,
    )

    lora_config = None
    if args.lora_config is not None:
        lora_config = load_config(args.lora_config)
    # pprint(lora_config)

    shortcut_config = None
    if args.shortcut_config is not None:
        shortcut_config = load_config(args.shortcut_config)
    # pprint(shortcut_config)

    data_module = AgsDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        max_token_len=args.max_token_len,
        num_workers=args.num_workers,
        load_from_cache_file=not args.disable_dataset_cache,
        load_from_saved_path=args.dataset_saved_path,
    )

    model_info = get_model_info(args.model)

    model = get_model(
        name=args.model,
        task=args.task,
        dataset_info=dataset_info,
        pretrained=args.is_pretrained,
        checkpoint=args.backbone_model
        if args.backbone_model is not None
        else checkpoint,
        lora_config=lora_config,
        shortcut_config=shortcut_config,
    )

    return model, model_info, tokenizer, data_module, dataset_info
