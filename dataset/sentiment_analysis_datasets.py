import torch
from torch.utils.data import Dataset
import datasets

from dataset import add_dataset_info


class SentimentAnalysisDatasetBase(Dataset):
    info = None  # MaseDatasetInfo

    # The mapping to update tokenizer's special token mapping
    # Some dataset contains special tokens like <unk> in the text
    # Keys should be in the list of predefined special attributes: [`bos_token`, `eos_token`, `unk_token`,
    # `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`].
    special_token_mapping: dict[str, str] = None

    sent_col_name: str = None
    label_col_name: str = None

    def __init__(
        self,
        split_name: str,
        tokenizer,
        max_token_len: int,
        num_workers: int,
        load_from_cache_file: bool = True,
        load_from_saved_path: str = None,
        auto_setup: bool = True,
    ):
        super().__init__()
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        self.load_from_saved_path = load_from_saved_path
        self.data_ = None

        if self.special_token_mapping is not None:
            self.tokenizer.add_special_tokens(self.special_token_mapping)

        if auto_setup:
            self.prepare_data()
            self.setup()

    def _download_dataset(self) -> datasets.DatasetDict:
        raise NotImplementedError

    def prepare_data(self):
        self._download_dataset()

    def setup(self):
        self.data_ = self._download_dataset()[self.split_name]

    def __len__(self):
        if self.data_ is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        return len(self.data_)

    def __getitem__(self, index):
        if self.data_ is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        data_row = self.data_[index]
        main_text = data_row[self.sent_col_name]
        labels = data_row[self.label_col_name]
        encoding = self.tokenizer(
            main_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        return dict(
            sentence=main_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=torch.tensor([labels]),
        )


@add_dataset_info(
    name="sst2",
    dataset_source="hf_datasets",
    available_splits=("train", "validation", "pred"),
    sequence_classification=True,
    num_classes=2,
)
class SentimentalAnalysisDatasetSST2(SentimentAnalysisDatasetBase):
    sent_col_name = "sentence"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        if self.load_from_cache_file and self.load_from_saved_path is not None:
            dataset_dict = datasets.load_dataset(
                "glue", "sst2", cache_dir=self.load_from_saved_path
            )
        else:
            dataset_dict = datasets.load_dataset("glue", "sst2")
        return dataset_dict


@add_dataset_info(
    name="cola",
    dataset_source="hf_datasets",
    available_splits=("train", "validation", "pred"),
    sequence_classification=True,
    num_classes=2,
)
class SentimentalAnalysisDatasetCoLa(SentimentAnalysisDatasetBase):
    """
    Accepatablity task
    """

    sent_col_name = "sentence"
    label_col_name = "label"

    def _download_dataset(self) -> datasets.DatasetDict:
        if self.load_from_cache_file and self.load_from_saved_path is not None:
            dataset_dict = datasets.load_dataset(
                "glue", "cola", cache_dir=self.load_from_saved_path
            )
        else:
            dataset_dict = datasets.load_dataset("glue", "cola")
        return dataset_dict
