from dataclasses import dataclass
from enum import Enum

from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    OPTConfig,
    GPT2Tokenizer,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    OPTForSequenceClassification,
    OPTForCausalLM,
    GemmaConfig,
    GemmaTokenizer,
)

from models.configuration_gemma_lora import GemmaLoraConfig
from models.configuration_llama_lora import LlamaLoraConfig
from models.configuration_opt_lora import OPTLoraConfig
from models.configuration_opt_lora_ags import (
    OPTLoraAgsConfig,
)
from models.configuration_roberta_lora import RobertaLoraConfig
from models.configuration_roberta_lora_ags import RobertaLoraAgsConfig
from models.modeling_gemma_lora import (
    GemmaLoraForSequenceClassification,
    GemmaLoraForCausalLM,
)
from models.modeling_opt_lora import (
    OPTLoraForSequenceClassification,
    OPTLoraForCausalLM,
)
from models.modeling_opt_lora_ags import (
    OPTLoraAgsForSequenceClassification,
    OPTLoraAgsForCausalLM,
)
from models.modeling_roberta_lora import (
    RobertaLoraForSequenceClassification,
    RobertaLoraForCausalLM,
)
from models.modeling_roberta_lora_ags import RobertaLoraForSequenceClassification as RobertaLoraAgsForSequenceClassification
from models.modeling_roberta_lora_ags import RobertaLoraForCausalLM as RobertaLoraAgsForCausalLM
from models.modeling_llama_lora import (
    LlamaForSequenceClassification,
    LlamaForCausalLM,
)
# from models.modeling_llama import (
#     LlamaForSequenceClassification,
#     LlamaForCausalLM,
# )


class ModelSource(Enum):
    """
    The source of the model, must be one of the following:
    - HF: HuggingFace
    - MANUAL: manually implemented
    """

    HF_TRANSFORMERS = "hf_transformers"
    MANUAL = "manual"


@dataclass
class AgsModelInfo:
    name: str
    model_source: ModelSource
    task_type: str = "nlp"
    # NLP models
    sequence_classification: bool = False
    seq2seqLM: bool = False
    causal_LM: bool = False
    # Manual models
    is_lora: bool = False
    is_ags: bool = False

    def __post_init__(self):
        self.model_source = (
            ModelSource(self.model_source)
            if isinstance(self.model_source, str)
            else self.model_source
        )
        assert self.sequence_classification or self.seq2seqLM or self.causal_LM
        if self.is_lora:
            assert (
                self.model_source == ModelSource.MANUAL
            ), "LoRA model must be a manual model."
        if self.is_ags:
            assert (
                self.model_source == ModelSource.MANUAL
            ), "AGS shortcut model must be a manual model."


HF_NLP_MODELS = {
    "roberta-base": {
        "config_cls": RobertaConfig,
        "tokenizer_cls": RobertaTokenizer,
        "info": AgsModelInfo(
            "roberta-base",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
        ),
    },
    "roberta-large": {
        "config_cls": RobertaConfig,
        "tokenizer_cls": RobertaTokenizer,
        "info": AgsModelInfo(
            "roberta-large",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
        ),
    },
    "facebook/opt-125m": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-125m",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-350m": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-350m",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-1.3b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-1.3b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-2.7b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-2.7b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-6.7b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-6.7b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-13b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-13b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-30b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-30b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "facebook/opt-66b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "facebook/opt-66b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "google/gemma-2b": {
        "config_cls": GemmaConfig,
        "tokenizer_cls": GemmaTokenizer,
        "info": AgsModelInfo(
            "google/gemma-2b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "google/gemma-7b": {
        "config_cls": GemmaConfig,
        "tokenizer_cls": GemmaTokenizer,
        "info": AgsModelInfo(
            "google/gemma-7b",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "google/gemma-2b-it": {
        "config_cls": GemmaConfig,
        "tokenizer_cls": GemmaTokenizer,
        "info": AgsModelInfo(
            "google/gemma-2b-it",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
    "google/gemma-7b-it": {
        "config_cls": GemmaConfig,
        "tokenizer_cls": GemmaTokenizer,
        "info": AgsModelInfo(
            "google/gemma-7b-it",
            model_source="hf_transformers",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
    },
}

MANUAL_MODELS = {
    "Llama2": {
        "config_cls": LlamaConfig,
        "tokenizer_cls": LlamaTokenizer,
        "info": AgsModelInfo(
            "llama_plain",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
        "sequence_classification": LlamaForSequenceClassification,
        "causal_LM": LlamaForCausalLM,
    },
    "Llama2_lora": {
        "config_cls": LlamaLoraConfig,
        "tokenizer_cls": LlamaTokenizer,
        "info": AgsModelInfo(
            "llama2_lora",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
            is_lora=True,
        ),
        "sequence_classification": LlamaForSequenceClassification,
        "causal_LM": LlamaForCausalLM,
    },
    # "Llama2_lora_ags": {
    #     "config_cls": LlamaLoraAgsConfig,
    #     "tokenizer_cls": LlamaTokenizer,
    #     "info": AgsModelInfo(
    #         "llama2_lora_ags",
    #         model_source="manual",
    #         task_type="nlp",
    #         sequence_classification=True,
    #         causal_LM=True,
    #         is_lora=True,
    #         is_ags=True,
    #     ),
    #     "sequence_classification": LlamaLoraAgsForSequenceClassification,
    #     "causal_LM": LlamaLoraAgsForCausalLM,
    # },
    "llama_plain": {
        "config_cls": LlamaConfig,
        "tokenizer_cls": LlamaTokenizer,
        "info": AgsModelInfo(
            "llama_plain",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
        "sequence_classification": LlamaForSequenceClassification,
        "causal_LM": LlamaForCausalLM,
    },
    "opt_plain": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "opt_plain",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
        ),
        "sequence_classification": OPTForSequenceClassification,
        "causal_LM": OPTForCausalLM,
    },
    "opt_lora": {
        "config_cls": OPTLoraConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "opt_lora",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
            is_lora=True,
        ),
        "sequence_classification": OPTLoraForSequenceClassification,
        "causal_LM": OPTLoraForCausalLM,
    },
    "roberta_lora": {
        "config_cls": RobertaLoraConfig,
        "tokenizer_cls": RobertaTokenizer,
        "info": AgsModelInfo(
            "roberta_lora",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
            is_lora=True,
        ),
        "sequence_classification": RobertaLoraForSequenceClassification,
        "causal_LM": RobertaLoraForCausalLM,
    },
    "roberta_lora_ags": {
    "config_cls": RobertaLoraAgsConfig,
    "tokenizer_cls": RobertaTokenizer,
    "info": AgsModelInfo(
        "roberta_lora_ags",
        model_source="manual",
        task_type="nlp",
        sequence_classification=True,
        causal_LM=True,
        is_lora=True,
        is_ags=True,
    ),
    "sequence_classification": RobertaLoraAgsForSequenceClassification,
    "causal_LM": RobertaLoraAgsForCausalLM,
    },
    "gemma_lora": {
        "config_cls": GemmaLoraConfig,
        "tokenizer_cls": GemmaTokenizer,
        "info": AgsModelInfo(
            "gemma_lora",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
            is_lora=True,
        ),
        "sequence_classification": GemmaLoraForSequenceClassification,
        "causal_LM": GemmaLoraForCausalLM,
    },
    "opt_lora_ags": {
        "config_cls": OPTLoraAgsConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": AgsModelInfo(
            "opt_lora_ags",
            model_source="manual",
            task_type="nlp",
            sequence_classification=True,
            causal_LM=True,
            is_lora=True,
            is_ags=True,
        ),
        "sequence_classification": OPTLoraAgsForSequenceClassification,
        "causal_LM": OPTLoraAgsForCausalLM,
    },
}


def get_model_info(name: str) -> AgsModelInfo:
    if name in MANUAL_MODELS:
        return MANUAL_MODELS[name]["info"]
    elif name in HF_NLP_MODELS:
        return HF_NLP_MODELS[name]["info"]
    else:
        raise ValueError(f"Model {name} not found.")
