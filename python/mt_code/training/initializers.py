from mt_code.config import Config
from transformers import AutoTokenizer, AutoConfig, create_optimizer, TFAutoModelForSeq2SeqLM


def initialize_tokenizer(config: Config):
    auto_config = AutoConfig.from_pretrained(config.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name)

    return auto_config, tokenizer


def initialize_optimizer(config: Config, train_batches_per_epoch: int):

    optimizer, _ = create_optimizer(
        init_lr=config.training.lr,
        num_train_steps=int(config.training.epochs * train_batches_per_epoch),
        num_warmup_steps=config.training.warmup_steps,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        weight_decay_rate=config.training.weight_decay,
    )
    return optimizer


def initialize_model(
        config: Config,
        auto_config: AutoConfig.from_pretrained,
        tokenizer: AutoTokenizer.from_pretrained,

):
    model = TFAutoModelForSeq2SeqLM.from_pretrained(
        config.model.model_name,
        config=auto_config,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model
