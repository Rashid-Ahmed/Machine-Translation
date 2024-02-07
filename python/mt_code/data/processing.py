from mt_code.config import Config
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq


def load_data(config: Config):
    dataset = load_dataset(config.data.dataset_name, "de-en")
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    return train_dataset, validation_dataset


def tokenize_data(examples, tokenizer, config: Config):
    de_examples = []
    en_examples = []
    prefix = "translate German to English: "
    for ex in examples:
        de_examples.append(prefix + ex['de'])
        en_examples.append(ex['en'])

    tokenized_examples = tokenizer(de_examples, text_target=en_examples, max_length=config.data.max_token_length,
                                   truncation=True,
                                   padding=config.data.pad_to_max_length)
    return tokenized_examples


def dataset_to_tf(train_dataset, validation_dataset, tokenizer, config, model):
    tokenized_train = train_dataset.map(lambda batch: tokenize_data(batch['translation'], tokenizer, config), batched=True,
                                        batch_size=config.training.batch_size_per_device)
    tokenized_validation = validation_dataset.map(lambda batch: tokenize_data(batch['translation'], tokenizer, config),
                                                  batched=True,
                                                  batch_size=config.training.batch_size_per_device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=config.model.model_name, return_tensors="tf")

    tf_train_set = model.prepare_tf_dataset(
        tokenized_train,
        shuffle=True,
        batch_size=config.training.batch_size_per_device,
        collate_fn=data_collator)

    tf_test_set = model.prepare_tf_dataset(
        tokenized_validation,
        shuffle=False,
        batch_size=config.training.batch_size_per_device,
        collate_fn=data_collator
    )

    return tf_train_set, tf_test_set
