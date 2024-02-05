import logging
from pathlib import Path
import os
from mt_code.config import Config
from mt_code.data.processing import process_dataset, dataset_to_tf
from mt_code.training.initializers import initialize_optimizer, initialize_model, initialize_tokenizer

logger = logging.getLogger(__name__)


def train(output_path: Path, config: Config):
    train_dataset, validation_dataset = process_dataset(config)
    train_batches_per_epoch = len(train_dataset) // config.training.train_batch_size_per_device
    validation_batches_per_epoch = len(validation_dataset) // config.training.validation_batch_size_per_device
    auto_config, tokenizer = initialize_tokenizer(config)

    with config.training.tf_strategy.scope():
        tf_train_dataset = dataset_to_tf(train_dataset, tokenizer, config)
        tf_validation_dataset = dataset_to_tf(train_dataset, config)
        optimizer = initialize_optimizer(config, train_batches_per_epoch)
        model = initialize_model(config, auto_config, tokenizer)
        model.compile(optimizer=optimizer)
        model.fit(
            tf_train_dataset,
            validation_data=tf_validation_dataset,
            epochs=int(config.training.epochs),
            steps_per_epoch=train_batches_per_epoch,
            validation_steps=validation_batches_per_epoch,
        )

    tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))
    model.save_pretrained(output_path)
    model.save(output_path)

