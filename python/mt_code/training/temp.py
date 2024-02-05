def transform_data(documents: List[List], config):
    dataset = pd.DataFrame(columns=["tokens", "ner_tags"])
    dataset_index = 0

    flattened_dict = flatten_document_data(documents)
    words_per_sentence = config.data.words_per_example
    word_index = 0
    while word_index < len(flattened_dict["words"]):
        if word_index + words_per_sentence < len(flattened_dict["words"]):
            tokens = flattened_dict["words"][word_index:word_index + words_per_sentence]
            tags = flattened_dict["labels"][word_index:word_index + words_per_sentence]
        else:
            tokens = flattened_dict["words"][word_index:]
            tags = flattened_dict["labels"][word_index:]
        dataset.loc[dataset_index] = [tokens, tags]
        dataset_index += 1
        word_index += words_per_sentence
    return Dataset.from_pandas(dataset)



def create_tf_datasets(config: Config, train_raw_dataset, validation_raw_dataset, tokenizer):
    num_replicas = config.training.strategy.num_replicas_in_sync

    train_dataset = preprocess(config, train_raw_dataset, tokenizer, config.data.label2id)
    total_train_batch_size = config.training.per_device_train_batch_size * num_replicas
    train_batches_per_epoch = len(train_dataset) // total_train_batch_size

    tf_train_dataset = dataset_to_tf(
        train_dataset,
        tokenizer,
        total_batch_size=total_train_batch_size,
        num_epochs=config.training.num_train_epochs,
        shuffle=True,
    )
    validation_dataset = preprocess(config, validation_raw_dataset, tokenizer, config.data.label2id)
    total_eval_batch_size = config.training.per_device_eval_batch_size * num_replicas
    validation_batches_per_epoch = len(validation_raw_dataset) // total_eval_batch_size

    tf_validation_dataset = dataset_to_tf(
        validation_dataset,
        tokenizer,
        total_batch_size=total_eval_batch_size,
        num_epochs=config.training.num_train_epochs,
        shuffle=False,
    )
    return (
        train_batches_per_epoch,
        tf_train_dataset,
        validation_batches_per_epoch,
        tf_validation_dataset,
    )



    with config.training.strategy.scope():
        model = initialize_model(config, auto_config, tokenizer)

        (
            train_batches_per_epoch,
            tf_train_dataset,
            validation_batches_per_epoch,
            tf_validation_dataset,
        ) = create_tf_datasets(config, train_dataset, validation_dataset, tokenizer)

        optimizer = initialize_optimizer(config, train_batches_per_epoch)
        model.compile(optimizer=optimizer)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {config.training.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {config.training.per_device_train_batch_size}")
        model.fit(
            tf_train_dataset,
            validation_data=tf_validation_dataset,
            epochs=int(config.training.num_train_epochs),
            steps_per_epoch=train_batches_per_epoch,
            validation_steps=validation_batches_per_epoch,
        )
    tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))

    model.save_pretrained(output_path)
    model.save(output_path)









