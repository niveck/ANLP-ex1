import sys
import numpy as np
from prepare import prepare
import evaluate
from datasets import load_dataset
from transformers import (
    set_seed, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction,
    AutoTokenizer, AutoConfig)
import wandb


PROJECT_NAME = "ANLP-ex1"
LOG_NAME = "Accuracy Comparison"
DATASET = "sst2"
MODEL_NAMES = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
RESULTS_PATH = "res.txt"
PREDICTIONS_OUTPUT_PATH = "predictions.txt"


# def get_model_training_dir(model_name, seed): # todo maybe remove
#     """
#     :param model_name:
#     :param seed:
#     :return: the training dir of the saved model with specified name and seed
#     """
#     return "training_dir_" + model_name.replace("/", "-") + f"_{seed}"


def train(dataset, model_names, number_of_seeds, number_of_training_samples,
          number_of_validation_samples):
    """
    Performs fine-tuning of all combinations of pre-trained model names and number of seeds, to
    sentiment analysis on given dataset, assesses their accuracy on the validation set, and
    documents their mean and standard deviation of accuracy across all seeds
    :param dataset:
    :param model_names: list of model names that fit the names on the HuggingFace Hub
    :param number_of_seeds:
    :param number_of_training_samples:
    :param number_of_validation_samples:
    :return: accumulated training time of all models (in seconds), trainer object of model with the
             highest mean accuracy (for the seed with the highest accuracy with that model), its
             respective tokenizer and the string documenting the results.
    """
    metric = evaluate.load("accuracy")
    compute_metrics = lambda p: metric.compute(predictions=p.predictions, references=p.label_ids)
    res = ""
    best_mean_accuracy = 0  # todo validate that this is the worse possible
    most_accurate_model = None
    most_accurate_model_tokenizer = None
    accumulated_training_time = 0  # todo validate right format for seconds
    for model_name in model_names:
        print(f"#######   started finetune for {model_name}")  # todo remove
        mean_accuracy, accuracy_std, model_best_trainer, tokenizer, training_time = \
            finetune_sentiment_analysis_model(dataset, model_name, number_of_seeds,
                                              number_of_training_samples,
                                              number_of_validation_samples, compute_metrics)
        res += f"{model_name},{mean_accuracy} +- {accuracy_std}\n"
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            most_accurate_model = model_best_trainer
            most_accurate_model_tokenizer = tokenizer
        accumulated_training_time += training_time
    print(f"#######   finished finetune of all models")  # todo remove
    wandb.finish()
    print(f"#######   passed wandb.finish")  # todo remove
    res += "----\n"
    return accumulated_training_time, most_accurate_model, most_accurate_model_tokenizer, res


def finetune_sentiment_analysis_model(dataset, model_name, number_of_seeds,
                                      number_of_training_samples, number_of_validation_samples,
                                      compute_metrics):
    """
    Fine-tunes a sentiment analysis model on given dataset
    :param dataset:
    :param model_name: name of pre-trained model to use, as listed in HuggingFace Hub
    :param number_of_seeds: -1 means use all
    :param number_of_training_samples: -1 means use all
    :param number_of_validation_samples: -1 means use all
    :param compute_metrics: a callback to calculate metrics in evaluation of the model
    :return: mean accuracy of the model as predicted on the validation samples across all seeds,
             its standard deviation, the model train object with the highest accuracy, its tokenizer
            and the accumulated training time across all seeds in seconds.
    """
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    print(f"#######   passed config, tokenizer, model from pretrained")  # todo remove
    preprocess = lambda examples: tokenizer(examples["sentence"], truncation=True, padding=True)
    trainers = []
    accuracies = []
    training_time = 0
    for seed in range(number_of_seeds):
        print(f"#######   started finetune for seed {seed}")  # todo remove
        # args = TrainingArguments(get_model_training_dir(model_name, seed))  # todo maybe remove
        set_seed(seed)
        preprocessed_data = dataset.map(preprocess, batched=True)  # todo batched?
        train_dataset = preprocessed_data["train"]
        if number_of_training_samples > 0:
            train_dataset = train_dataset.select(range(number_of_training_samples))
        eval_dataset = preprocessed_data["validation"]
        if number_of_validation_samples > 0:
            eval_dataset = eval_dataset.select(range(number_of_validation_samples))
        # trainer = Trainer(model=model, args=args, train_dataset=train_dataset,  # todo maybe remove
        trainer = Trainer(model=model, train_dataset=train_dataset,
                          eval_dataset=eval_dataset, compute_metrics=compute_metrics,
                          tokenizer=tokenizer)
        # todo maybe use DataLoader and/or DataLoader
        print(f"#######   passed creating a trainer")  # todo remove
        train_result = trainer.train()
        print(f"#######   passed training")  # todo remove
        eval_results = trainer.evaluate()
        print(f"#######   passed evaluate")  # todo remove
        print(f"#######   eval_results.metrics: {eval_results.metrics}")  # todo remove
        accuracy = eval_results.metrics["accuracy"]  # todo validate key
        wandb.log({"Model": model_name, "Seed": seed, "Accuracy": accuracy})
        print(f"#######   passed wandb.log")  # todo remove
        trainers.append(trainer)
        accuracies.append(accuracy)
        training_time += train_result.metrics["train_runtime"]  # todo validate it's in seconds
    print(f"#######   passed all seeds for model {model_name}")  # todo remove
    return np.mean(accuracies), np.std(accuracies), trainers[np.argmax(accuracies)], tokenizer, \
        training_time


def predict(dataset, trainer, tokenizer, number_of_prediction_samples,
            predictions_output_path=PREDICTIONS_OUTPUT_PATH):
    """
    Uses the fine-tuned model trainer object to predict the sentiment of samples from given
    dataset's test set, and saves the results in the specified path
    :param dataset:
    :param trainer:
    :param tokenizer:
    :param number_of_prediction_samples: -1 means use all
    :param predictions_output_path:
    :return: prediction time in seconds
    """
    trainer.eval()
    # todo if have problems maybe use this line:
    # dataset.set_format("pt", output_all_columns=True)
    preprocess = lambda examples: tokenizer(examples["sentence"], truncation=True, padding=False)  # todo validate
    preprocessed_data = dataset.map(preprocess, batched=True)  # todo batched?
    test_dataset = preprocessed_data["test"]
    if number_of_prediction_samples > 0:
        test_dataset = test_dataset.select(range(number_of_prediction_samples))
    predictions = trainer.predict(test_dataset=test_dataset)
    print(f"#######   passed trainer.predict")  # todo remove
    output = ""
    for sentence, prediction in zip(test_dataset["sentence"], predictions.predictions):
        output += f"{sentence}###{prediction}\n"
    with open(predictions_output_path, "w") as f:
        f.write(output)
    print(f"#######   passed writing output")  # todo remove
    return predictions.metrics["predict_runtime"]


def main():
    if len(sys.argv) != 5:
        raise ValueError("Wrong number of arguments, for usage see README.md")
    number_of_seeds = int(sys.argv[1])
    number_of_training_samples = int(sys.argv[2])
    number_of_validation_samples = int(sys.argv[3])
    number_of_prediction_samples = int(sys.argv[4])

    prepare()
    print(f"#######   passed prepare script")  # todo remove

    dataset = load_dataset(DATASET)
    print(f"#######   passed loading dataset script")  # todo remove

    wandb.login()
    wandb.init(project=PROJECT_NAME, name=LOG_NAME)
    print(f"#######   passed wandb.init")  # todo remove

    accumulated_training_time, most_accurate_model, tokenizer, res = train(
        dataset, MODEL_NAMES, number_of_seeds, number_of_training_samples,
        number_of_validation_samples)
    print(f"#######   passed all train")  # todo remove

    prediction_time = predict(dataset, most_accurate_model, tokenizer, number_of_prediction_samples)
    print(f"#######   passed all predict")  # todo remove

    res += f"train time,{accumulated_training_time}\npredict time,{prediction_time}\n"
    with open(RESULTS_PATH, "w") as f:
        f.write(res)
    print(f"#######   passed writing of res")  # todo remove


if __name__ == "__main__":
    main()
