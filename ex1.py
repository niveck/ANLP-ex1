import sys
import numpy as np
from time import time
from prepare import prepare
import evaluate
from datasets import load_dataset
from transformers import (
    set_seed, AutoModelForSequenceClassification, Trainer,
    EvalPrediction, AutoTokenizer, AutoConfig)
import wandb
import torch


PROJECT_NAME = "ANLP-ex1"
DATASET = "sst2"
MODEL_NAMES = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
RESULTS_PATH = "res.txt"
PREDICTIONS_OUTPUT_PATH = "predictions.txt"


def create_metric(metric):
    def compute_metrics(p: EvalPrediction):
        # inspired from examples/pytorch/text-classification/run_glue.py on GitHub
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.argmax(predictions, axis=1)
        result = metric.compute(predictions=predictions, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    return compute_metrics


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
    compute_metrics = create_metric(metric)
    res = ""
    best_mean_accuracy = 0
    most_accurate_model = None
    most_accurate_model_tokenizer = None
    accumulated_training_time = 0
    for model_name in model_names:
        mean_accuracy, accuracy_std, model_best_trainer, tokenizer, training_time = \
            finetune_sentiment_analysis_model(dataset, model_name, number_of_seeds,
                                              number_of_training_samples,
                                              number_of_validation_samples, compute_metrics)
        res += f"{model_name},{mean_accuracy} +- {accuracy_std}\n"
        if mean_accuracy >= best_mean_accuracy:  # >= is used in edge case of all accuracies are 0
            best_mean_accuracy = mean_accuracy
            most_accurate_model = model_best_trainer
            most_accurate_model_tokenizer = tokenizer
        accumulated_training_time += training_time
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
    preprocess = lambda examples: tokenizer(examples["sentence"], truncation=True, padding=True)
    trainers = []
    accuracies = []
    training_time = 0
    for seed in range(number_of_seeds):
        set_seed(seed)
        if seed == 0:
            wandb.init(project=PROJECT_NAME, name=model_name.replace("/", "-"))
        preprocessed_data = dataset.map(preprocess, batched=True)
        train_dataset = preprocessed_data["train"]
        if number_of_training_samples > 0:
            train_dataset = train_dataset.select(range(number_of_training_samples))
        eval_dataset = preprocessed_data["validation"]
        if number_of_validation_samples > 0:
            eval_dataset = eval_dataset.select(range(number_of_validation_samples))
        trainer = Trainer(model=model, train_dataset=train_dataset,
                          eval_dataset=eval_dataset, compute_metrics=compute_metrics,
                          tokenizer=tokenizer)
        train_result = trainer.train()
        eval_results = trainer.evaluate()
        if seed == 0:
            loss_history = [log["loss"] for log in trainer.state.log_history]
            for step, loss in enumerate(loss_history):
                wandb.log({"train_loss": loss}, step=step)
            wandb.finish()
        accuracy = eval_results["eval_accuracy"]
        trainers.append(trainer)
        accuracies.append(accuracy)
        training_time += train_result.metrics["train_runtime"]
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
    trainer.model.eval()
    test_dataset = dataset["test"]
    if number_of_prediction_samples > 0:
        test_dataset = test_dataset.select(range(number_of_prediction_samples))
    output = ""
    prediction_time = 0
    device = torch.device("cuda:0")
    for sentence in test_dataset["sentence"]:
        tokenized_sentence = tokenizer(sentence, truncation=True, return_tensors='pt')
        tokenized_sentence = {key: value.to(device) for key, value in tokenized_sentence.items()}
        before_predict_time = time()
        prediction = trainer.model(**tokenized_sentence).logits.argmax(dim=1).item()
        prediction_time += (time() - before_predict_time)
        output += f"{sentence}###{prediction}\n"
    with open(predictions_output_path, "w") as f:
        f.write(output)
    return prediction_time


def main():
    if len(sys.argv) != 5:
        raise ValueError("Wrong number of arguments, for usage see README.md")
    number_of_seeds = int(sys.argv[1])
    number_of_training_samples = int(sys.argv[2])
    number_of_validation_samples = int(sys.argv[3])
    number_of_prediction_samples = int(sys.argv[4])

    prepare()

    dataset = load_dataset(DATASET)

    wandb.login()

    accumulated_training_time, most_accurate_model, tokenizer, res = train(
        dataset, MODEL_NAMES, number_of_seeds, number_of_training_samples,
        number_of_validation_samples)

    prediction_time = predict(dataset, most_accurate_model, tokenizer, number_of_prediction_samples)

    res += f"train time,{accumulated_training_time}\npredict time,{prediction_time}\n"
    with open(RESULTS_PATH, "w") as f:
        f.write(res)


if __name__ == "__main__":
    main()
