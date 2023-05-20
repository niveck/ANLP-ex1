import sys
import numpy as np
from prepare import prepare
import evaluate
from datasets import load_dataset
from transformers import (
    set_seed, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction,
    AutoTokenizer, AutoConfig)
import wandb


DATASET = "sst2"
MODEL_NAMES = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
RESULTS_PATH = "res.txt"
PREDICTIONS_OUTPUT_PATH = "predictions.txt"


def get_model_training_dir(model_name, seed):
    """
    :param model_name:
    :param seed:
    :return: the training dir of the saved model with specified name and seed
    """
    return "training_dir_" + model_name.replace("/", "-") + f"_{seed}"


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
    :return: accumulated training time of all models (in seconds), the name of model with the
    highest mean accuracy, the seed with the highest accuracy with that model and the string
    documenting the results.
    """
    metric = evaluate.load("accuracy")
    compute_metrics = lambda p: metric.compute(predictions=p.predictions, references=p.label_ids)
    res = ""
    best_mean_accuracy = 0  # todo validate that this is the worse possible
    most_accurate_model = None
    best_seed = None
    accumulated_training_time = 0  # todo validate right format for seconds
    for model_name in model_names:
        mean_accuracy, accuracy_std, model_best_seed, training_time = \
            finetune_sentiment_analysis_model(dataset, model_name, number_of_seeds,
                                              number_of_training_samples,
                                              number_of_validation_samples, compute_metrics)
        res += f"{model_name},{mean_accuracy} +- {accuracy_std}\n"
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            most_accurate_model, best_seed = model_name, model_best_seed
        accumulated_training_time += training_time
    res += "----\n"
    return accumulated_training_time, most_accurate_model, best_seed, res


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
             its standard deviation, the seed used for the best accuracy and the accumulated
             training time across all seeds in seconds.
    """
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    preprocess = lambda examples: tokenizer(examples["sentence"], truncation=True, padding=True)
    accuracies = []
    training_time = 0
    for i in range(number_of_seeds):
        args = TrainingArguments(get_model_training_dir(model_name, i))
        set_seed(i)
        preprocessed_data = dataset.map(preprocess, batched=True)  # todo batched?
        train_dataset = preprocessed_data["train"]
        if number_of_training_samples > 0:
            train_dataset = train_dataset[:number_of_training_samples]
        eval_dataset = preprocessed_data["Validation"]
        if number_of_validation_samples > 0:
            eval_dataset = eval_dataset[:number_of_validation_samples]
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                          eval_dataset=eval_dataset, compute_metrics=compute_metrics,
                          tokenizer=tokenizer)
        train_result = trainer.train()
        accuracies.append(train_result.metrics["accuracy"])  # todo validate key
        training_time += train_result.metrics["train_runtime"]  # todo validate it's in seconds
    return np.mean(accuracies), np.std(accuracies), np.argmax(accuracies), training_time


def predict(dataset, model, seed, number_of_prediction_samples,
            predictions_output_path=PREDICTIONS_OUTPUT_PATH):
    """
    Uses the fine-tuned model with the specified name and seed to predict the sentiment of samples
    from given dataset's test set, and saves the results in the specified path
    :param dataset:
    :param model:
    :param seed:
    :param number_of_prediction_samples: -1 means use all
    :param predictions_output_path:
    :return: prediction time in seconds
    """
    # todo right after using map for the tokenizer, run line from Daria L in WA
    # todo Remember to run the model.eval() command before prediction
    return None


def main():
    if len(sys.argv) != 5:
        raise ValueError("Wrong number of arguments, for usage see README.md")
    number_of_seeds = int(sys.argv[1])
    number_of_training_samples = int(sys.argv[2])
    number_of_validation_samples = int(sys.argv[3])
    number_of_prediction_samples = int(sys.argv[4])

    prepare()

    dataset = load_dataset(DATASET)

    accumulated_training_time, most_accurate_model, best_seed, res = train(
        dataset, MODEL_NAMES, number_of_seeds, number_of_training_samples,
        number_of_validation_samples)

    predicting_time = predict(dataset, most_accurate_model, best_seed, number_of_prediction_samples)

    res += f"train time,{accumulated_training_time}\npredict time,{predicting_time}\n"
    with open(RESULTS_PATH, "w") as f:
        f.write(res)
    # TODO Note: During prediction, unlike during training, you should not pad the samples at all.


if __name__ == "__main__":
    main()
