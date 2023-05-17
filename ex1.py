import sys
from transformers import AutoModelForSequenceClassification  # todo use this one

MODEL_NAMES = ["bert-base-uncased", "roberta-base", "google/electra-base-generator"]
RESULTS_PATH = "res.txt"
PREDICTIONS_OUTPUT_PATH = "predictions.txt"


def finetune_sentiment_analysis_model_on_sst2(
        model_name, number_of_seeds, number_of_training_samples, number_of_validation_samples):
    """
    Fine-tunes a sentiment analysis model on the SST2 dataset
    :param number_of_seeds: -1 means use all
    :param number_of_training_samples: -1 means use all
    :param number_of_validation_samples: -1 means use all
    :param model_name: name of pre-trained model to use, as listed in HuggingFace Hub
    :return: mean accuracy of the model as predicted on the validation samples across all seeds,
             its standard deviation, the seed used for the best accuracy and the accumulated
             training time across all seeds in seconds.
    """
    # todo
    # todo use the function "set_seed" from the transformers package before starting every new run.
    # todo Remember to run the model.eval() command before prediction
    # todo make sure training time is in seconds
    return None, None, None, None  # todo change


def predict_sst2(model, seed, number_of_prediction_samples,
                 predictions_output_path=PREDICTIONS_OUTPUT_PATH):
    """
    Uses the fine-tuned model with the specified name and seed to predict the sentiment of samples
    from the SST2 test set, and saves the results in the specified path
    :param model:
    :param seed:
    :param number_of_prediction_samples: -1 means use all
    :param predictions_output_path:
    :return: prediction time in seconds
    """
    # todo
    return None


def train_sst2(model_names, number_of_seeds, number_of_training_samples,
               number_of_validation_samples):
    """
    Performs fine-tuning of all combinations of pre-trained model names and number of seeds, to
    sentiment analysis on the SST2 dataset, assesses their accuracy on the validation set, and
    documents their mean and standard deviation of accuracy across all seeds
    :param model_names: list of model names that fit the names on the HuggingFace Hub
    :param number_of_seeds:
    :param number_of_training_samples:
    :param number_of_validation_samples:
    :return: accumulated training time of all models (in seconds), the name of model with the
    highest mean accuracy, the seed with the highest accuracy with that model and the string
    documenting the results.
    """
    res = ""
    best_mean_accuracy = 0  # todo validate that this is the worse possible
    most_accurate_model = None
    best_seed = None
    accumulated_training_time = 0  # todo validate right format for seconds
    for model_name in model_names:
        mean_accuracy, accuracy_std, model_best_seed, training_time = \
            finetune_sentiment_analysis_model_on_sst2(model_name, number_of_seeds,
                                                      number_of_training_samples,
                                                      number_of_validation_samples)
        res += f"{model_name},{mean_accuracy} +- {accuracy_std}\n"
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            most_accurate_model, best_seed = model_name, model_best_seed
        accumulated_training_time += training_time
    res += "----\n"
    return accumulated_training_time, most_accurate_model, best_seed, res


def main():
    if len(sys.argv) != 5:
        raise ValueError("Wrong number of arguments, for usage see README.md")
    number_of_seeds = sys.argv[1]
    number_of_training_samples = sys.argv[2]
    number_of_validation_samples = sys.argv[3]
    number_of_prediction_samples = sys.argv[4]

    accumulated_training_time, most_accurate_model, best_seed, res = train_sst2(
        MODEL_NAMES, number_of_seeds, number_of_training_samples, number_of_validation_samples)

    predicting_time = predict_sst2(most_accurate_model, best_seed, number_of_prediction_samples)

    res += f"train time,{accumulated_training_time}\npredict time,{predicting_time}\n"
    with open(RESULTS_PATH, "w") as f:
        f.write(res)
    # TODO Note: During prediction, unlike during training, you should not pad the samples at all.


if __name__ == "__main__":
    main()
