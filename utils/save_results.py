import json
import numpy as np


def save_results(results_path: str, hyperparameters: dict, validation_score: float, test_score: float, score_type: str):

    if isinstance(validation_score, np.float32):
        validation_score = float(validation_score)
    if isinstance(test_score, np.float32):
        test_score = float(test_score)

    # try to load the best configuration found so far
    try:
        with open(f'{results_path}/validation_score.json', 'r') as f:
            best_validation_score = json.load(f)
    except FileNotFoundError:
        best_validation_score = {score_type: 0.0}

    if validation_score > best_validation_score[score_type]:
        # Save the best hyperparameters and score
        best_validation_score = {score_type: validation_score}
        with open(f'{results_path}/hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        with open(f'{results_path}/validation_score.json', 'w') as f:
            json.dump(best_validation_score, f, indent=4)

        with open(f'{results_path}/test_score.json', 'w') as f:
            json.dump({score_type: test_score}, f, indent=4)
