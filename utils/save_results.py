import json
import numpy as np


def save_results(results_path: str, hyperparameters: dict, mean_validation_score: float, std_validation_score: float,
                 mean_test_score: float, std_test_score: float, score_type: str, relation: str):

    if isinstance(mean_validation_score, np.float32):
        mean_validation_score = float(mean_validation_score)
    if isinstance(mean_test_score, np.float32):
        mean_test_score = float(mean_test_score)
    if isinstance(std_validation_score, np.float32):
        std_validation_score = float(std_validation_score)
    if isinstance(std_test_score, np.float32):
        std_test_score = float(std_test_score)

    # try to load the best configuration found so far
    try:
        with open(f'{results_path}/validation_score.json', 'r') as f:
            best_validation_score = json.load(f)
    except FileNotFoundError:
        if relation == 'greater':
            best_validation_score = {'mean_' + score_type: 0.0, 'std_' + score_type: 0.0}
        elif relation == 'less':
            best_validation_score = {'mean_' + score_type: np.inf, 'std_' + score_type: np.inf}

    if relation == 'greater':
        update_condition = mean_validation_score > best_validation_score['mean_' + score_type]
    elif relation == 'less':
        update_condition = mean_validation_score < best_validation_score['mean_' + score_type]
    if update_condition:
        # Save the best hyperparameters and score
        best_validation_score = {'mean_' + score_type: mean_validation_score, 'std_' + score_type: std_validation_score}
        with open(f'{results_path}/hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        with open(f'{results_path}/validation_score.json', 'w') as f:
            json.dump(best_validation_score, f, indent=4)

        with open(f'{results_path}/test_score.json', 'w') as f:
            json.dump({'mean_' + score_type: mean_test_score, 'std_' + score_type: std_test_score}, f, indent=4)
