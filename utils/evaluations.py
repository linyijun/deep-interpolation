import numpy as np
import math
from sklearn.metrics import r2_score


def compute_error(ground_truth, predictions):
    
    mse = np.nanmean((ground_truth - predictions) ** 2)
    rmse = math.sqrt(mse)
    r2 = r2_score(ground_truth[~np.isnan(ground_truth)], predictions[~np.isnan(ground_truth)])
    return rmse, r2


def compute_error_by_locations(ground_truth, predictions, locations):
    
    for i, loc in enumerate(locations):
        rmse, r2 = compute_error(ground_truth[:, i], predictions[:, i])
        print('Error @{}: RMSE = {}, R2 = {}'.format(loc, rmse, r2))
