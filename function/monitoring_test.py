from sklearn.metrics import precision_score, recall_score, roc_auc_score
import numpy as np

def calculate_csi(baseline, current, bins=10, add_bin_zero=False):
    """
    Calculate Characteristic Stability Index (CSI) between baseline and current distributions.
    Args:
    - baseline (array-like): Array of characteristic values from the baseline distribution.
    - current (array-like): Array of characteristic values from the current distribution.
    - bins (int): Number of bins to divide the values into. Default is 10.
    Returns:
    - csi (float): The calculated CSI value.
    """
    # Step 1: Create bins for both distributions
    if len(set(baseline)) < 10:
        if add_bin_zero == True:
            bin_edges = list(sorted(set([0] + baseline.to_list()))) + [np.inf]

        else:
            bin_edges = list(sorted(set(baseline))) + [np.inf]

    else:
        bin_edges = np.histogram_bin_edges(baseline, bins=bins)
    #print(bin_edges)
    # Step 2: Calculate the proportion of baseline and current in each bin
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)
    # Step 3: Convert counts to proportions
    baseline_proportions = baseline_counts / len(baseline)
    current_proportions = current_counts / len(current)

    # Step 4: Replace zeros to avoid division and log errors
    baseline_proportions = np.where(baseline_proportions == 0, (1 - 0.05 ** (1 / len(baseline))), baseline_proportions)
    current_proportions = np.where(current_proportions == 0, (1 - 0.05 ** (1 / len(current))), current_proportions)
    #print(baseline_proportions)
    #print(current_proportions)
    # Step 5: Calculate CSI using the formula
    csi = np.sum((current_proportions - baseline_proportions) * np.log(current_proportions / baseline_proportions))
    return csi

def ar(Y, X):
    if len(np.unique(Y)) != 2:
        return 0
    else:
        return 2*roc_auc_score(Y,X) - 1

