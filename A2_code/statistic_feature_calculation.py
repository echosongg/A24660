import numpy as np
import pandas as pd
# Linear Features Calculation
import numpy as np
from scipy.stats import entropy
from itertools import permutations


# 2. Approximate Entropy
def approximate_entropy(U, m, r):
    """Compute approximate entropy (ApEn) of a time series."""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)
    return abs(_phi(m + 1) - _phi(m))


# 3. Fuzzy Entropy
def fuzzy_entropy(U, m, r, f=2):
    """Compute fuzzy entropy of a time series."""

    def _phi(m):
        x = [U[j] for j in range(N - m + 1)]  # Form subsequences
        C = [0] * (N - m + 1)
        for i in range(N - m + 1):
            for j in range(N - m + 1):
                if abs(U[j] - U[i]) < r:
                    C[i] += 1
                else:
                    C[i] += np.exp(-abs(U[j] - U[i]) / r) / f
        return sum(C)

    N = len(U)
    return np.log(_phi(m) / _phi(m + 1))


# 4. Shannon's Entropy
def shannons_entropy(data):
    p_data = data.value_counts() / len(data)
    return entropy(p_data)


# 5. Permutation Entropy
def permutation_entropy(time_series, m, delay):
    n = len(time_series)
    permutations_list = list(permutations(range(m)))
    c = [0] * len(permutations_list)

    # Create the series of rank vectors
    for i in range(n - delay * (m - 1)):
        sorted_idx = list(np.argsort(time_series[i:i + delay * m:delay]))
        for j, perm in enumerate(permutations_list):
            if sorted_idx == list(perm):
                c[j] += 1

    c = [element for element in c if element != 0]
    c = np.array(c) / float(sum(c))
    return -np.sum(c * np.log(c))


# 6. Hjorth Parameters
def hjorth_parameters(time_series):
    first_diff = np.diff(time_series)
    second_diff = np.diff(time_series, 2)

    var_zero = np.var(time_series)
    var_d1 = np.var(first_diff)
    var_d2 = np.var(second_diff)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity


# 7. Hurst Exponent
def hurst_exponent(time_series):
    lags = range(2, 20)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


data = pd.read_csv('./preprocessing_data.csv')
# Mean, Maximum, Minimum, Standard Deviation were already calculated, so we'll just add the rest
#linear
data['std_F7'] = data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']].std(axis=1)
data['iqr_F7'] = data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']].apply(lambda row: np.percentile(row, 75) - np.percentile(row, 25), axis=1)
data['sum_F7'] = data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']].sum(axis=1)
data['variance_F7'] = data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']].var(axis=1)
data['skewness_F7'] = data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']].skew(axis=1)
data['kurtosis_F7'] = data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']].kurtosis(axis=1)
data['rms_F7'] = np.sqrt(np.mean(data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']]**2, axis=1))
data['signal_power_F7'] = np.mean(data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']]**2, axis=1)
data['integrated_signals_F7'] = np.trapz(data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']], axis=1)
data['log_detector_F7'] = np.log(np.abs(data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']])).mean(axis=1)
data['diff_std_F7'] = data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']].diff(axis=1).std(axis=1)

# First differences
first_diff = data[['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']].diff(axis=1)
data['mean_abs_first_diff_F7'] = np.abs(first_diff).mean(axis=1)

# Saving the combined reshaped data to CSV
st_file_path = "./preprocessing_st_data.csv"
data.to_csv(st_file_path, index=False)

print("save over")
