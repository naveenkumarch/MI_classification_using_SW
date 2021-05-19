"""
Created by team 12 for CE903 at the University of Essex, 2020-21.
Adapted with permission from our supervisor Dr Anirban Chowdhury.

This program evaluates the model trained under a separate file.
"""

# import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal, stats
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

from userDefFunc import my_function
from userDefFunc import f_logVar

# configuration variables
data_source = "/home/jak/Uni/CE903/Dataset/"
trained_model_source = "/home/jak/Uni/CE903/codeBase/"

subject_id_list = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]

# these are time frames for a 2s sliding window
'''
time_frames = {
    "P01": {"start": 3, "end": 6},
    "P02": {"start": 3.5, "end": 6.5},
    "P03": {"start": 3.5, "end": 6.5},
    "P04": {"start": 3, "end": 6},
    "P05": {"start": 4.5, "end": 7.5},
    "P06": {"start": 4, "end": 7},
    "P07": {"start": 4, "end": 7},
    "P08": {"start": 5, "end": 8},
    "P09": {"start": 4.5, "end": 7.5},
    "P10": {"start": 3, "end": 6}
}
'''

# these are time frames for a 2.5s sliding window
time_frames = {
    "P01": {"start": 4.5, "end": 8},
    "P02": {"start": 4.5, "end": 8},
    "P03": {"start": 3, "end": 6.5},
    "P04": {"start": 4.5, "end": 8},
    "P05": {"start": 3.5, "end": 7},
    "P06": {"start": 4.5, "end": 8},
    "P07": {"start": 3, "end": 6.5},
    "P08": {"start": 4.5, "end": 8},
    "P09": {"start": 3.5, "end": 7},
    "P10": {"start": 3, "end": 6.5}
}

sliding_window_length = 2.5  # changed from 2
stride = 0.1

debug_data_loading = False
debug_bandpass_filter = False
debug_dimension_change = False

accuracies_mode = []
accuracies_lcr = []


# definitions
def classify_by_mode(predictions):
    """Return the mode of all given predictions within a sliding window"""

    return stats.mode(predictions)[0][0]


def classify_by_lcr(predictions):
    """Return the longest consecutive repeating character within a sliding window"""

    no_of_predictions = len(predictions)
    max_count = 0
    answer = predictions[0]
    curr_count = 1

    for i in range(no_of_predictions - 1):
        if predictions[i] == predictions[i + 1]:
            curr_count += 1
        else:
            if curr_count > max_count:
                max_count = curr_count
                answer = predictions[i]
            curr_count = 1
    return answer


def load_data(param_id: str) -> (np.array, np.array, np.array):
    """Loads the data of the given subject"""

    file_name_with_path = data_source + "parsed_" + param_id + "E.mat"

    if debug_data_loading:
        print(file_name_with_path)

    data_file = loadmat(file_name_with_path)
    raw_eeg_data = data_file["RawEEGData"]
    labels = data_file["Labels"]
    sampling_rate = data_file["sampRate"]

    if debug_data_loading:
        dimensions = raw_eeg_data.shape
        no_of_trials = dimensions[0]  # getting the number of trials
        no_of_channels = dimensions[1]  # getting the number of channels
        no_of_samples_in_trial = dimensions[2]  # getting the number of samples per trial
        trial_duration_in_seconds = no_of_samples_in_trial / sampling_rate

        print("No of Trails conducted for a subject", no_of_trials)
        print("No of EEG channels measured for a subject during each trail", no_of_channels)
        print("No of samples noted during each trail", no_of_samples_in_trial)
        print("Duration of each trail is %d sec" % trial_duration_in_seconds)

    return raw_eeg_data, labels, sampling_rate


def select_frames(param_id: str, raw_eeg_data: np.array, sampling_rate: np.array) -> np.array:
    """Returns the given raw EEG data within a subject-specific time frame"""

    frame = time_frames[param_id]

    start_sampling_point = int(frame["start"] * sampling_rate)
    end_sampling_point = int(frame["end"] * sampling_rate)

    result = raw_eeg_data[:, :, start_sampling_point:end_sampling_point]

    return result


def bandpass_filter(raw_eeg_data: np.array, mu_band: np.array, beta_band: np.array) -> (np.array, np.array):
    """Filters the raw EEG data with given bands"""

    order = 4

    mu_b, mu_a = signal.butter(order, mu_band, 'bandpass', analog=False)
    beta_b, beta_a = signal.butter(order, beta_band, 'bandpass', analog=False)

    dimensions = raw_eeg_data.shape
    mu_raw_eeg_data = np.empty(dimensions)
    beta_raw_eeg_data = np.empty(dimensions)

    no_of_trials = dimensions[0]
    for trial_index in range(no_of_trials):
        sig = raw_eeg_data[trial_index, :, :]

        if debug_bandpass_filter:
            print("Properties:")
            print(type(sig))
            print(sig.shape)

        mu_temp = signal.lfilter(mu_b, mu_a, sig, 1)
        beta_temp = signal.lfilter(beta_b, beta_a, sig, 1)

        mu_raw_eeg_data[trial_index, :, :] = mu_temp
        beta_raw_eeg_data[trial_index, :, :] = beta_temp
    return mu_raw_eeg_data, beta_raw_eeg_data


def load_pickle_data(param_id: str) -> any:
    """Returns the model, beta/mu bands and beta/mu csp for the given subject"""

    # load pickle file which contains trained model and important information
    path_to_pickle_file = trained_model_source + "trainedModel" + param_id
    with open(path_to_pickle_file, 'rb') as model_file:
        pickle_data = pickle.load(model_file)

    # extract data found in pickle file
    model = pickle_data[0]
    csp_mu = pickle_data[1]
    csp_beta = pickle_data[2]
    mu_band = pickle_data[3]
    beta_band = pickle_data[4]

    return model, csp_mu, csp_beta, mu_band, beta_band


def make_features_from_trials(mu_raw_eeg_data, beta_raw_eeg_data, csp_mu, csp_beta) -> np.array:
    """Creates features by combining trials within a sliding window"""

    no_of_trials = mu_raw_eeg_data.shape[0]
    no_of_sliding_windows = int(
        (time_frames[subject_id]["end"] - time_frames[subject_id]["start"] - sliding_window_length) / stride)

    features = np.empty((no_of_trials, no_of_sliding_windows, 4))

    for trial_index in range(no_of_trials):
        for window in range(no_of_sliding_windows):
            # find the first and last set of samples within this sliding window
            start_sampling_point = int((window * stride) * 512)
            end_sampling_point = int((sliding_window_length + (window * stride)) * 512)

            # obtain samples within sliding window for either band
            mu_temp = mu_raw_eeg_data[trial_index, :, start_sampling_point:end_sampling_point]
            beta_temp = beta_raw_eeg_data[trial_index, :, start_sampling_point:end_sampling_point]

            # calculate the Z matrix for either band
            mu_temp = np.matmul(csp_mu, mu_temp)
            beta_temp = np.matmul(csp_beta, beta_temp)

            # calculate the log-variance for either band
            log_var_mu = f_logVar(mu_temp)
            log_var_beta = f_logVar(beta_temp)

            # create features for this window
            features[trial_index, window, :] = [log_var_mu[0], log_var_mu[len(log_var_mu) - 1], log_var_beta[0],
                                                log_var_beta[len(log_var_beta) - 1]]

    return features


def test_on_window(model, features) -> (list, list):
    """Tests the model on provided features and returns the results for both LCR and mode"""

    no_of_trials = features.shape[0]
    lcr_results = []
    mode_results = []

    for trial_index in range(no_of_trials):
        X = features[trial_index, :, :]
        y_hat = model.predict(X)

        lcr_result = classify_by_lcr(y_hat)
        lcr_results.append(lcr_result)

        mode_result = classify_by_mode(y_hat)
        mode_results.append(mode_result)

    return lcr_results, mode_results


# main program
for subject_index in range(10):
    # get the subject's id as in matlab files, e.g. "PO1"
    subject_id = subject_id_list[subject_index]

    # load the subject's matlab training file
    subject_raw_eeg, subject_labels, subject_sampling_rate = load_data(subject_id)

    # slice the eeg data within a specified time frame
    subject_slice_eeg = select_frames(subject_id, subject_raw_eeg, subject_sampling_rate)

    # load pickle file which contains trained model and important information
    subject_model, subject_csp_mu, subject_csp_beta, subject_mu_band, subject_beta_band = load_pickle_data(subject_id)

    # filter the raw eeg data using provided bands
    subject_mu_eeg, subject_beta_eeg = \
        bandpass_filter(subject_slice_eeg, subject_mu_band, subject_beta_band)

    # create features for testing
    subject_features = make_features_from_trials(subject_mu_eeg, subject_beta_eeg, subject_csp_mu, subject_csp_beta)
    y = subject_labels[:, 0]

    # test model on features for both lcr and mode
    subject_lcr_results, subject_mode_results = test_on_window(subject_model, subject_features)

    # Get accuracy of lcr and mode methods
    test_accuracy_lcr = accuracy_score(y, subject_lcr_results)
    test_accuracy_mode = accuracy_score(y, subject_mode_results)

    print("Subject " + subject_id)
    print("LCR acc", test_accuracy_lcr)
    print("Mode acc", test_accuracy_mode)

    # get all accuracies so far for average calculation
    accuracies_mode.append(test_accuracy_mode)
    accuracies_lcr.append(test_accuracy_lcr)

print()
print("Average LCR accuracy across all subjects", np.mean(accuracies_lcr))
print("Average mode accuracy across all subjects", np.mean(accuracies_mode))
