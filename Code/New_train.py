"""
Created by team 12 for CE903 at the University of Essex, 2020-21.
Adapted with permission from our supervisor Dr Anirban Chowdhury.

This program loads raw EEG training data and labels from Matlab files, filters the EEG data to return the beta
and mu band channels, and then trains a number of classifiers on that data.

"""

# import statements
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

from sklearn.preprocessing import StandardScaler

from userDefFunc import my_function
from userDefFunc import f_logVar

# configuration variables
data_source = "/home/jak/Uni/CE903/Dataset/"  # location of MATLAB data files

subject_id_list = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
'''
# time frames for a 2s frame
time_frames = {
    "P01": {"start": 3.5, "end": 5.5},
    "P02": {"start": 4, "end": 6},
    "P03": {"start": 3.5, "end": 5.5},
    "P04": {"start": 5, "end": 7},
    "P05": {"start": 4.5, "end": 6.5},
    "P06": {"start": 5, "end": 7},
    "P07": {"start": 5, "end": 7},
    "P08": {"start": 6, "end": 8},
    "P09": {"start": 4.5, "end": 6.5},
    "P10": {"start": 5, "end": 7}
}
'''

# time frames for a 2.5s frame
time_frames = {
    "P01": {"start": 5.5, "end": 8},
    "P02": {"start": 5, "end": 7.5},
    "P03": {"start": 3.5, "end": 6},
    "P04": {"start": 5.5, "end": 8},
    "P05": {"start": 4, "end": 6.5},
    "P06": {"start": 5, "end": 7.5},
    "P07": {"start": 3.5, "end": 6},
    "P08": {"start": 5, "end": 7.5},
    "P09": {"start": 4, "end": 6.5},
    "P10": {"start": 3.5, "end": 6}
}

all_times = [3, 8]

debug_data_loading = False
debug_bandpass_filter = False
debug_dimension_change = False

accuracies = []


# definitions
def load_data(param_id: str) -> (np.array, np.array, np.array):
    """Loads the data of the given subject"""

    file_name_with_path = data_source + "parsed_" + param_id + "T.mat"

    if debug_data_loading:
        print(file_name_with_path)

    data_file = loadmat(file_name_with_path)
    raw_eeg_data = data_file["RawEEGData"]  # this is the EEG data we convert into features (the X variable)
    labels = data_file["Labels"]  # these are the labels which our model will fit against (the y variable)
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
        print("Duration of each trial is %d sec" % trial_duration_in_seconds)

    return raw_eeg_data, labels, sampling_rate


def select_frames(param_id: str, raw_eeg_data: np.array, sampling_rate: np.array) -> np.array:
    """Returns the given raw EEG data within a subject-specific time frame"""

    frame = time_frames[param_id]

    start_sampling_point = int(frame["start"] * sampling_rate)
    end_sampling_point = int(frame["end"] * sampling_rate)

    result = raw_eeg_data[:, :, start_sampling_point:end_sampling_point]

    return result


def bandpass_filter(raw_eeg_data: np.array, sampling_rate: int) -> (np.array, np.array, np.array, np.array):
    """Filters the raw EEG data to return the mu and beta band"""

    order = 4
    mu_band = ([8, 12] / sampling_rate) * 2
    mu_band = mu_band[0]
    beta_band = ([16, 24] / sampling_rate) * 2
    beta_band = beta_band[0]

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
    return mu_raw_eeg_data, beta_raw_eeg_data, mu_band, beta_band


def create_class_labels(labels: np.array):
    """Returns class 1 and class 2 labels"""

    class_1 = np.where(labels == 1)
    class_1 = class_1[0]
    class_2 = np.where(labels == 2)
    class_2 = class_2[0]

    return class_1, class_2


def reshape_filter_data(filter_data: np.array, class_1_labels: np.array, class_2_labels: np.array):
    """Splits the filtered data into class 1 and 2, then reshapes it"""

    filter_data_class_1 = filter_data[class_1_labels, :, :]  # band data from class 1 trials
    filter_data_class_2 = filter_data[class_2_labels, :, :]  # band data from class 2 trials

    if debug_dimension_change:
        print("filter class 1 Raw EEG data shape", filter_data_class_1.shape)
        print("filter class 2 Raw EEG data shape", filter_data_class_2.shape)

    filter_data_class_1 = np.swapaxes(filter_data_class_1, 0, 1)
    filter_data_class_2 = np.swapaxes(filter_data_class_2, 0, 1)

    if debug_dimension_change:
        print("filter class 2 data shape after axis swap ", filter_data_class_1.shape)
        print("filter class 2 data shape after axis swap ", filter_data_class_2.shape)

    class_1_dimensions = filter_data_class_1.shape
    class_2_dimensions = filter_data_class_2.shape

    csp_filter_data_class_1 = np.reshape(filter_data_class_1,
                                         (class_1_dimensions[0], class_1_dimensions[1] * class_1_dimensions[2]))
    csp_filter_data_class_2 = np.reshape(filter_data_class_2,
                                         (class_2_dimensions[0], class_2_dimensions[1] * class_2_dimensions[2]))

    if debug_dimension_change:
        print("Shape of filter class 1 after converting from 3 dim to 2 dim ", csp_filter_data_class_1.shape)
        print("Shape of filter class 2 after converting from 3 dim to 2 dim ", csp_filter_data_class_2.shape)

    return csp_filter_data_class_1, csp_filter_data_class_2


def make_features_from_trials(mu_raw_eeg_data, beta_raw_eeg_data, csp_mu, csp_beta):
    """Creates features by combining trials"""

    no_of_trials = mu_raw_eeg_data.shape[0]
    features = np.empty((no_of_trials, 4))

    for trial_index in range(no_of_trials):
        # obtain samples for this trial
        mu_temp = mu_raw_eeg_data[trial_index, :, :]
        beta_temp = beta_raw_eeg_data[trial_index, :, :]

        # calculate the Z matrix for either band
        mu_temp = np.matmul(csp_mu, mu_temp)
        beta_temp = np.matmul(csp_beta, beta_temp)

        # calculate the log-variance for either band
        log_var_mu = f_logVar(mu_temp)
        log_var_beta = f_logVar(beta_temp)

        features[trial_index, :] = [log_var_mu[0], log_var_mu[len(log_var_mu) - 1], log_var_beta[0],
                                    log_var_beta[len(log_var_beta) - 1]]

    return features


def create_lda():
    """Creates a pipeline with an LDA classifier"""

    clf = LinearDiscriminantAnalysis()

    pipe = Pipeline(steps=[('classifier', clf)])

    param_grid = {
        'classifier__solver': ['svd', 'lsqr', 'eigen'],
    }

    result = GridSearchCV(pipe, param_grid, cv=10)

    return result


def create_svm():
    """Creates a pipeline with an SVM classifier"""

    # clf = svm.SVC()  # non-linear kernels often cause overfitting
    classifier = svm.LinearSVC()  # more efficient than regular SVC for linear kernel
    scaler = StandardScaler()

    pipe = Pipeline(steps=[('scaler', scaler), ('classifier', classifier)])

    cs = np.logspace(-2, 5, 8)  # change from (0, 4, 5)
    #  gs = np.logspace(-5, 5, 11)  # change from (0, 4, 5), redundant for linear kernel
    param_grid = {
        #  'classifier__gamma': gs, #  redundant for linear kernel
        'classifier__C': cs,
        'classifier__max_iter': [10000000],
        'classifier__dual': [False],  # should be false if no_of_samples > no_of_features
        'scaler__with_mean': [False],
        'scaler__with_std': [True, False],
    }

    result = GridSearchCV(pipe, param_grid, cv=10)

    return result


# main program
for subject_index in range(10):
    # get the subject's id as in matlab files, e.g. "PO1"
    subject_id = subject_id_list[subject_index]

    # load the subject's matlab training file
    subject_raw_eeg, subject_labels, subject_sampling_rate = load_data(subject_id)

    # slice the eeg data within a specified time frame
    subject_slice_eeg = select_frames(subject_id, subject_raw_eeg, subject_sampling_rate)

    # filter the raw eeg data. we want the mu and beta band channels
    subject_mu_eeg, subject_beta_eeg, subject_mu_band, subject_beta_band = \
        bandpass_filter(subject_slice_eeg, subject_sampling_rate)

    # get the class 1 and 2 labels
    subject_class_1_labels, subject_class_2_labels = create_class_labels(subject_labels)

    # reshape mu and beta data so we can apply the CSP algorithm
    subject_mu_eeg_class_1, subject_mu_eeg_class_2 = \
        reshape_filter_data(subject_mu_eeg, subject_class_1_labels, subject_class_2_labels)
    subject_beta_eeg_class_1, subject_beta_eeg_class_2 = \
        reshape_filter_data(subject_beta_eeg, subject_class_1_labels, subject_class_2_labels)

    # apply CSP algorithm
    subject_csp_mu = my_function(subject_mu_eeg_class_1, subject_mu_eeg_class_2)
    subject_csp_beta = my_function(subject_beta_eeg_class_1, subject_beta_eeg_class_2)

    # create features for training
    subject_features = make_features_from_trials(subject_mu_eeg, subject_beta_eeg, subject_csp_mu, subject_csp_beta)

    X = subject_features
    y = subject_labels[:, 0]
    #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    search = create_lda()
    # search = create_svm()
    search.fit(X, y)

    print("Subject " + subject_id)
    print("Best train accuracy", search.best_score_, " Best params", search.best_params_)

    accuracies.append(search.best_score_)

    outputFilename = "trainedModel" + subject_id
    with open(outputFilename, 'wb') as f:
        pickle.dump([search.best_estimator_, subject_csp_mu, subject_csp_beta, subject_mu_band, subject_beta_band], f)

print()
print("Average accuracy across all subjects:", np.mean(accuracies))
