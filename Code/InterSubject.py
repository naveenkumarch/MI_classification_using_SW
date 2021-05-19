#!/usr/bin/env python
# import statements
from scipy.io import loadmat
from scipy import signal
from scipy import stats
import numpy as np

from userDefFunc import my_function
from userDefFunc import f_logVar

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# configuration
dataSRC = "/home/jak/Uni/CE903/Dataset/"
trainedModelSRC = "/home/jak/Uni/CE903/codeBase/"

subList = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]

"""
train_time_frames = {
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

test_time_frames = {
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
"""

# time frames for a 2.5s frame
train_time_frames = {
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


# these are time frames for a 2.5s sliding window
test_time_frames = {
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

# this is rankings for worst to best performance for 2.5s window length
subject_rankings = {
    "P01": ["P06", "P10", "P07", "P08", "P04", "P09", "P05", "P01", "P03", "P02"],
    "P02": ["P10", "P09", "P08", "P07", "P04", "P01", "P05", "P06", "P03", "P02"],
    "P03": ["P06", "P01", "P10", "P05", "P04", "P09", "P08", "P07", "P03", "P02"],
    "P04": ["P06", "P07", "P09", "P10", "P08", "P03", "P02", "P05", "P01", "P04"],
    "P05": ["P02", "P09", "P07", "P10", "P06", "P08", "P04", "P03", "P01", "P05"],
    "P06": ["P10", "P07", "P08", "P01", "P09", "P04", "P03", "P05", "P06", "P02"],
    "P07": ["P06", "P04", "P03", "P01", "P05", "P02", "P09", "P08", "P10", "P07"],
    "P08": ["P06", "P04", "P03", "P02", "P01", "P05", "P09", "P08", "P07", "P10"],
    "P09": ["P06", "P05", "P04", "P03", "P02", "P01", "P10", "P08", "P09", "P07"],
    "P10": ["P09", "P07", "P06", "P05", "P04", "P03", "P02", "P01", "P08", "P10"]
}

sliding_window_length = 2
stride = 0.1

debug_data_loading = False
debug_bandpass_filter = False
debug_dimension_change = False

train_accuracies = []
mode_test_accuracies = []
LCR_test_accuracies = []

def select_configuration_parameters(time_frames: dict, no_of_windows: int = 10, stride: int = 0.1):
    windows_time = no_of_windows * stride

    max_duration = 8
    for patient in time_frames:
        assert sliding_window_length + windows_time < max_duration, \
            "Can not generate given no of sliding windows with current settings please configure values properly"


        if (time_frames[patient]["start"] - (windows_time / 2)) > 3:
            time_frames[patient]["start"] = time_frames[patient]["start"] - (windows_time / 2)
        else:
            time_frames[patient]["end"] = time_frames[patient]["end"] + (
                        time_frames[patient]["start"] - (windows_time / 2) - 3)
            time_frames[patient]["start"] = 3
        if (time_frames[patient]["end"] + (windows_time / 2)) < max_duration:
            time_frames[patient]["end"] = time_frames[patient]["end"] + (windows_time / 2)
        else:
            time_frames[patient]["start"] = time_frames[patient]["start"] - (
                        time_frames[patient]["end"] + (windows_time / 2) - max_duration)
            time_frames[patient]["end"] = max_duration


def set_time_frames(subject_id: str):
    """Returns the testing time frames replaced with the subject's training time"""

    current_frame = test_time_frames[subject_id]
    result = train_time_frames.copy()
    result[subject_id] = current_frame
    return result


def classify_by_mode(predictions):
    """Returns the mode of the predictions"""

    return stats.mode(predictions)[0][0]


def classify_by_lcr(predictions):
    """Returns the longest consecutive repeating character"""

    n = len(predictions)
    max_count = 0
    ans = predictions[0]
    curr_count = 1

    for i in range(n - 1):
        if predictions[i] == predictions[i + 1]:
            curr_count += 1
        else:
            if curr_count > max_count:
                max_count = curr_count
                ans = predictions[i]
            curr_count = 1
    return ans


def load_data(subject_id: str, is_load_training: bool = True) -> (np.array, np.array, np.array):
    """Loads the data of the given subject id"""

    if is_load_training:
        file_name = dataSRC + "parsed_" + subject_id + "T.mat"
    else:
        file_name = dataSRC + "parsed_" + subject_id + "E.mat"

    if debug_data_loading:
        print(file_name)

    data_file = loadmat(file_name)
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

    # return Temp_data_holder, labels, sampling_rate


def select_frames(param_id: str, raw_eeg_data: np.array, sampling_rate: np.array) -> np.array:
    """Returns the given raw EEG data within a subject-specific time frame"""

    # create time frame from target's testing time and other's training time
    current_frame = test_time_frames[param_id]
    time_frames = train_time_frames.copy()
    time_frames[param_id] = current_frame

    frame = time_frames[param_id]

    start_sampling_point = int(frame["start"] * sampling_rate)
    end_sampling_point = int(frame["end"] * sampling_rate)

    result = raw_eeg_data[:, :, start_sampling_point:end_sampling_point]

    return result


def BandPass_filter(RawEEGData, sampRate):
    "Filters the raw EEG data to return the mu and beta band"
    dim = RawEEGData.shape
    order = 4
    muBand = ([8, 12] / sampRate) * 2
    muBand = muBand[0]
    betaBand = ([16, 24] / sampRate) * 2
    betaBand = betaBand[0]

    mu_B, mu_A = signal.butter(order, muBand, 'bandpass', analog=False)
    beta_B, beta_A = signal.butter(order, betaBand, 'bandpass', analog=False)
    muRawEEGData = np.empty((dim))
    betaRawEEGData = np.empty((dim))

    for trlIndex in range(dim[0]):
        sig = RawEEGData[trlIndex, :, :]
        if debug_bandpass_filter == True:
            print("Properties:")
            print(type(sig))
            print(sig.shape)
        mu_temp = signal.lfilter(mu_B, mu_A, sig, 1)
        beta_temp = signal.lfilter(beta_B, beta_A, sig, 1)

        muRawEEGData[trlIndex, :, :] = mu_temp
        betaRawEEGData[trlIndex, :, :] = beta_temp
    return muRawEEGData, betaRawEEGData, muBand, betaBand


def createLDA(X, y):
    clf = LinearDiscriminantAnalysis()

    pipe = Pipeline(steps=[('classifier', clf)])

    param_grid = {
        'classifier__solver': ['svd'],
        # 'classifier__solver': ['svd', 'lsqr', 'eigen'],
        # 'classifier__solver': ['svd', 'lsqr', 'eigen'],
        # 'classifier__shrinkage': ['auto', 0.5],
        # 'classifier__priors': ['linear'],
        # 'classifier__n_components': ['linear'],
        # 'classifier__store_covariance': ['True', 'False'],
        # 'classifier__tol': ['linear'],
        # 'classifier__covariance_estimator': ['linear'],
    }

    search = GridSearchCV(pipe, param_grid, cv=10)
    search.fit(X, y)

    return search.best_score_, search.best_estimator_


def createSVM(X, y):
    clf = svm.SVC()

    pipe = Pipeline(steps=[('classifier', clf)])

    Cs = np.logspace(0, 4, 5)
    Gs = np.logspace(0, 4, 5)
    param_grid = {
        'classifier__gamma': Gs,
        'classifier__C': Cs,
        'classifier__kernel': ['linear'],
    }

    search = GridSearchCV(pipe, param_grid, cv=10)
    search.fit(X, y)

    return search.best_score_, search.best_estimator_


# main program
test_subject = "P06"
time_frames = set_time_frames(test_subject)

test_raw_eeg, test_labels, test_sampling_rate = load_data(test_subject, False)
test_slice_eeg = select_frames(test_subject, test_raw_eeg, test_sampling_rate)

test_mu_raw_eeg_data, test_beta_raw_eeg_data, test_mu_band, test_beta_band = BandPass_filter(test_slice_eeg, test_sampling_rate)

no_of_sliding_windows = int((time_frames[test_subject]["end"] - time_frames[test_subject]["start"] - sliding_window_length) / stride)

test_no_of_trials = test_slice_eeg.shape[0]
features = np.empty((test_no_of_trials, no_of_sliding_windows, 4))

train_eeg_data = []
train_labels = []

target_rankings = subject_rankings[test_subject]
for outerIndex in range(10):  # Training
    training_subject = target_rankings[outerIndex]

    if training_subject == test_subject:
        continue

    train_raw_eeg, Labels, sampling_rate = load_data(training_subject)
    train_slice_eeg = select_frames(training_subject, train_raw_eeg, sampling_rate)

    dim = train_slice_eeg.shape
    noOfTrials = dim[0]
    noOfChann = dim[1]
    noOfSampsInTrial = dim[2]

    for trial in range(noOfTrials):
        # trainEEGData[(noOfTrials*incrementor)+trial] = RawEEGData[trial]
        train_eeg_data.append(train_slice_eeg[trial])
        train_labels.append(Labels[trial, 0])

    dimTrain = np.shape(train_eeg_data)
    print(dimTrain)
    noOfTrialsTrain = dimTrain[0]

    # print("trainEEGDATA", trainEEGData)
    train_eeg_data = np.array(train_eeg_data)

    muRawEEGData, betaRawEEGData, muBand, betaBand = BandPass_filter(train_eeg_data, sampling_rate)

    labelsCls1 = np.where(Labels == 1)
    labelsCls1 = labelsCls1[0]
    labelsCls2 = np.where(Labels == 2)
    labelsCls2 = labelsCls2[0]

    muRawEEGDataCls1 = muRawEEGData[labelsCls1, :, :]  # mu band data from class 1 trials
    muRawEEGDataCls2 = muRawEEGData[labelsCls2, :, :]  # mu band data from class 2 trials

    betaRawEEGDataCls1 = betaRawEEGData[labelsCls1, :, :]  # beta band data from class 1 trials
    betaRawEEGDataCls2 = betaRawEEGData[labelsCls2, :, :]  # beta band data from class 2 trials
    if debug_dimension_change == True:
        print("mu class 1 Raw EEG data shape", muRawEEGDataCls1.shape)
        print("mu class 2 Raw EEG data shape", muRawEEGDataCls2.shape)
        print("beta class 1 Raw EEG data shape", betaRawEEGDataCls1.shape)
        print("beta class 2 Raw EEG data shape", betaRawEEGDataCls2.shape)
    muRawEEGDataCls1 = np.swapaxes(muRawEEGDataCls1, 0, 1)
    muRawEEGDataCls2 = np.swapaxes(muRawEEGDataCls2, 0, 1)
    betaRawEEGDataCls1 = np.swapaxes(betaRawEEGDataCls1, 0, 1)
    betaRawEEGDataCls2 = np.swapaxes(betaRawEEGDataCls2, 0, 1)
    if debug_dimension_change == True:
        print("Mu band Cls1 data shape after axis swap ", muRawEEGDataCls1.shape)
        print("Mu band Cls2 data shape after axis swap ", muRawEEGDataCls2.shape)
        print("Beta band Cls1 data shape after axis swap ", betaRawEEGDataCls1.shape)
        print("Beta band Cls2 data shape after axis swap ", betaRawEEGDataCls2.shape)

    dim_mu_cls1 = muRawEEGDataCls1.shape
    dim_mu_cls2 = muRawEEGDataCls2.shape
    dim_beta_cls1 = betaRawEEGDataCls1.shape
    dim_beta_cls2 = betaRawEEGDataCls2.shape

    cspMuRawEEGDataCls1 = np.reshape(muRawEEGDataCls1, (dim_mu_cls1[0], dim_mu_cls1[1] * dim_mu_cls1[2]))
    cspMuRawEEGDataCls2 = np.reshape(muRawEEGDataCls2, (dim_mu_cls2[0], dim_mu_cls2[1] * dim_mu_cls2[2]))
    cspBetaRawEEGDataCls1 = np.reshape(betaRawEEGDataCls1, (dim_beta_cls1[0], dim_beta_cls1[1] * dim_beta_cls1[2]))
    cspBetaRawEEGDataCls2 = np.reshape(betaRawEEGDataCls2, (dim_beta_cls2[0], dim_beta_cls2[1] * dim_beta_cls2[2]))
    if debug_dimension_change == True:
        print("Shape of Mu Cls1 after converting from 3 dim to 2 dim ", cspMuRawEEGDataCls1.shape)
        print("Shape of Mu Cls2 after converting from 3 dim to 2 dim ", cspMuRawEEGDataCls2.shape)
        print("Shape of beta Cls1 after converting from 3 dim to 2 dim ", cspBetaRawEEGDataCls1.shape)
        print("Shape of beta Cls2 after converting from 3 dim to 2 dim ", cspBetaRawEEGDataCls2.shape)
    wCSP_mu = my_function(cspMuRawEEGDataCls1, cspMuRawEEGDataCls2)
    wCSP_beta = my_function(cspBetaRawEEGDataCls1, cspBetaRawEEGDataCls2)
    feat = np.empty((noOfTrialsTrain, 4))

    for trlIndex in range(noOfTrialsTrain):
        muTemp = muRawEEGData[trlIndex, :, :]
        betaTemp = betaRawEEGData[trlIndex, :, :]

        mu_temp = np.matmul(wCSP_mu, muTemp);  # calculating the Z matrix for mu band
        beta_temp = np.matmul(wCSP_beta, betaTemp);  # calculating the Z matrix for beta band

        logVarMu = f_logVar(mu_temp);  # calculating the logvariance for Mu
        logVarBeta = f_logVar(beta_temp);  # calculating the logvariance for Beta

        feat[trlIndex, :] = [logVarMu[0], logVarMu[len(logVarMu) - 1], logVarBeta[0], logVarBeta[len(logVarBeta) - 1]]

    X = feat
    y = train_labels[:]

    score, clf = createLDA(X, y)
    # score, clf = createSVM(X, y)
    # print("Training score", score)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # clf = LinearDiscriminantAnalysis()
    # clf.fit(X, y)

    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # accs.append(clf.score(X, y))

    # print("Subject #" + str(sb))
    for trlIndex in range(test_no_of_trials):
        for window in range(no_of_sliding_windows):
            muTemp = test_mu_raw_eeg_data[trlIndex, :,
                     int((window * stride) * 512):int((sliding_window_length + (window * stride)) * 512)]
            betaTemp = test_beta_raw_eeg_data[trlIndex, :,
                       int((window * stride) * 512):int((sliding_window_length + (window * stride)) * 512)]

            mu_temp = np.matmul(wCSP_mu, muTemp);  # calculating the Z matrix for mu band
            beta_temp = np.matmul(wCSP_beta, betaTemp);  # calculating the Z matrix for beta band

            logVarMu = f_logVar(mu_temp);  # calculating the logvariance for Mu
            logVarBeta = f_logVar(beta_temp);  # calculating the logvariance for Beta

            features[trlIndex, window, :] = [logVarMu[0], logVarMu[len(logVarMu) - 1], logVarBeta[0],
                                             logVarBeta[len(logVarBeta) - 1]]

    # score, search = createLDA(X, y)
    # accs.append(score)

    Test_Y = test_labels[:, 0]
    Pred_Y_LCR = []
    Pred_Y_Mode = []
    # print("Features length",features.shape)
    for trlIndex in range(test_no_of_trials):
        Test_X = features[trlIndex, :, :]
        # print(features) # TODO
        Test_Y_predicted = clf.predict(Test_X)

        lcrRes = classify_by_lcr(Test_Y_predicted)
        Pred_Y_LCR.append(lcrRes)

        modeRes = classify_by_mode(Test_Y_predicted)
        Pred_Y_Mode.append(modeRes)

    testAccLCR = accuracy_score(Test_Y, Pred_Y_LCR)
    testAccMode = accuracy_score(Test_Y, Pred_Y_Mode)
    # print(Pred_Y_LCR)
    # print(Pred_Y_Mode)
    # print(Test_Y)
    print("Subject " + training_subject)
    print("LCR acc", testAccLCR)
    print("Mode acc", testAccMode)

    mode_test_accuracies.append(testAccMode)
    LCR_test_accuracies.append(testAccLCR)

    '''
    import pickle

    outputFilename="trainedModel"+subID
    with open(outputFilename, 'wb') as f:
        #pickle.dump([search.best_estimator_, wCSP_mu, wCSP_beta, muBand, betaBand], f)
        pickle.dump([search, wCSP_mu, wCSP_beta, muBand, betaBand], f)
    '''

    train_eeg_data = train_eeg_data.tolist()

    print()

print("LCR", np.mean(LCR_test_accuracies))
print("Mode", np.mean(mode_test_accuracies))
