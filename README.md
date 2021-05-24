# MI_classification_using_SW

### Dataset

* A custom dataset recorded in University of essex was used for this project.
* The data set consists of 10 stroke patients left and right hand motor imagery data. Each patient has perfomed 40 trails each(20 left hand movement and 20 right hand movement) in a session and two sessions are conducted. so a total of 80 trails are presented for each patient and separate test session is conducted which consists of 20 trails.
* Each trail is of 8 secs each and timeing diagram of the each trail is as shown below.
![Timing Diagram](https://github.com/naveenkumarch/MI_classification_using_SW/blob/main/Results/Trail_timing_diagram.png?raw=true)

### Model 
* The Model uses common spatial pattern algorithm(CSP) for feature extraction from EEG signals and the extracted features are then feed to any type of classification algorithm for classificiation into either of the tasks. For the testing the data is converted into 10 sliding windows with a stride of 0.1 and LCA and Mode logics are used for final classification.
* In this project two classification algorithms are used SVC and LDA. The system architecture with SVC algorithm can be found below. 
![System Architecture](https://github.com/naveenkumarch/MI_classification_using_SW/blob/main/Results/System_diagram.png?raw=true)

## Requirments
  * python=3.7
  * numpy
  * pandas
  * tensorflow
  * seaborn
  * matplotlib
  * jupyter
  * scipy
  * scikit-learn
  * pickle5

### Implementation
* From the timing diagram its clear that user is not performing the said action through out the trail period and it has been observed that stroke patients are not able to concentrate on a specific task for more time.
* So analysis is perfomed to select time frames for each subject which contain lowest signal to noise ratio. For selecting the length of the timing window the average best accuracy of all subjects are considered.
* The best accuracies achieved by different length time windows are shown below.  
![time_length_difference](https://github.com/naveenkumarch/MI_classification_using_SW/blob/main/Results/Sliding_window_length_comp.png?raw=true)
* The no of sliding windows and stride between windows are also selected using similar approach
![no_of_windows_effect](https://github.com/naveenkumarch/MI_classification_using_SW/blob/main/Results/N0_windows_accuracy_effect.png?raw=true)
![Stride_effect](https://github.com/naveenkumarch/MI_classification_using_SW/blob/main/Results/Stride_effect_on_accuracy.png?raw=true)
