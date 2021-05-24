# MI_classification_using_SW

### Dataset

* A custom dataset recorded in University of essex was used for this project.
* The data set consists of 10 stroke patients left and right hand motor imagery data. Each patient has perfomed 40 trails each(20 left hand movement and 20 right hand movement) in a session and two sessions are conducted. so a total of 80 trails are presented for each patient and separate test session is conducted which consists of 20 trails.
* Each trail is of 8 secs each and timeing diagram of the each trail is as shown below.
![Timing Diagram](https://github.com/naveenkumarch/MI_classification_using_SW/blob/main/Results/Trail_timing_diagram.png?raw=true)

### Model 
* The Model uses common spatial pattern algorithm(CSP) for feature extraction from EEG signals and the extracted features are then feed to any type of classification algorithm for classificiation into either of the tasks. 
* In this project two classification algorithms are used SVC and LDA. The system architecture with SVC algorithm can be found below. 
[System Architecture](https://github.com/naveenkumarch/MI_classification_using_SW/blob/main/Results/System_diagram.png?raw=true)

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
