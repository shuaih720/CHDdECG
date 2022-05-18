# CHDdECG
Scripts and modules for training and testing deep neural networks for  automatic classification of pediatric Congenital Heart Diseases(CHDs) using Electrocardiogram(ECG).
Companion code to the paper "Automatic Diagnosis of Congenital Heart Diseases from 9-Lead Pediatric Electrocardiogram Using Deep Neural Networks".
https://www.xxxxxxxxx.com/

--------------------

Citation:
```
Authors et al. Automatic Diagnosis of Congenital Heart Diseases from 9-Lead Pediatric Electrocardiogram Using Deep Neural Networks.
journal. doi
```

Bibtex:
```
@article{,
  title = {Automatic Diagnosis of Congenital Heart Diseases from 9-Lead Pediatric Electrocardiogram Using Deep Neural Networks},
  author = {},
  year = {},
  volume = {},
  pages = {},
  doi = {},
  journal = {},
  number = {}
}
```
--------------------

## Abstract information
  Early detection and timely intervention are prerequisites for improved treatment outcomes of pediatric patients with congenital
heart diseases (CHDs). Nowadays, transthoracic echocardiogram (TTE) is considered the standard of care for CHD detection
in developed countries. However, a large number of CHD patients still do not have access to timely detection techniques,
especially in low-resource countries and regions. Since electrocardiogram (ECG) is a commonly-used tool worldwide and
previous studies have found correlations between ECG manifestations and CHDs, it is highly desirable to devise an effective
ECG-based differential diagnosis approach for CHD. To achieve this and to facilitate timely CHD detection and intervention, we
developed a new deep learning based approach, Congenital Heart Disease diagnosis via Electrocardiogram (CHDdECG),
which provided differential diagnosis for pediatric CHDs and their subtypes using routinely acquired ECG examination. In
particular, CHDdECG automatically extracted clinically useful features from raw ECG signals that might not be recognized by
human experts. During processing, these extracted features were integrated with expert knowledge as well as features obtained
from wavelet transformation in order to provide a robust, scalable, and accurate analysis. In total, 9-lead ECG-waveform
data collected from 85,006 pediatric patients presenting to two major referral centers were analyzed to train and validate the
CHDdECG model that identified the status as CHD, including its various subtypes. On the test set and the out-of-distribution
external test set, CHDdECG achieved good performances, with overall ROC-AUC scores around 0.91 and median confidences
over 0.80. We also found that features automatically extracted from ECG-waveform data by CHDdECG were critical, which
contributed to higher CHD detection sensitivity compared to cardiologists. We further employed the Grad-CAM approach to
identify specific ECG manifestations associated with CHD subtypes, and obtained results consistent with previous observations
on adult ECG data. Our study demonstrated that the deep learning based CHDdECG approach could be used as an assistive
tool in conjunction with or in lieu of comprehensive diagnostic work-ups for pediatric CHD screening and early detection. The
impact of this study is directly on pediatric CHD diagnosis with ECG data, yet the benefits of exploring the potentials of ECG
data are of general significance.

--------------------
## Requirements

This code was tested on Python 3 with Tensorflow `2.6.0`

## Framework illustration

- **input**: `shape = (N, 5000, 9)`. The input tensor, a signal of 10 seconds should contain the 5000 points of the ECG tracings sampled at 500Hz both in the training and in the test set. The last dimension of the tensor contains points of the 9 different leads. The leads are ordered in the following order: `{I, II, III, AVR, AVL, AVF, V1, V3, V5}`. All signal are preprocessed with noise removal techniques before feeding it to the neural network model. 
![example](https://github.com/shuaih720/CHDdECG/blob/main/Figures/ECG%20example.png)
- **framework illustration**: ``CHDdECG.py``: Auxiliary module that defines the architecture of the deep neural network. The internal module structure is in the following files：``layers_Resblock.py``,``layers_TemporalAttention.py``,``layers_Transformer.py``.
![example1](https://github.com/shuaih720/CHDdECG/blob/main/Figures/An%20illustration%20of%20the%20deep%20learning%20based%20model.png)
- **train and test**: ``main.py``:Script for training the neural network and generating the neural network predictions on a given dataset.
- **output**: `shape = (N, 2)`. Each entry contains a probability between 0 and 1, and can be understood as the probability of a given abnormality to be present.
