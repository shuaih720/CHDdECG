# CHDdECG
Scripts and modules for training and testing deep neural networks that conducts congenital heart disease detection using electrocardiogram (ECG).
Companion code to the paper "Congenital Heart Disease Detection by Pediatric Electrocardiogram Based Deep Learning Integrated with Human Concepts".

<!-- https://www.xxxxxxxxx.com/

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
-------------------- -->

## Abstract information

Early detection is critical to achieving improved treatment outcomes for child patients with congenital heart diseases (CHDs). Therefore, developing effective CHD detection techniques using low-cost and non-invasive pediatric electrocardiogram are highly desirable. We propose a deep learning approach for CHD detection, CHDdECG, which automatically extracts features from pediatric electrocardiogram signals and wavelet transformation characteristics, integrating them with human concept features. Trained with 65,869 cases, CHDdECG achieved ROC-AUC of 0.915 and 0.917 on real-world and external test sets containing 12,000 and 7,137 cases, respectively. Overall specificities were 0.881 and 0.937, respectively, with major CHD subtype specificities above 0.91. Notably, CHDdECG outperformed cardiologists in comparison, and the automatically extracted features from electrocardiogram were about 8 times more significant than human concept features, implying that CHDdECG find some knowledge beyond human cognition. Our study directly impacts CHD detection with pediatric electrocardiogram and highlights the potential of pediatric electrocardiogram for broader benefits.

--------------------
## Requirements

This code was tested on Python 3 with Tensorflow `2.6.0`

In addition, the packages we are calling now is as follows:
- [x] tensorflow2.0     
- [x] sklearn
- [x] random
- [x] scipy
- [x] pandas
- [x] numpy
- [x] tabnet
- [x] tensorflow_addons  

## Framework illustration

- **input**: `shape = (N, 5000, 9)`. The input tensor, a signal of 10 seconds should contain the 5000 points of the ECG tracings sampled at 500Hz both in the training and in the test set. The last dimension of the tensor contains points of the 9 different leads. The leads are ordered in the following order: `{I, II, III, AVR, AVL, AVF, V1, V3, V5}`. All signal are preprocessed with noise removal techniques before feeding it to the neural network model. 
![example](https://github.com/shuaih720/CHDdECG/blob/main/Figures/ECG%20example.png)
- **framework illustration**: ``CHDdECG.py``: Auxiliary module that defines the architecture of the deep neural network. The internal module structure is in the following files：``layers_Resblock.py``,``layers_TemporalAttention.py``,``layers_Transformer.py``,``layers_Tabnet.py``.
![example1](https://github.com/shuaih720/CHDdECG/blob/main/Figures/An%20illustration%20of%20the%20deep%20learning%20based%20model.png)
- **train and test**: ``main.py``:Script for training the neural network and generating the neural network predictions on a given dataset.
- **output**: `shape = (N, 2)`. Each entry contains a probability between 0 and 1, and can be understood as the probability of a given abnormality to be present.

## Install from Github
```python
python
>>> git clone https://github.com/shuaih720/CHDdECG
>>> cd CHDdECG
>>> python setup.py install
```
(Typical install time on a "normal" desktop computer: very variable)

## Instructions for use
```python
python
>>> cd CHDdECG
>>> python import.py
>>> python layers_Tabnet.py
>>> python layers_Resblock.py
>>> python layers_TemporalAttention.py
>>> python layers_Transformer.py
>>> python CHDdECG.py
>>> python main.py
```
OR running the integrated version 
```python
python
>>> cd CHDdECG
>>> python Merged_CHD.py
```
Training the neural network and generating the neural network predictions on given datasets.
## License

This project is covered under the Apache 2.0 License.
