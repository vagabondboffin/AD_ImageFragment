# Anomaly Detection in File Fragment Classification of Image File Formats
## A. FRAMEWORK 

File fragment classification is the first step to any application concerning file content analysis such as digital forensics, intrusion detection, web content filtering, and file carving. Throughout the years, numerous methodologies have been proposed for this problem. In the so-far applied methods, the main hypothesis is that all the fragments under examination belong to one of the training classes. Here, I have proposed a framework that can determine the non-image file fragment. Simply, an anomaly detector is added before the classifier. This module distinguishes images from other files and passes any normal (i.e. image) fragments to the classifier. 
The training phase of the proposed method is illustrated below. In the training phase, the first step is to train a model for anomaly detection. This model is trained in a semi-supervised manner using only normal data samples. As can be seen, before the data is ready for training, a scaling or feature selection step might be needed. The other step is to train a classifier with normal samples. An anomaly detection model and a classifier is the output of the training phase. Also, scaling and feature selection parameters are stored as a part of the model.
<figure>
  <img src="https://user-images.githubusercontent.com/52859501/206126242-245edb07-8772-4fe9-b3c9-517cb1f59a10.jpg" alt=".." title="Fig. 1" />
  <figcaption><sub>Fig. 1. The training phase of the proposed framework.</sub></figcaption>
</figure>



Fig. 2 shows the test phase of the proposed method. In the test phase, if a fragment is determined as normal, it is passed to the classifier. Otherwise, the fragment is labeled as an anomaly.
<figure>
  <img src="https://user-images.githubusercontent.com/52859501/206126435-16c8e9a2-1e8f-44ea-a686-cecc4933f096.jpg" alt=".." title="Fig. 2" />
  <figcaption><sub>Fig. 2. The proposed approach for image file format classification in presence of anomalous data.<sub/></figcaption>
</figure>

## B. EXPERIMENTS AND RESULTS
### Dataset
In this work, 10 different image formats with different compression settings are considered to form the normal class labels. The dataset presented in [[1]](#1) is used.
A complete set of features is considered with length 577. The feature extraction phase is performed using Fragments-Expert that is open-source software designed exclusively for the task of file fragment classification [[2]](#2). Selected features are content-based; they are computed from the bytes that form each fragment. 
### Methods
**Autoencoder**

I train a stacked autoencoder on the training dataset. Two hidden layers with 200 and 100 neurons and an output layer with five neurons are the overall architecture of the autoencoder. After training, I encoded the normal dataset with the already trained model and calculated the reconstruction errors. I used the mean of the calculated errors (reconstruction errors) as a threshold in the test phase. Mean squared error is used for calculating the errors.

**Simple Statistical**

By calculating the mean and standard deviation for each feature among all training samples and employing upper and lower limits for each feature, we can count how many features of a fragment exceed these boundaries.

**K-means**

With the assumption that in a feature space, anomalies are far from normal samples, distance-based approaches can be used for anomaly detection. Clustering methods fall into this category. In a semi-supervised approach, we train clusters on the normal dataset. Then, a threshold can be applied to the distance of a sample from the nearest cluster for determining whether it is a normal sample or not.

### Result
To focus on the evaluation of anomaly detection models, I considered a fixed classifier throughout our experiment. The classic random forest with 100 trees is used for classification. The classifier is trained on the training dataset with six normal labels. 
When there is no anomaly detector, the overall accuracy of the classifier among seven classes (six image classes and one anomaly class) is 72%. 

| Anomaly Detector | Precision | Recall | Accuracy |
| -------------- | :---------: | :----------: | :----------: |
| - | - | - | 72% |
| Autoencoder | 0.8 | 0.44 | 63% |
| Simple Statistical | 0.77 | 0.35 | 62% |
| K-means | 0.91 | 0.12 | 73% |

Reported results show that depending on the anomaly detection method, the overall accuracy of the classifier may decrease or increase. But what is significant is the trade-off between the true positive and the true negative rates. This fact points out the importance of choosing the right tool for anomaly detection.

## References
<a id="1">[1]</a> R. Fakouri and M. Teimouri, "Dataset for file fragment classification of image file formats," BMC Research Notes, vol. 12, p. 774, 2019/11/27 2019.

<a id="2">[2]</a> M. Teimouri, Z. Seyedghorban, and F. Amirjani, "Fragments-Expert: A graphical user interface MATLAB toolbox for classification of file fragments," Concurrency and Computation: Practice and Experience, vol. 33, p. e6154, 2021/05/10 2021.

<a id="3">[3]</a> Z. Seyedghorban and M. Teimouri, "Anomaly Detection in File Fragment Classification of Image File Formats," 2021 11th International Conference on Computer Engineering and Knowledge (ICCKE), 2021, pp. 248-253, doi: 10.1109/ICCKE54056.2021.9721457.

## Acknowledgement
All the functions that end with "_FFC" are built-in functions from [Fragments-Expert](https://github.com/vagabondboffin/Fragments-Expert). 

For a detailed explanation of the methods and parameters used in each please refer to [[3]](#3). 
