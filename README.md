This repository contains the code for the paper [On the Inference of Soft Biometrics from Typing Patterns Collected in a Multi-device Environment](https://arxiv.org/pdf/2006.09501.pdf) authored by [Vishaal Udandarao*](https://vishaal27.github.io/), [Mohit Agrawal*](https://sites.google.com/view/mohit-agrawal/home), [Rajesh Kumar](https://sites.google.com/view/kumar7) and [Rajiv Ratn Shah](https://www.iiitd.edu.in/~rajivratn/) (* indicates equal contribution). The code has been tested on Pytorch 1.3.1, Sklearn 0.22.2 and Python 3.6.8.

## Abstract
In this paper, we study the inference of gender, major/minor (computer science, non-computer science), typing style, age, and height from the typing patterns collected from 117 individuals in a multi-device environment. The inference of the first three identifiers was considered as classification tasks, while the rest as regression tasks. For classification tasks, we benchmark the performance of six classical machine learning (ML) and four deep learning (DL) classifiers. On the other hand, for regression tasks, we evaluated three ML and four DL-based regressors. We also present a novel method for the construction of feature space from keystrokes, which can be consumed directly by DL models. The overall experiment consisted of two text-entry (free and fixed) and four devices (Desktop, Tablet, Phone, and Combined) configurations. The best arrangements achieved accuracies of 96.15\%, 93.02\%, and 87.80\% for typing style, gender, and major/minor, respectively, and mean absolute errors of 1.77 years and 2.65 inches for age and height, respectively. The results are promising considering the variety of application scenarios that we have listed in this work.

## Getting Started 
Please install all the required dependendencies by running the following command:
```
pip install -r requirements.txt
```

## Experiments
All the experiments can be found in the `inferring_soft_biometrics_from_keystrokes_multi_device.ipynb` notebook. 

## Contact
If you face any problem in running this code, you can contact us at {vishaal16119, rajivratn}@iiitd.ac.in, mohit.nittrichy@gmail.com and rkumar@haverford.edu

## License
Copyright (c) 2020 Vishaal Udandarao, Mohit Agrawal, Rajesh Kumar, Rajiv Ratn Shah.

For license information, see LICENSE or http://mit-license.org
