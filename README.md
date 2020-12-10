# Simple Machine Learning Toolbox for Classification

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/f9d2bb8c-ebfe-4590-b88c-d4ff92fa6f8f/c4229dd2-aaa5-4146-bafa-4fcccb2b1d30/images/screenshot.PNG) 

* This toolbox contains 7 widely used machine learning algorithms   

* The < A_Main.m file > shows examples of how to use these machine learning methods with the benchmark dataset.  

## Output
* The displayed results include:  
  + Overall accuracy 
  + Confusion matrix
  + Computational time (s)

## Type of validation
There are three types of validation strategies:
  + Hold-out validation
```code 
opts.tf    = 1;
opts.ho    = 0.3;   % 30% data for testing 
```
  + *K*-fold cross-validation
```code 
opts.tf    = 2
opts.kfold = 10;    % 10-fold cross-validation
```
+ Leave-one-out validation
```code 
opts.tf    = 3 
```
  

### Example 1: *K*-nearest neighbor (KNN) with *k*-fold cross-validation
```code 
% Parameter settings
opts.tf    = 2;     
opts.kfold = 10;    
opts.k     = 3;     % k-value in KNN

% Load data
load iris.mat;

% Classification
KNN = jml('knn',feat,label,opts);

% Accuracy
disp(KNN.acc) 
% Confusion matrix
disp(KNN.con)

```

### Example 2: Multi-class support vector machine  (MSVM) with hold-out validation
```code 
% Parameter settings
opts.tf    = 1;     
opts.ho    = 0.3;       
opts.fun   = 'r';     % radial basis kernel function in SVM

% Load data
load iris.mat;

% Classification
MSVM = jml('msvm',feat,label,opts);

% Accuracy
disp(MSVM.acc) 
% Confusion matrix
disp(MSVM.con)

```

### Example 3: Decision Tree (DT) with leave-one-out validation
```code 
% Parameter settings
opts.tf     = 3;          
opts.nSplit = 50;    % number of split in DT 

% Load data
load iris.mat;

% Classification
DT = jml('dt',feat,label,opts);

% Accuracy
disp(DT.acc) 
% Confusion matrix
disp(DT.con)

```


## List of available machine learning methods

| No. | Abbreviation | Name                                   | 
|-----|--------------|----------------------------------------|
| 09  | 'gmm'        | Gaussian Mixture Model                 | 
| 08  | 'knn'        | *K*-nearest Neighbor (./Description.md)|
| 07  | 'msvm'       | Multi-class Support Vector Machine     |
| 06  | 'svm'        | Support Vector Machine                 |
| 05  | 'dt'         | Decision Tree                          |
| 04  | 'da'         | Discriminate Analysis Classifier       |
| 03  | 'nb'         | Naive Bayes                            |
| 02  | 'rf'         | Random Forest                          |
| 01  | 'et'         | Ensemble Tree                          |                    




