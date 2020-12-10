% Machine Learning Toolbox by Jingwei Too - 10/12/2020   


%---Input-------------------------------------------------------------
% feat       : feature vector matrix (Instances x Features)
% label      : label matrix (Instances x 1)
% opts       : parameter settings 
% opts.tf    : choose either hold-out / k-fold / leave-one-out
% opts.ho    : ratio of testing data in hold-out validation
% opts.kfold : number of folds in k-fold cross-validation


%---Output-------------------------------------------------------------
% ML      : machine learning model (It contains several results)
% ML.acc  : classification accuracy 
% ML.con  : confusion matrix
% ML.t    : computational time (s)
%----------------------------------------------------------------------


%% Example 1: K-nearest neighbor (KNN) with k-fold cross-validation
% Parameter settings
opts.tf    = 2;     
opts.kfold = 10;    
opts.k     = 5;     % k-value in KNN
% Load data
load iris.mat; 
% Classification
ML = jml('knn',feat,label,opts);
% Accuracy
accuracy = ML.acc; 
% Confusion matrix
confmat  = ML.con;


%% Example 2: Multi-class support vector machine (MSVM) with hold-out validation
% Parameter settings
opts.tf    = 1;     
opts.ho    = 0.3;       
opts.fun   = 'r';     % radial basis kernel function in SVM
% Load data
load iris.mat;
% Classification
ML = jml('msvm',feat,label,opts);
% Accuracy
accuracy = ML.acc; 
% Confusion matrix
confmat  = ML.con;


%% Example 3: Decision Tree (DT) with leave-one-out validation
% Parameter settings
opts.tf     = 3;          
opts.nSplit = 50;    % number of split in DT 
% Load data
load iris.mat;
% Classification
ML = jml('dt',feat,label,opts);
% Accuracy
accuracy = ML.acc; 
% Confusion matrix
confmat  = ML.con;


