%-------------------------------------------------------------------------%
%  Machine learning algorithms source codes demo version                  %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%


%---Input------------------------------------------------------------------
% feat:  feature vector (instances x features)
% label: labelling 
% kfold: Number of cross-validation
%---Output-----------------------------------------------------------------
% A struct that contains three results as follows:
% fold: Accuracy for each fold
% acc:  Average accuracy over k-folds
% con:  Confusion matrix
%--------------------------------------------------------------------------

%% Machine Learning
clc, clear
% Benchmark data set 
load iris.mat; 

% (1) Perform k-nearest neighbor (KNN)
kfold=10; k=3; % k-value in KNN
KNN=jKNN(feat,label,k,kfold); 

% (2) Perform discriminate analysis (DA)
kfold=10; Disc='l'; % The Discriminate can selected as follows:
% 'l' : linear 
% 'q' : quadratic
% 'pq': pseudoquadratic
% 'pl': pseudolinear
% 'dl': diaglinear
% 'dq': diagquadratic
DA=jDA(feat,label,Disc,kfold); 

% (3) Perform Naive Bayes (NB)
kfold=10; Dist='n'; % The Distribution can selected as follows:
% 'n' : normal distribution 
% 'k' : kernel distribution
NB=jNB(feat,label,Dist,kfold); 

% (4) Perform decision tree (DT)
kfold=10; nSplit=50; % Number of split in DT
DT=jDT(feat,label,nSplit,kfold);

% (5) Perform support vector machine (SVM with one versus one)
kfold=10; kernel='r'; % The Kernel can selected as follows:
% 'r' : radial basis function  
% 'l' : linear function 
% 'p' : polynomial function 
% 'g' : gaussian function
SVM=jSVM(feat,label,kernel,kfold);

% (6) Perform random forest (RF)
kfold=10; nBag=50; % Number of bags in RF
RF=jRF(feat,label,nBag,kfold);



