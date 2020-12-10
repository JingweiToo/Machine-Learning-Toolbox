% Machine Learning toolbox by Jingwei Too - 10/12/2020

function ML = jml(type,feat,label,opts)
switch type
  case 'gmm'     ; fun = @mGaussianMixtureModel; 
  case 'knn'     ; fun = @mKNearestNeighbor; 
  case 'da'      ; fun = @mDiscriminateAnalysis; 
  case 'nb'      ; fun = @mNaiveBayesECOC; 
  case 'msvm'    ; fun = @mMultiClassSupportVectorMachineECOC; 
  case 'svm'     ; fun = @mSupportVectorMachine; 
  case 'dt'      ; fun = @mDecisionTree; 
  case 'rf'      ; fun = @mRandomForest; 
  case 'et'      ; fun = @mEnsembleTree; 
end
tic; ML = fun(feat,label,opts); 
% Store 
time = toc; 
ML.t = time;

fprintf('\n Processing Time (s): %f % \n',time); fprintf('\n');
end



