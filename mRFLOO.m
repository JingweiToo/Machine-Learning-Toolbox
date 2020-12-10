% Random Forest (10/12/2020)

function RF = mRFLOO(feat,label,num_tree)
% [Leave-one out]
fold     = cvpartition(label,'LeaveOut');
% Size of data
num_data = length(label); 
Afold    = zeros(num_data,1); 
pred2    = [];
ytest2   = [];

for i = 1:num_data
  % Call train & test data
  trainIdx = fold.training(i); testIdx = fold.test(i);
  xtrain   = feat(trainIdx,:); ytrain  = label(trainIdx);
  xtest    = feat(testIdx,:);  ytest   = label(testIdx); 
  % Train model
  Model = TreeBagger(num_tree,xtrain,ytrain,...
    'OOBPred','On',...
    'Method','Classification');
  pred = predict(Model,xtest);
  % Conversion
  num_test = size(pred,1);
  Z        = zeros(num_test,1);
  % Test
  for j = 1:num_test
    Z(j,1) = str2double(pred{j,1});
  end
  % Accuracy
  Afold(i) = sum(Z == ytest) / length(ytest);
  % Store temporary
  pred2  = [pred2(1:end); Z]; 
  ytest2 = [ytest2(1:end); ytest];
end
% Confusion matrix
confmat = confusionmat(ytest2,pred2); 
% Overall accuracy 
acc = mean(Afold);
% Store result
RF.acc  = acc;
RF.con  = confmat; 

fprintf('\n Accuracy (RF-LOO): %g %%',100 * acc); 
end

