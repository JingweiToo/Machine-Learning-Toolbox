%-------------------------------------------------------------------------%
%  Machine learning algorithms source codes demo version                  %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

function RF=jRF(feat,label,nBag,kfold)
fold=cvpartition(label,'kfold',kfold); 
pred2=[]; ytest2=[]; Afold=zeros(kfold,1); 
for i=1:kfold
	trainIdx=fold.training(i); testIdx=fold.test(i);
  xtrain=feat(trainIdx,:); ytrain=label(trainIdx);
  xtest=feat(testIdx,:); ytest=label(testIdx);
  Model=TreeBagger(nBag,xtrain,ytrain,'OOBPred','On','Method','Classification');
  Pred0=predict(Model,xtest); A=size(Pred0,1); pred=zeros(A,1); 
  for j=1:A
  	pred(j,1)=str2double(Pred0{j,1});
  end
  con=confusionmat(ytest,pred);
  Afold(i)=100*sum(diag(con))/sum(con(:));  
  pred2=[pred2(1:end);pred]; ytest2=[ytest2(1:end);ytest];
end
confmat=confusionmat(ytest2,pred2); 
acc=mean(Afold);
RF.fold=Afold; RF.acc=acc; RF.con=confmat; 
fprintf('\n Classification Accuracy (RF): %g %%',acc); 
end

