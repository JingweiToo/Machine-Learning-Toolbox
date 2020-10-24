function SVM=jSVM(feat,label,kernel,kfold)
switch kernel
	case'l'; kernel='linear';     
  case'r'; kernel='rbf'; 
  case'p'; kernel='polynomial';   
  case'g'; kernel='gaussian';
end
Temp=templateSVM('KernelFunction',kernel,'KernelScale','auto');
Model=fitcecoc(feat,label,'Learners',Temp);
C=crossval(Model,'kfold',kfold);
pred=kfoldPredict(C);
confmat=confusionmat(label,pred); 
Afold=100*(1-kfoldLoss(C,'mode','individual')); 
acc=mean(Afold); 
SVM.fold=Afold; SVM.acc=acc; SVM.con=confmat; 
fprintf('\n Classification Accuracy (SVM): %g %%',acc); 
end

