function NB=jNB(feat,label,Dist,kfold)
switch Dist
	case'n'; Dist='normal';
	case'k'; Dist='kernel';
end
Model=fitcnb(feat,label,'Distribution',Dist);
C=crossval(Model,'kfold',kfold);
Pred=kfoldPredict(C); 
confmat=confusionmat(label,Pred); 
Afold=100*(1-kfoldLoss(C,'mode','individual')); 
acc=mean(Afold); 
NB.fold=Afold; NB.acc=acc; NB.con=confmat; 
fprintf('\n Classification Accuracy (NB): %g %%',acc); 
end

