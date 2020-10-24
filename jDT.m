function DT=jDT(feat,label,nSplit,kfold)
Model=fitctree(feat,label,'MaxNumSplits',nSplit);
C=crossval(Model,'KFold',kfold); 
Pred=kfoldPredict(C);
confmat=confusionmat(label,Pred);
Afold=100*(1-kfoldLoss(C,'mode','individual')); 
acc=mean(Afold); 
DT.fold=Afold; DT.acc=acc; DT.con=confmat; 
fprintf('\n Classification Accuracy (DT): %g %%',acc);
end

