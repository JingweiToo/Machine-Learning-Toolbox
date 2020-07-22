%-------------------------------------------------------------------------%
%  Machine learning algorithms source codes demo version                  %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

function KNN=jKNN(feat,label,k,kfold)
Model=fitcknn(feat,label,'NumNeighbors',k,'Distance','euclidean');
C=crossval(Model,'KFold',kfold);
Pred=kfoldPredict(C); 
confmat=confusionmat(label,Pred);
Afold=100*(1-kfoldLoss(C,'mode','individual'));
acc=mean(Afold); 
KNN.fold=Afold; KNN.acc=acc; KNN.con=confmat; 
fprintf('\n Classification Accuracy (KNN): %g %%',acc); 
end


