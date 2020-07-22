%-------------------------------------------------------------------------%
%  Machine learning algorithms source codes demo version                  %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

function DA=jDA(feat,label,Disc,kfold)
switch Disc
  case'l' ; Disc='linear'; 
  case'pq'; Disc='pseudoquadratic';
  case'q' ; Disc='quadratic';    
  case'dl'; Disc='diaglinear';
  case'pl'; Disc='pseudolinear'; 
  case'dq'; Disc='diagquadratic';
end
Model=fitcdiscr(feat,label,'DiscrimType',Disc); 
C=crossval(Model,'kfold',kfold);
pred=kfoldPredict(C); 
confmat=confusionmat(label,pred); 
Afold=100*(1-kfoldLoss(C,'mode','individual'));
acc=mean(Afold); 
DA.fold=Afold; DA.acc=acc; DA.con=confmat;
fprintf('\n Classification Accuracy (DA): %g %%',acc);
end



