% Random Forest (10/12/2020)

function RF = mRandomForest(feat,label,opts)
% Default
tf       = 2; 
num_tree = 20; 
kfold    = 10; 

if isfield(opts,'kfold'), kfold = opts.kfold; end
if isfield(opts,'ho'), ho = opts.ho; end
if isfield(opts,'nTree'), num_tree = opts.nTree; end 
if isfield(opts,'tf'), tf = opts.tf; end 

% Selection 
if tf == 1
  RF = mRFHO(feat,label,num_tree,ho);
elseif tf == 2
  RF = mRFCV(feat,label,num_tree,kfold);
elseif tf == 3
  RF = mRFLOO(feat,label,num_tree);
end
end

