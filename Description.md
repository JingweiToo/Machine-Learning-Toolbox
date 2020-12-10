# Detail Parameter Settings / Default Setting

## *K*-nearest Neighbor (KNN) 
* Number of k-value
```code 
opts.k = 5;    
```

## Support Vector Machine (SVM)
* Selection of kernel function. You may choose one
```code 
opts.fun = 'r';    % radial basis function
opts.fun = 'g';    % gaussian 
opts.fun = 'l';    % linear 
opts.fun = 'p';    % polynomial
```

## Decision Tree (DT)
* Number of splitting
```code 
opts.nSplit = 50;   
```

## Random Forest (RF)
* Number of trees
```code 
opts.nTree = 20;   
```

## Ensemble Tree (ET)
* Number of splitting
```code 
opts.nSplit = 50;   
```

## Discriminate Analysis (DA)
* Selection of discriminate function. You may choose one
```code 
opts.fun = 'l';     % linear
opts.fun = 'pq';    % pseudoquadratic
opts.fun = 'q';     % quadratic    
opts.fun = 'dl';    % diaglinear
opts.fun = 'pl';    % pseudolinear 
opts.fun = 'dq';    % diagquadratic
```

## Discriminate Analysis (DA)
* Selection of discriminate function. You may choose one
```code 
opts.fun = 'l';     % linear
opts.fun = 'pq';    % pseudoquadratic
opts.fun = 'q';     % quadratic    
opts.fun = 'dl';    % diaglinear
opts.fun = 'pl';    % pseudolinear 
opts.fun = 'dq';    % diagquadratic
```

## Naive Bayes (NB)
* Selection of distribution. You may choose one
```code 
opts.fun = 'n';     % normal 
opts.fun = 'k';     % kernel 
opts.fun = 'm';     % multinomial    
opts.fun = 'mm';    % multivariate multinomial
```



