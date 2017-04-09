num_folds = 5

# method 1, split X and Y
X_train_folds=np.array_split(X_train,num_folds)
y_train_folds=np.array_split(y_train,num_folds)

  for i in range(0,num_folds):        
        X_folds = list(X_train_folds)
        X_val  = X_folds.pop(i)
        X_folds = np.concatenate(X_folds)
        y_folds = list(y_train_folds)
        y_val  = y_folds.pop(i)
        y_folds = np.concatenate(y_folds)

# method 2, split index

mask=np.array_split(np.random.permutation(num_train),num_folds)

  for i in range(0,num_folds):        
        mask_train = list(mask)
        mask_val  = mask_train.pop(i)
        mask_train = np.concatenate(mask_train)        
        
        X_train[mask_train]
        y_train[mask_train]
        
        X_train[mask_val]
        y_train[mask_val]
