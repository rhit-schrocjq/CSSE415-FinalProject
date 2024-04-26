def DecisionBoundary(clf,feature_names,target_name,df):
    
    from sklearn.preprocessing import StandardScaler,LabelEncoder
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    N     = 250   # N x N number of points for 2D grid generation
    alpha = 0.2   # point transparency
    pt_size = 200
    sns.set_palette('bright')
    cmap = sns.color_palette('bright')

    # create 2D grid of points in feature space
    X1 = df[feature_names[0]]
    X2 = df[feature_names[1]]
    XX1,XX2 = np.meshgrid(
    np.linspace(X1.min(),X1.max(),N),
    np.linspace(X2.min(),X2.max(),N))
    
    # integer code categories
    df['target_code'] = LabelEncoder().fit_transform(df[target_name])
    
    # shuffle and stratified-split data
    (df_train,df_test) = train_test_split(df,
                                          train_size=0.8,
                                          test_size=0.2,
                                          shuffle=True,
                                          stratify=df[target_name],
                                          random_state=0)
    
    # assign features and targets
    features_train = df_train[feature_names]
    features_test  = df_test[feature_names]
    target_train   = df_train['target_code']
    target_test    = df_test['target_code']
    
    # standardize
    stnd = StandardScaler()
    stnd.fit(features_train)
    features_train = stnd.transform(features_train)
    features_test  = stnd.transform(features_test)
    
    # fit classifier
    clf.fit(features_train,target_train)
    
    # predict targets on 2D grid
    features_grid = np.stack([XX1.ravel(),XX2.ravel()],axis=1)
    features_grid = stnd.transform(features_grid)
    Y = clf.predict(features_grid)
    YY = Y.reshape(XX1.shape)
    
    
    # plot train set
    fig1,ax1 = plt.subplots(figsize=(10,10))
    sns.scatterplot(x=feature_names[0],
                    y=feature_names[1],
                    hue=target_name,
                    data=df_train,
                    s=pt_size,
                    ax=ax1)
    plt.contourf(XX1,XX2,YY,alpha=alpha,colors=cmap)
    plt.title('train set')
    
    # plot test set
    fig2,ax2 = plt.subplots(figsize=(10,10))
    sns.scatterplot(x=feature_names[0],
                          y=feature_names[1],
                          hue=target_name,
                          data=df_test,
                          s=pt_size,
                          ax=ax2)
    plt.contourf(XX1,XX2,YY,alpha=alpha,colors=cmap)
    plt.title('test set')
    
    acc_train = clf.score(features_train,target_train)
    acc_test  = clf.score(features_test,target_test)
    
    return acc_train,acc_test