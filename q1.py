from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features

def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.plot(X[:,i], y,'.')
        plt.xlabel(features[i])
        #plt.ylabel('target y')
        #TODO: Plot feature i against y
    
    plt.tight_layout()
    plt.show()

def fit_regression(X,Y):

    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    X=np.column_stack((np.ones(X.shape[0]).reshape(X.shape[0],1),X))
    return np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))

    #raise NotImplementedError()

def split(X,y):
    index=np.random.choice(X.shape[0],X.shape[0]*8//10,replace=False);
    index.sort();
    Xtrain=X[0,:];
    Xtest=X[0,:]
    Ytrain=y[index[0]];
    Ytest=y[index[0]];
    for i in range(X.shape[0]):
        if i in index:
            Xtrain=np.row_stack((Xtrain,X[i]))
            Ytrain=np.row_stack((Ytrain,y[i]))
        else:
            Xtest=np.row_stack((Xtest,X[i]))
            Ytest=np.row_stack((Ytest,y[i]))
    return np.delete(Xtest,0,0),np.delete(Xtrain,0,0),np.delete(Ytest,0,0),np.delete(Ytrain,0,0);

def tabulate(features,w):
    hashmap={'1':'w[0]'};
    print ('1 :',w[0]);
    for i,item in enumerate(features):
        hashmap[item]=w[i+1];
        print (item,':',w[i+1]);
    return hashmap

def errors(x,y,w):
    x = np.column_stack((np.ones(x.shape[0]).reshape(x.shape[0],1), x));
    MSE=np.mean((x.dot(w)-y) ** 2)
    return MSE,np.mean(abs(x.dot(w)-y)),np.sqrt(MSE);

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    Xshape = X.shape;
    yshape = y.shape;

    print ('Feature Size: ' , Xshape[0],'       Feature dimension: ',Xshape[1]) #>> (506,13)
    print ('y Size: ', yshape[0], '        y dimension:' ,1)

    # Visualize the features
    visualize(X, y, features)
    #TODO: Split data into train and test
    X_test,X_train,y_test,y_train=split(X,y);
    #print (train.shape,test.shape)

    # Fit regression model
    w = fit_regression(X_train,y_train);
    #print(w)

    hashmap=tabulate(features,w);

    # Compute fitted values, MSE, etc.
    MSE,MAE,RMSE=errors(X_test,y_test,w);
    print('\nMean Square Error: ',MSE,'\nMean Absolute Error: ',MAE,\
          '\nRoot Mean Square Error: ',RMSE);


if __name__ == "__main__":
    main()
