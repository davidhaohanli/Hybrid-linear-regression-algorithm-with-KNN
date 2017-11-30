# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']
idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions-y_test)**2).mean()
    return losses

#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    #print(x_train.shape, y_train.shape,test_datum.shape,tau)
    l2_mod = -1*l2(test_datum,x_train)/(2*tau**2);
    B = max(l2_mod[0])
    expStable=np.exp(l2_mod[0]-B)
    A = np.diag(expStable/np.sum(expStable));
    w_star = np.linalg.solve(np.dot(np.dot(x_train.T,A),x_train)+np.diag([lam]*x_train.shape[1]),\
                             np.dot(np.dot(x_train.T,A),y_train));
    ## TODO
    return test_datum[0].dot(w_star);

def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO

    pieceLen = N // k;
    losses = np.zeros((1,taus.shape[0]));
    for i in range(k):
        piece = np.append(idx[0 : pieceLen*i],idx[pieceLen*(i+1) : -1])
        x_train = x[piece,:];
        y_train = y[piece].reshape(pieceLen*(k-1),1);
        x_test = x[idx[pieceLen*i : pieceLen*(i+1)],:];
        y_test = y[idx[pieceLen*i : pieceLen*(i+1)]].reshape(pieceLen,1);
        losses=np.append(losses,[run_on_fold(x_test,y_test,x_train,y_train,taus)],axis=0)
    losses=np.delete(losses,0,0)

    ## TODO
    return np.mean(losses, axis=0);




########################Test########################
def test(name):
    if name == 'lrls':
        print ('Single test data: ', LRLS_Test(10));
    if name == 'onFold':
        print ('One fold of test data: ', onFold_Test(taus = np.logspace(1,3,200)));
    if name == 'kFold':
        print ('K times fold of test data: ', kFold_Test(taus = np.logspace(1,3,200)));
####################################################

#####################Test_Sets######################
def LRLS_Test(tau):
    return LRLS(x[0].reshape(1,x[0].shape[0]),x,y.reshape(y.shape[0],1),tau);

def onFold_Test(taus):
    return run_on_fold(x[0:100,:],y.reshape(N,1)[0:100,:],x[101:-1,:]\
                      ,y.reshape(N,1)[101:-1,:],taus)

def kFold_Test(taus):
    return run_k_fold(x,y,taus,k=5)
####################################################

########################main########################
def main():
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1,3,200);
    #print (taus)
    losses = run_k_fold(x,y,taus,k=5)
    plt.figure(figsize=(20, 5))
    plt.plot(taus,losses)
    plt.xlabel('taus')
    plt.ylabel('losses')
    plt.show()
    print("min loss = {}".format(losses.min()))
####################################################

if __name__ == "__main__":
    main();
