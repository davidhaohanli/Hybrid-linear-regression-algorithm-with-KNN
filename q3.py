import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50;

k=500;

feature_names=['1 (bias)','CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',\
               'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B' ,'LSTAT']

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1) #add constant one feature - no bias needed
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w, features


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    l_gradient=np.zeros((X.shape[0],X.shape[1]));
    for i in range(X.shape[0]):
        l_gradient[i]=-2*(y[i]*(X[i])+w.T.dot(X[i].T)*X[i]);
    return l_gradient.mean(axis=0);

    #raise NotImplementedError()
def batch_gradient(w,d,batch_sampler,m=None):
    ##########mean batch_gradient for 500 times of randomly selected batch###########
    batch_grads=np.zeros((k,d));

    # Example usage
    for i in range(k):
        X_b, y_b = batch_sampler.get_batch(m);
        batch_grads[i] = lin_reg_gradient(X_b, y_b, w);

    return batch_grads.mean(axis=0),batch_grads.var(axis=0);
    #print(batch_grad)


def visualize(X, y,features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plt.plot(y,X[:, i],'.')
        plt.xlabel(features[i])
        # TODO: Plot feature i against y

    plt.tight_layout()
    plt.show()

def main():
    # Load data and randomly initialise weights
    X, y, w, d = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)


    batch_grad,_=batch_gradient(w,d,batch_sampler)
    ###########true gradient##############
    true_grad=lin_reg_gradient(X,y,w);


    ###########Square Distance Metric##########

    print ('Square Distance Metric: ',np.sqrt(((batch_grad-true_grad)**2).sum()))

    ###########Cosine Similarity##########

    print ('Cosine Similarity: ',cosine_similarity(batch_grad,true_grad))

    ###########Q6: for m= 1 to 400, plot sigma(j) vs. m##########
    var=np.zeros((400,d));
    m=np.arange(400)
    for i in m:
        batch_sampler = BatchSampler(X, y, i+1)
        _,var[i] = batch_gradient(w, d, batch_sampler)

    visualize(np.log(var),np.log(m+1),feature_names);



if __name__ == '__main__':
    main()