#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import normalize, StandardScaler


# ## Parameter

# In[2]:


imsizesq = 28
num_classes = 10
batch = 100
num_epochs = 5
learning_rate = 0.001
k_folds = 10 # 90/10
#k_folds = 5 # 80/20


# ## Data preproccessing

# In[3]:


X_train, y_train = loadlocal_mnist(images_path='./data/train-images-idx3-ubyte', labels_path='./data/train-labels-idx1-ubyte')
X_test, y_test = loadlocal_mnist(images_path='./data/t10k-images-idx3-ubyte', labels_path='./data/t10k-labels-idx1-ubyte')
X_train = X_train.astype('float32')
X_train = normalize(X_train)
X_test = X_test.astype('float32')    
X_test = normalize(X_test)


# In[4]:


X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


# ### split training and testing data

# In[12]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42, shuffle = True)


# In[6]:


kfold = KFold(n_splits=k_folds, shuffle=True)


# In[13]:


tuned_parameters = [{'kernel': ['rbf'],  'C': [0.001, 0.10, 0.1, 1, 10]},
                    {'kernel': ['sigmoid'], 'C': [0.001, 0.10, 0.1, 10]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10]}]

for train_index, test_index in kfold.split(x_train):
    print("TRAIN:", train_index, "TEST:", test_index)
    xtrain, xtest = x_train[train_index], x_test[test_index]
    ytrain, ytest = y_train[train_index], y_test[test_index]
    
    clf = GridSearchCV(SVC(), tuned_parameters, cv = 3, scoring= 'accuracy', n_jobs=6, verbose=2)


    clf.fit(xtrain, ytrain.ravel())


    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    y_true, y_pred = ytest, clf.predict(xtest)
    print(classification_report(y_true, y_pred))


# ## Define k-fold




