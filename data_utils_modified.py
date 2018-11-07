import numpy as np
import os
import glob
import random


def loadfromfolder():
    ROOT ='data/best'
# 	"Function goes through the data and returns file  names and parameters neeeded for the NN in the next step"
    names = []
    T= []
    Cov_dim = []
    for filename in os.listdir(ROOT):
        print(filename)  # name composition is like angles_cycles_frames_06t_8f_angles_22persons
        t =  int(filename[21:23])
        #print(T)
        mat_dim = int(filename[25:27])
        #print(mat_dim)
        T.append(t)
        Cov_dim.append(mat_dim)
        names.append(filename)
    return names, T, Cov_dim


def Selection_train_test(X_all, Persons, Y, Persons_test):
#    import random
    " This fucntion return train and set samples, taking the specified persons for testing"
    uniquePersons = np.unique(Persons)
    num_sampl, num_dim, _ = np.shape(X_all)
    X_train =np.array([], dtype=np.float).reshape(0,num_dim, num_dim)
    Y_train = np.array([], dtype = np.int64)
    X_test = np.array([], dtype=np.float).reshape(0,num_dim, num_dim)
    Y_test = np.array([], dtype = np.int64)



    print(X_train.shape)
    
    Persons_for_training = np.array([]) # for display
    Persons_for_testing = np.array([])
    ind2remove = Persons_test-1
    mask = np.ones(len(uniquePersons), dtype=bool) 
    mask[ind2remove] = False
    Persons_train = uniquePersons[mask] # actually, removes all test persons from the list
   
    train_N = Persons_train.size
   
    for Person in range(0,train_N):
        indexes = np.where(Persons ==Persons_train[Person])
        X_train= np.append(X_train, np.squeeze(X_all[[indexes], :,:]), axis=0)
        Y_train = np.append(Y_train,Y[indexes].astype(int))
        label =  np.full((indexes[0].size),Persons_train[Person])
        Persons_for_training = np.append(Persons_for_training, label, axis = 0)
    


    for Person in range(0,Persons_test.size):
        indexes = np.where(Persons ==Persons_test[Person])
        X_test = np.append(X_test, np.squeeze(X_all[[indexes], :,:]), axis=0)
        Y_test =  np.append(Y_test, Y[indexes].astype(int))
        label =  np.full((indexes[0].size),Persons_test[Person])
        Persons_for_testing = np.append(Persons_for_testing , label, axis = 0)
      
    print('Persons for train: ')
    print(np.unique(Persons_for_training))
    print('Persons for test: ')
    print(np.unique(Persons_for_testing))
    return X_train, Y_train, X_test, Y_test, Persons_for_training
