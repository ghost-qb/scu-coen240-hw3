"""
Santa Clara University

COEN 240 - Machine Learning

HW3

Quan Bach


By running thi file:

"""
# for randoming index
import math
import random

# to 80/20 divide the dataset
import numpy as np

# for trainning model with k-NNs 
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

# for plotting 
import matplotlib.pyplot as plt 
import seaborn as sns



if __name__ == '__main__':
    
    # load data from sklearn 
    dataset = load_wine()
    print('...')
    print('Data wine loaded from sklearn...')
    
    # given k-set for k-NNs         
    k_set = {1,3,5,8,15,20}
    
    stat_dict = {'num_of_classes':len(dataset.target_names), 'num_of_samples':len(dataset.data), 'num_of_features':len(dataset.data[1])}
    #display statistics 
    print('===== DATASET STATISTICS =====')
    for key, value in stat_dict.items():
        print('{} : {}'.format(key,value))
    print('...')
    # random 20% of the dataset.data to form the validation set 
    idx = set()
    # compute the length of 20% of the dataset.data
    validation_set_len = math.ceil(0.2*len(dataset.target))
    
    
    # add random index from 0 to lenght of the dataset and add to the idx set 
    while (len(idx) < validation_set_len):
        rand_idx = random.randint(0,len(dataset.target)-1)
        idx.add(rand_idx)
        
    
    # initialize the custom datasets 
    X_validation = []
    y_validation = []
    
    X_train = dataset.data
    y_train = dataset.target
    
    
    # form the validation and training sets
    for i in sorted(idx,reverse=True):
        # add elements to the validation sets
        X_validation.append(dataset.data[i])
        y_validation.append(dataset.target[i])
        
        # trimming the orignal dataset to only contains the correct 80%
        X_train = np.delete(X_train,i,0)
        y_train = np.delete(y_train,i)
     
    print('Partitioned data into 80/20...')
    print('...')
    # implement KNNs 
    for k in k_set:
        clf_knn = KNeighborsClassifier(n_neighbors=k)
        
        clf_knn.fit(X_train,y_train)
        
        # predict from validation set 
        y_predict = clf_knn.predict(X_validation)
        print('\n'*2)
        print(' **************** FOR K =' + str(k) + ' **********************')
        # Print evalulation 
        print()
        print('----------Performance metrics----------')
        print(metrics.classification_report(y_validation, y_predict, target_names=dataset.target_names))
        print ()
        print('----------Confusion Matrix----------')
        print(metrics.confusion_matrix(y_validation,y_predict))
        print()
        print ('NMI Score: ', nmi(y_validation,y_predict))
        print('Classification Error of KNN: ', 1 - metrics.accuracy_score(y_validation,y_predict))
        print('Sensitive of KNN: ', metrics.recall_score(y_validation,y_predict, average='weighted'))
        print('Precision of KNN: ', metrics.precision_score(y_validation,y_predict,average='weighted'))
        print('F-measure of KNN: ', metrics.f1_score(y_validation,y_predict,average='weighted'))    
        
        plot_title = 'Confusion matrix for KNNs k= ' + str(k)
        f, ax = plt.subplots(figsize=(16, 12))
        
        plt.title(plot_title)
        ax = sns.heatmap(metrics.confusion_matrix(y_validation,y_predict), annot=True, cmap='Blues',annot_kws={"size":15})
        
        figname = 'knn_confusion_matrix_k=' + str(k) +'.png'
        plt.savefig(figname)
        plt.show()