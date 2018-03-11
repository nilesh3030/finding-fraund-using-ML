# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 22:02:37 2017

@author: NILESH
"""
# In this user_function , I have made few functions which will be used in my poi_id code

import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif

# global fumction for making scatter plots 
def scatter_plot(data,x,y,x_lab,y_lab):
    for point in data:
        X = point[x]
        Y = point[y]
        plt.scatter( X, Y )
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()
    
#function for replacing NaN values with 0
def replaceNaN(value):
    if value =='NaN':
        value = 0
    return value

#function for Selecting k best features for optimal performance
f_scores =[]
def select_features(features,labels,features_list,k) :
    clf = SelectKBest(f_classif,k)
    selected_feature = clf.fit_transform(features,labels)
    feature_l=[features_list[i+1] for i in clf.get_support(indices=True)]
    f_scores = zip(features_list[1:],clf.scores_[:])
    f_scores = sorted(f_scores,key=lambda x: x[1],reverse=True)
    return selected_feature, ['poi'] + feature_l, f_scores