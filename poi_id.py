
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from tester import dump_classifier_and_data, test_classifier
from time import time
from sklearn.metrics import accuracy_score,recall_score,precision_score
from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.feature_selection import SelectKBest,f_classif
import user_function as fn

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments', 
                'loan_advances', 'bonus', 'restricted_stock_deferred', 
                'deferred_income', 'total_stock_value', 'expenses', 
                'exercised_stock_options', 'other', 'long_term_incentive', 
                'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
#there are so many features available and firslty we will make explore these features below
print "Number of data points :", len(data_dict)
#Number of features
lengths = [len(v) for v in data_dict.values()]
print "Number of features for each data point:", len(data_dict.values()[0])

#Number of records which are POIs and non POIs before removing outliers
poi_counter = 0
for i in data_dict.values():
    if i['poi']!= 0:
        poi_counter = poi_counter+1
print "Number of POIs before removing outliers :", poi_counter
print "Number of non POIs before removing outliers :", len(data_dict)-poi_counter

### Task 2: Remove outliers
# I have listed ouliers below which were identified during my analysis

identified_outliers = ["TOTAL", "LAVORATO JOHN J", "MARTIN AMANDA K", "URQUHART JOHN A", 
                       "MCCLELLAN GEORGE", "SHANKMAN JEFFREY A", "WHITE JR THOMAS E", 
                       "PAI LOU L","THE TRAVEL AGENCY IN THE PARK"]

for outlier in identified_outliers:
    data_dict.pop(outlier)
data = featureFormat(data_dict, features_list, sort_keys = True)

#Number of records which are POIs and non POIs after removing outliers
poi_counter = 0
for i in data_dict.values():
    if i['poi']!= 0:
        poi_counter = poi_counter+1
print "Number of POIs after removing outliers:", poi_counter
print "Number of non POIs after removing outliers:", len(data_dict)-poi_counter

# Total payments versus salary
salary = features_list.index('salary')
total_payments = features_list.index('total_payments')
fn.scatter_plot(data,salary,total_payments,features_list[1],features_list[3])  

for i in data_dict:
    if data_dict[i]['total_payments'] > 50000000 and data_dict[i]['salary']>800000:
        if data_dict[i]['total_payments'] != 'NaN' and  data_dict[i]['salary'] != 'NaN':
            print "People with high total payments and high salary:",i,data_dict[i]['total_payments']    
#in this case we see LAY KENNETH L as an outlier and there will be many more people above this mentioned range
#beacuse they are at top positions so these type of outliers can be handled by limitting the dataset

# salary versus bonus
salary = features_list.index('salary')
bonus = features_list.index('bonus')
fn.scatter_plot(data,salary,bonus,features_list[1],features_list[5])

### Task 3: Create new feature(s)
#i'm creating POI_interaction and email_interaction which respectively gives an idea about 
#the involvement with POIs
for key in data_dict:
    from_this_person_to_poi  = fn.replaceNaN(data_dict[key]['from_this_person_to_poi'])
    shared_receipt_with_poi = fn.replaceNaN(data_dict[key]['shared_receipt_with_poi'])
    from_poi_to_this_person = fn.replaceNaN(data_dict[key]['from_poi_to_this_person'])
    POI_interaction = from_this_person_to_poi + shared_receipt_with_poi + from_poi_to_this_person
                        
    to_messages = fn.replaceNaN(data_dict[key]['to_messages'])
    from_messages = fn.replaceNaN(data_dict[key]['from_messages'])

    Mail_interaction = to_messages + from_messages 
    data_dict[key]["Mail_interaction"] = Mail_interaction 
    if  (POI_interaction ==0) &  (Mail_interaction ==0) :
        data_dict[key]["fraction_POI_interaction"]= 0
    else :
        data_dict[key]["fraction_POI_interaction"]= float(POI_interaction)/float(Mail_interaction)

my_feature_list = ['poi','fraction_POI_interaction','Mail_interaction']

data = featureFormat(data_dict, my_feature_list, sort_keys = True)
plt.scatter(data[:,2],data[:,1],s=50,c=data[:,0], marker = 'o', cmap = plt.cm.coolwarm );
plt.xlabel("Total Mails")
plt.ylabel("Fraction of POI Mails")
plt.show()

# making a total_woth  feature which gives the total income in a year
for key in data_dict:
    total_payment  = fn.replaceNaN(data_dict[key]['total_payments'])
    stock_value = fn.replaceNaN(data_dict[key]['total_stock_value'])
    data_dict[key]["total_worth"]  = total_payment + stock_value
my_feature_list = ['poi','salary','total_worth']
data = featureFormat(data_dict, my_feature_list, sort_keys = True)
plt.scatter(data[:,1],data[:,2],s=50,c=data[:,0], marker = 'o', cmap = plt.cm.coolwarm );
plt.xlabel("Salary")
plt.ylabel("Total Worth")
plt.show()

### Store to my_dataset for easy export below.
features_list += ['total_worth','fraction_POI_interaction','Mail_interaction']
my_dataset = data_dict
#print features_list
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
#GaussianNB
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf,my_dataset,features_list,folds = 1000)

#decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_leaf=1)
test_classifier(clf,my_dataset,features_list,folds = 1000)

#Adaboost
from sklearn.ensemble import AdaBoostClassifier 
clf = AdaBoostClassifier()
test_classifier(clf,my_dataset,features_list,folds = 1000)

#kNearestNeighbours
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors = 4)
test_classifier(clf,my_dataset,features_list)

#all features and their rspective scores
features,features_list,f_scores = fn.select_features(features,labels,features_list,k='all')
print ("features_list---" ,features_list)
print("feature scores")
for feature in f_scores:
    print feature

#upon going through the values for each figures we see that there are a huges difference 
#in values between 'fraction_POI_interaction' and 'total_payments' . So we are considering 
#optimum figures upto 'fraction_POI_interaction' and this is the point of our cutoff hence 
#we select k=10
features,features_list,f_scores = fn.select_features(features,labels,features_list,k=10)
print ("features_list with 10 best features ---" ,features_list)

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#adaboost with PCA
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(random_state=40)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
dt = []
for i in range(6):
    dt.append(DecisionTreeClassifier(max_depth=(i+1)))
params = {'base_estimator': dt,'n_estimators': [60,45, 101,10]}
t0 = time()
boost = GridSearchCV(clf, params, scoring='f1',)
boost = boost.fit(features_train,labels_train)
print("AdaBoost fitting time: %rs" % round(time()-t0, 3))
clf = boost.best_estimator_
t0 = time()
test_classifier(clf, data_dict, features_list, folds = 100)
print("AdaBoost evaluation time: %rs" % round(time()-t0, 3))


### Task 5:  Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#GaussianNB with PCA
t0 = time()
pipe = Pipeline([('pca',PCA()),('classifier',GaussianNB())])
param = {'pca__n_components':[5,6,7,8]}
gsv = GridSearchCV(pipe, param_grid=param,n_jobs=2,scoring = 'f1',cv=2)
gsv.fit(features_train,labels_train) 
clf = gsv.best_estimator_
print("GausianNB with PCA fitting time: %rs" % round(time()-t0, 3))
pred = clf.predict(features_test)

t0 = time()
test_classifier(clf,my_dataset,features_list,folds = 1000)

#GaussianNB is giving better recall score in comparison to the adaboost algorith so my final algo is GaussianNB

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)