#Semester Project for AI - Deven Damiano - dad152@zips.uakron.edu - Nicholas Horvath - nch16@zips.uakron.edu

import sys
import numpy as np
import pandas as pd
import graphviz
import random
from sklearn import tree
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.tree import _tree
from sklearn.preprocessing import OneHotEncoder

#-------------------OPTION 1--------------------------

def learn_tree():
    
    # Additional testing is needed here, trying to cover multiple input methods, user can input a full data table, or individual attributes
    
    print("--------------SUBMENU 1--------------")
    
    file_name = input("Enter the filename of the dataset: ")
    
    try:
        data = pd.read_csv(file_name)
    except:
        print("File not found..")
        return
    
    data_type = "NULL"

attribute_list = list(data)

    data_type = input("If your data is numerical, enter 1, otherwise, enter 2: ")
    
    if(data_type == '1'):
        
        data_type = "1"
    
    else:
        
        data_type = "2"
        data = pd.get_dummies(data)
        data = data.T.reindex(attribute_list).T.fillna(0)

train, test = train_test_split(data, test_size = .15)

    c = tree.DecisionTreeClassifier(min_samples_split=2)
    
    target = input("What label would you like to use as the target value? : ");
    
    if(target in attribute_list):
        
        attribute_list.remove(target)

    X_train = train[attribute_list]
    y_train = train[target]

X_test = test[attribute_list]
y_test = test[target]

test.to_csv('test.csv', sep=',', encoding='utf-8')

    dt = c.fit(X_train, y_train)
    
    #classifier_list = data[target].unique()
    
    #tree.export_graphviz(dt, out_file='tree.dot', class_names = classifier_list)
    tree.export_graphviz(dt, out_file='tree.dot', class_names = attribute_list)
    
    #print(classifier_list)
    
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    
    #write tree to file
    
    joblib.dump(dt, 'tree.joblib')
    
    print("Tree saved as 'tree.joblib'")
    
    return target, attribute_list, data_type

#-----------------OPTION 2-------------------------

def test_accuracy(target,attribute_list, data_type):
    
    tree = joblib.load('tree.joblib')
    
    print("-------------SUBMENU 2---------------")
    
    file_name = input("Please enter the file name of testing data: ")
    
    try:
        data = pd.read_csv(file_name)
    except:
        print("File not found...")
        return
    
    tree = joblib.load('tree.joblib')

attribute_list.append(target)

if(data_type == '1'):
    
    data_type = "1"
    
    else:
        
        data_type = "2"
        data = pd.get_dummies(data)
        data = data.T.reindex(attribute_list).T.fillna(0)

    #train, test = train_test_split(data, test_size = .70)

    attribute_list.remove(target)

X_test = data[attribute_list]
y_test = data[target]

#accuracy

y_pred = tree.predict(X_test)
    
    score = accuracy_score(y_test, y_pred) * 100
    
    print("Accuracy: ", score)
    
    #confusion matrix
    
    print(confusion_matrix(y_test, y_pred))

#-------------------OPTION 3--------------------------

def apply_tree(target, attribute_list):
    
    ##this is wrong, run through tree, then give a predicted target value
    
    print("-------------SUBMENU 3---------------")
    
    tree = joblib.load('tree.joblib')
    
    tree_ = tree.tree_
    feature_name = [
                    attribute_list[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature
                    ]
        
                    value_list = attribute_list
                    
                    tree_dict = {}
                    
                    def recurse(node, depth):
                        
                        if tree_.feature[node] != _tree.TREE_UNDEFINED:
                            
                            name = feature_name[node]
                        
                            threshold = tree_.threshold[node]
                            
                            if name not in tree_dict:
                                
                                print(name, ": ")
                                
                                value = input()
                                
                                    tree_dict[name] = value
                                        
                                        if(float(tree_dict[name]) <= float(threshold)):
                                            
                                            recurse(tree_.children_left[node], depth + 1)
                                                
                                                else:
                                                    
                                                    recurse(tree_.children_right[node], depth + 1)
                                                        
                                                        else:
                                                            
                                                            print("Predicted Result: ", tree.classes_[np.argmax(tree_.value[node])])

recurse(0, 1)


#----------------------OPTION 4-----------------------

def load_model():
    
    #Load the model and allow the user to enter a case for the tree, then predict an answer
    
    print("-------------SUBMENU 4---------------")
    
    tree_file = input("Please enter the filename of the tree: ")
    
    try:
        tree = joblib.load(tree_file)
    except:
        print("File not found...")
        return

    tree_ = tree.tree_
    feature_name = [
                    attribute_list[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature
                    ]

#value_list = attribute_list

#print(tree.classes_)

tree_dict = {}
    
    print(tree.feature_names)
    
    def recurse(node, depth):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            
            name = feature_name[node]
            
            threshold = tree_.threshold[node]
            
            if name not in tree_dict:
                
                print(name, ": ")
                
                value = input()
                
                tree_dict[name] = value
        
            if(float(tree_dict[name]) <= float(threshold)):
                
                recurse(tree_.children_left[node], depth + 1)
            
            else:
                
                recurse(tree_.children_right[node], depth + 1)

else:
    
    print("Predicted Result: ", tree.classes_[np.argmax(tree_.value[node])])
    
    recurse(0, 1)

#---------------------------------------------

running = True

attribute_list = []
target = "NULL"
data_type = "NULL"

tree_created = False
new_cases_created = False

while(running):
    
    print("----------------MAIN MENU------------------")
    print("1. Learn a decision tree and save the tree")
    print("2. Testing accuracy of the decision tree")
    print("3. Applying the decision tree to new cases")
    print("4. Load a tree model and apply to new cases interactively as in menu 3")
    print("5. Quit")
    
    choice = input("Please enter 1-5 to select operation: ")
    
    if(choice == "1"):
        target, attribute_list, data_type = learn_tree()
        tree_created = True
    elif(choice == "2"):
        if(tree_created):
            test_accuracy(target, attribute_list, data_type)
        else:
            print("You must execute option 1 first..")
    elif(choice == "3"):
        if(tree_created):
            apply_tree(target, attribute_list)
            new_cases_created = True
        else:
            print("You must execute option 1 first..")
    elif(choice == "4"):
        #if(new_cases_created):
        load_model()
    #else:
    #print("You must execute option 3 first..")
    elif(choice == "5" or choice == "q"):
        print("Aborting..")
        exit()
    else:
        print("invalid input")
