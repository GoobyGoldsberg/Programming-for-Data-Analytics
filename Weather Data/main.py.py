#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score



weather_data = pd.read_csv("weather.csv")
weather_data_cleaned = weather_data.dropna()



def task1():
    
    unique_locations = weather_data_cleaned["Location"].str.lower()
    
    top5_locations = unique_locations.value_counts().nsmallest(5)
    
    percentage_per_location = (top5_locations / len(weather_data_cleaned)) * 100
    print("Percentage for each location relative to total number of locations: " )
    print(percentage_per_location)
    
    plt.figure(figsize=(10, 6))
    top5_locations.plot(kind='bar', color='skyblue')
    plt.title('Top 5 Locations with Fewest Records')
    plt.xlabel('Location')
    plt.ylabel('Number of Records')
    plt.ylim(400, 1800)
    plt.show()
    
    


def task2():

    weather_data_cleaned_copied = weather_data_cleaned.copy()
    
    weather_data_cleaned_copied['PressureDiff'] = abs(weather_data_cleaned_copied['Pressure9am'] - weather_data_cleaned_copied['Pressure3pm'])

    weather_data_cleaned_copied['PressureDiff'] = weather_data_cleaned_copied['PressureDiff'].apply(lambda x: math.ceil(x))

    x_values = []
    y_values = []

    for d in range(1, 12):

        selected_rows = weather_data_cleaned_copied.loc[weather_data_cleaned_copied['PressureDiff'] >= d]


        rainy_days = selected_rows[selected_rows['RainTomorrow'] == 'Yes'].shape[0]
        non_rainy_days = selected_rows[selected_rows['RainTomorrow'] == 'No'].shape[0]

        if non_rainy_days > 0 and rainy_days > 0:
            ratio = rainy_days / non_rainy_days
        elif non_rainy_days == 0:
            ratio = rainy_days
        else:
            ratio = non_rainy_days

        x_values.append(d)
        y_values.append(ratio)

    plt.plot(x_values, y_values, marker='o')
    plt.xlabel('Minimum Pressure Difference (D)')
    plt.ylabel('Rainfall Ratio (Rainy Days / Non-Rainy Days)')
    plt.title('Effect of Pressure Difference on Subsequent Rainfall')
    plt.show()
        
    # By looking at the graph we can spot an occurring trend where the rainfall ratio, which is
    # prediction chance of having rain the next day, increases at relatively high differences between pressure,
    # However, it is not enough to validate prior assertion about pressure's effect on subsequent rainfall due
    # to the fact that we are creating subsets of our datasets for each iteration of D, and when D reaches
    # around 4, the subsets become smaller and smaller, consequently causing the ratio to be less stable and trustful.
    # for example, when D is 11, the amount of rainy days is 16, and non rainy days is 6, the graph correctly shows
    # us the right ratio, 2.6, but the subset is too small to make prior assumption valid. Because when D is at least 1
    # the number of rainy days is 12320 and non rainy days is 43745.
    
def task3():
    
    attributes = ['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                'Pressure9am', 'Temp9am', 'Temp3pm', 'RainTomorrow']
    sub_df = weather_data_cleaned[attributes] #Dataframing
    sub_df_transformed = sub_df.copy() #To get rid of SettingWithCopy Warning


    labelEncoder = preprocessing.LabelEncoder()
    sub_df_transformed['RainTomorrow'] = labelEncoder.fit_transform(sub_df['RainTomorrow'])


    predictors = sub_df.drop('RainTomorrow', axis=1)
    target = sub_df['RainTomorrow']

    feature_importances = {}

    for max_depth in range(1, 36):
        classifier = DecisionTreeClassifier(max_depth=max_depth)
        
        classifier.fit(predictors, target)
        
        feature_importances[max_depth] = classifier.feature_importances_

    importance_df = pd.DataFrame(feature_importances, index=predictors.columns) #Dataframing the dictionary

    importance_df.T.plot(marker='o', figsize=(12, 8))
    plt.title('Feature Importances at Different Maximum Depths')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Feature Importance')
    plt.legend(title='Feature')
    plt.show()
    
    # By looking at the created plot, I can say that as the Maximum depth increases,
    # the feature importance becomes less and less relevant if it was important at the beginning,
    # but the features that were less important at the low depth level slightly increase, both
    # features seem to stagnate at a certain level and flat-line.

def task4():
    
    features_A = weather_data_cleaned[['Pressure9am', 'Pressure3pm']]
    features_B = weather_data_cleaned[['WindDir9am', 'WindDir3pm']]
    target = weather_data_cleaned['RainTomorrow']
    
    features_A = pd.get_dummies(features_A)
    features_B = pd.get_dummies(features_B)

    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(features_A, target, test_size=0.33, random_state=42)
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(features_B, target, test_size=0.33, random_state=42)

    def get_accuracy(X_train, X_test, y_train, y_test):
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        # Predictions on training set
        train_predicted_labels = clf.predict(X_train)
        train_accuracy = accuracy_score(train_predicted_labels, y_train)

        # Predictions on test set
        test_predicted_labels = clf.predict(X_test)
        test_accuracy = accuracy_score(test_predicted_labels, y_test)

        return train_accuracy, test_accuracy
    
    train_accuracy_A, test_accuracy_A = get_accuracy(X_train_A, X_test_A, y_train_A, y_test_A)
    train_accuracy_B, test_accuracy_B = get_accuracy(X_train_B, X_test_B, y_train_B, y_test_B)

    print(f'Training Accuracy for part A (Pressure): {train_accuracy_A}')
    print(f'Test Accuracy for part A (Pressure): {test_accuracy_A}')
    print(f'Training Accuracy for part B (WindDir): {train_accuracy_B}')
    print(f'Test Accuracy for part B (WindDir): {test_accuracy_B}')
                
    # Based ny my results, the model related to features related to pressure shows higher training accuracy
    # than the model with wind direction.
    # However, Pressure model has lower test accuracy compared to the WindDir model
    # This means that the model with Wind Direction may generalize better to unseen data.
    # With that being said, the Wind Direction model appears to be more effective for predicting RainTomorrow.
    
    
def task5():
    
    selected_columns = ['RainTomorrow', 'WindDir9am', 'WindGustDir', 'WindDir3pm']
    sub_df = weather_data[selected_columns]

    mask = sub_df[['WindDir9am', 'WindGustDir', 'WindDir3pm']].apply(lambda x: x.str.len() == 3).any(axis=1)

    filtered_sub_df = sub_df[mask]

    filtered_sub_df = pd.get_dummies(filtered_sub_df, columns=['WindDir9am', 'WindGustDir', 'WindDir3pm'])
    

    
    
    
    def decisionTree(X, y, depths):
        print("Running the decision tree function")
        train_accuracy_list = []
        test_accuracy_list = []

        # Encode categorical variables
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.ravel())

        kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        for depth in range(1,11):
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)

            train_accuracies = []
            test_accuracies = []
            print("Starting the nested dt loop")
            for train, test in kfold.split(X, y_encoded):
                
                clf.fit(X.iloc[train], y_encoded[train])
                train_accuracy = clf.score(X.iloc[train], y_encoded[train])
                test_accuracy = clf.score(X.iloc[test], y_encoded[test])

                train_accuracies.append(np.mean(train_accuracy))
                test_accuracies.append(np.mean(test_accuracy))
            
            
            train_accuracy_list.append(np.mean(train_accuracies))
            test_accuracy_list.append(np.mean(test_accuracies))

        print("Ending the dt function")
        return train_accuracy_list, test_accuracy_list

    # From my experience this function takes about an hour to finish its 10 loops.
    def kNeighbours(X, y, neighbours):
        print("Running the knn function")
        train_accuracy_list = []
        test_accuracy_list = []

        kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

        print("Running the for loop")
        for n_neighbours in neighbours:
            
            clf = KNeighborsClassifier(n_neighbors=n_neighbours)

            print("Training")
            train_accuracies = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')
            print("Testing")
            test_accuracies = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

            train_accuracy_list.append(np.mean(train_accuracies))
            test_accuracy_list.append(np.mean(test_accuracies))
        print("Ending the for loop")

        return train_accuracy_list, test_accuracy_list


    
    
    def plot_everything(depths, train_acc_dt, test_acc_dt, neighbours, train_acc_knn, test_acc_knn):
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(depths, train_acc_dt, label='Training Accuracy', marker='o')
        plt.plot(depths, test_acc_dt, label='Test Accuracy', marker='o')
        plt.title('Decision Tree Classifier')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(neighbours, train_acc_knn, label='Training Accuracy', marker='o')
        plt.plot(neighbours, test_acc_knn, label='Test Accuracy', marker='o')
        plt.title('K-Nearest Neighbors Classifier')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()     

    labelEncoder = LabelEncoder()
    filtered_sub_df['RainTomorrow'] = labelEncoder.fit_transform(filtered_sub_df['RainTomorrow'])
    
    X = filtered_sub_df.drop(columns=['RainTomorrow'])
    y = filtered_sub_df['RainTomorrow']
    
    depths = list(range(1, 11))
    train_acc_dt, test_acc_dt = decisionTree(X, y, depths)
    
    neighbours = list(range(1, 11))
    train_acc_knn, test_acc_knn = kNeighbours(X, y, neighbours)
    
    plot_everything(depths, train_acc_dt, test_acc_dt, neighbours, train_acc_knn, test_acc_knn)
    
    # By looking at the graphs we can see:
    # in the Decision tree graph, as the depth number increases, the training accuracy increases largely aswell,
    # however, because of overfitting, it largely decreases the test accuracy.
    # in the K-nearest neighbour, as the number of neighbours increase, both the training and test accuracy increase,
    # because of that, Knn is the most optimal choice between the two algorithms


def task6():

    selected_columns = ['MinTemp', 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
                    'Pressure9am', 'Pressure3pm', 'Rainfall', 'Temp9am', 'Temp3pm']

    # Ensure numerical values
    numeric_df = weather_data_cleaned[selected_columns].apply(pd.to_numeric, errors='coerce').dropna()

        # Standardize the data
    scaler = preprocessing.MinMaxScaler()
    numeric_df = scaler.fit_transform(numeric_df) 

    clusters = [2,3,4,5,6,7,8]
    
    inertia = []
    
    for k in clusters:
        kmeans = KMeans(n_clusters=k+1, random_state=42, n_init='auto')
        kmeans.fit(numeric_df)
        inertia.append(kmeans.inertia_)
    
    # Elbow method
    plt.figure(figsize=(8, 6))
    plt.plot(clusters, inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Distortion Function')
    plt.show()


    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(numeric_df)

    # Scatter graph
    plt.figure(figsize=(10, 8))
    plt.scatter(numeric_df[:, 0], numeric_df[:, 1], c=clusters, cmap='viridis', alpha=1)
    plt.title('K-Means Clustering (Optimal K={})'.format(k))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Cluster')
    plt.show()
    
    # From looking at the Elbow method graph, I believe that the optimal number of cluster is 4.
    # From looking at our scatter graph, we can see that the clusters and their centroids are somehow defined
    # except for the first cluster (0), where it is mostly covered by its surrounding clusters. 
    
def task7():
    selected_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Pressure9am', 'Cloud9am', 'Temp9am', 'RainToday']
    target = 'RainTomorrow'

    X = weather_data_cleaned[selected_columns]
    X = pd.get_dummies(X, columns=['RainToday'], drop_first=True)

    y = weather_data_cleaned[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)

    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    
    print(f"Accuracy: {accuracy}")
    print(f"Based on all the data, there is {accuracy}% chance that it will rain tomorrow.")
    
    # The concept of this task is to predict whether it will rain tomorrow based on all the historical data
    # Using Random Forest algorithm approach
    # The model's accuracy provides insights that can be used in weather forecasting.

#task1()
#task2()
#task3()
#task4()
#task5()
#task6()
#task7()
