import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def gnb(x_train, y_train, x_test):
    clfGNB = GaussianNB().fit(train_x, train_y)
    predicted = clfGNB.predict(test_x)
    return predicted


def bdtc(x_train, y_train, x_test):
    clfBDT = DecisionTreeClassifier().fit(train_x, train_y)
    predicted = clfBDT.predict(test_x)
    return predicted


def tdtc(x_train, y_train):
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10], 'min_samples_split': [2, 3, 4]}
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
    clf.fit(train_x, train_y)
    return clf


def pc(x_train, y_train, x_test):
    clfP = Perceptron().fit(train_x, train_y)
    predicted = clfP.predict(test_x)
    return predicted


def bmlp(x_train, y_train, x_test):
    clfBMLP = MLPClassifier(activation='logistic', solver='sgd',
                            max_iter=10000)  # 1 hidden layer with 100 neurons is default
    clfBMLP.fit(train_x, train_y)
    predicted = clfBMLP.predict(test_x)
    return predicted


def tmlp(x_train, y_train):
    param_grid_TMLP = {'activation': ['logistic', 'tanh', 'identity', 'relu'],
                       'hidden_layer_sizes': [(10, 10, 10), (30, 50)], 'solver': ['adam', 'sgd'], 'max_iter': [10000]}
    clf = GridSearchCV(MLPClassifier(), param_grid_TMLP)
    clf.fit(train_x, train_y)
    return clf


if __name__ == '__main__':

    # 2)
    dataset = pd.read_csv('drug200.csv')

    # 3)
    # extract drug column
    drugsCol = dataset.T.loc['Drug']

    # Count drug instances
    numDrugA = 0
    numDrugB = 0
    numDrugC = 0
    numDrugX = 0
    numDrugY = 0

    for i in range(dataset.shape[0]):
        if drugsCol[i] == "drugA":
            numDrugA = numDrugA + 1
        elif drugsCol[i] == "drugB":
            numDrugB = numDrugB + 1
        elif drugsCol[i] == "drugC":
            numDrugC = numDrugC + 1
        elif drugsCol[i] == "drugX":
            numDrugX = numDrugX + 1
        elif drugsCol[i] == "drugY":
            numDrugY = numDrugY + 1

    # Arrange positions and data
    posi = [1, 2, 3, 4, 5]
    dist = [numDrugA, numDrugB, numDrugC, numDrugX, numDrugY]

    # Create the plot
    plt.bar(posi, dist)

    # Fix labels
    labels = ['DrugA', 'DrugB', 'DrugC', 'DrugX', 'DrugY']
    plt.xticks(posi, labels)

    # Save plot
    plt.savefig('drug-distribution.pdf', dpi=100)

    # 4)
    # Convert ordinal and nominal values to numerical
    dataset.BP = pd.Categorical(dataset.BP, ['LOW', 'NORMAL', 'HIGH'], ordered=True)
    dataset.BP = dataset.BP.cat.codes

    dataset.Cholesterol = pd.Categorical(dataset.Cholesterol, ['NORMAL', 'HIGH'], ordered=True)
    dataset.Cholesterol = dataset.Cholesterol.cat.codes

    # dataset.Drug = pd.Categorical(dataset.Drug, ['drugA', 'drugB', 'drugC', 'drugX', 'drugY'], ordered=True)
    # dataset.Drug = dataset.Drug.cat.codes

    dataset = pd.get_dummies(dataset, columns=['Sex'])

    # 5)
    dataset = dataset.drop(['Drug'], axis=1)

    train_x, test_x, train_y, test_y = \
        train_test_split(dataset, drugsCol, test_size=0.2, random_state=None)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(train_x)
        print(train_y)

    # 6)

    # FIRST RUN *********************************************************************************************

    # a)
    predictedGNB1 = gnb(train_x, train_y, test_x)

    reportGNB1 = classification_report(test_y, predictedGNB1, output_dict=True, zero_division=0)
    macrof1_GNB1 = reportGNB1['macro avg']['f1-score']
    weightedf1_GNB1 = reportGNB1['weighted avg']['f1-score']
    accuracy_GNB1 = reportGNB1['accuracy']

    # b)
    predictedBDT1 = bdtc(train_x, train_y, test_x)

    reportBDT1 = classification_report(test_y, predictedBDT1, output_dict=True, zero_division=0)
    macrof1_BDT1 = reportBDT1['macro avg']['f1-score']
    weightedf1_BDT1 = reportBDT1['weighted avg']['f1-score']
    accuracy_BDT1 = reportBDT1['accuracy']

    # c)
    clfTDT1 = tdtc(train_x, train_y)
    predictedTDT1 = clfTDT1.predict(test_x)

    reportTDT1 = classification_report(test_y, predictedTDT1, output_dict=True, zero_division=0)
    macrof1_TDT1 = reportTDT1['macro avg']['f1-score']
    weightedf1_TDT1 = reportTDT1['weighted avg']['f1-score']
    accuracy_TDT1 = reportTDT1['accuracy']

    # d)
    predictedP1 = pc(train_x, train_y, test_x)

    reportP1 = classification_report(test_y, predictedP1, output_dict=True, zero_division=0)
    macrof1_P1 = reportP1['macro avg']['f1-score']
    weightedf1_P1 = reportP1['weighted avg']['f1-score']
    accuracy_P1 = reportP1['accuracy']

    # e)
    predictedBMLP1 = bmlp(train_x, train_y, test_x)

    reportBMLP1 = classification_report(test_y, predictedBMLP1, output_dict=True, zero_division=0)
    macrof1_BMLP1 = reportBMLP1['macro avg']['f1-score']
    weightedf1_BMLP1 = reportBMLP1['weighted avg']['f1-score']
    accuracy_BMLP1 = reportBMLP1['accuracy']

    # f)
    clfTMLP1 = tmlp(train_x, train_y)
    predictedTMLP1 = clfTMLP1.predict(test_x)

    reportTMLP1 = classification_report(test_y, predictedTMLP1, output_dict=True, zero_division=0)
    macrof1_TMLP1 = reportTMLP1['macro avg']['f1-score']
    weightedf1_TMLP1 = reportTMLP1['weighted avg']['f1-score']
    accuracy_TMLP1 = reportTMLP1['accuracy']

    # *********************************************************************************************************

    # SECOND RUN *********************************************************************************************

    # a)
    predictedGNB2 = gnb(train_x, train_y, test_x)

    reportGNB2 = classification_report(test_y, predictedGNB2, output_dict=True, zero_division=0)
    macrof1_GNB2 = reportGNB2['macro avg']['f1-score']
    weightedf1_GNB2 = reportGNB2['weighted avg']['f1-score']
    accuracy_GNB2 = reportGNB2['accuracy']

    # b)
    predictedBDT2 = bdtc(train_x, train_y, test_x)

    reportBDT2 = classification_report(test_y, predictedBDT2, output_dict=True, zero_division=0)
    macrof1_BDT2 = reportBDT2['macro avg']['f1-score']
    weightedf1_BDT2 = reportBDT2['weighted avg']['f1-score']
    accuracy_BDT2 = reportBDT2['accuracy']

    # c)
    clfTDT2 = tdtc(train_x, train_y)
    predictedTDT2 = clfTDT2.predict(test_x)

    reportTDT2 = classification_report(test_y, predictedTDT2, output_dict=True, zero_division=0)
    macrof1_TDT2 = reportTDT2['macro avg']['f1-score']
    weightedf1_TDT2 = reportTDT2['weighted avg']['f1-score']
    accuracy_TDT2 = reportTDT2['accuracy']

    # d)
    predictedP2 = pc(train_x, train_y, test_x)

    reportP2 = classification_report(test_y, predictedP2, output_dict=True, zero_division=0)
    macrof1_P2 = reportP2['macro avg']['f1-score']
    weightedf1_P2 = reportP2['weighted avg']['f1-score']
    accuracy_P2 = reportP2['accuracy']

    # e)
    predictedBMLP2 = bmlp(train_x, train_y, test_x)

    reportBMLP2 = classification_report(test_y, predictedBMLP2, output_dict=True, zero_division=0)
    macrof1_BMLP2 = reportBMLP2['macro avg']['f1-score']
    weightedf1_BMLP2 = reportBMLP2['weighted avg']['f1-score']
    accuracy_BMLP2 = reportBMLP2['accuracy']

    # f)
    clfTMLP2 = tmlp(train_x, train_y)
    predictedTMLP2 = clfTMLP2.predict(test_x)

    reportTMLP2 = classification_report(test_y, predictedTMLP2, output_dict=True, zero_division=0)
    macrof1_TMLP2 = reportTMLP2['macro avg']['f1-score']
    weightedf1_TMLP2 = reportTMLP2['weighted avg']['f1-score']
    accuracy_TMLP2 = reportTMLP2['accuracy']

    # *********************************************************************************************************

    # THIRD RUN *********************************************************************************************

    # a)
    predictedGNB3 = gnb(train_x, train_y, test_x)

    reportGNB3 = classification_report(test_y, predictedGNB3, output_dict=True, zero_division=0)
    macrof1_GNB3 = reportGNB3['macro avg']['f1-score']
    weightedf1_GNB3 = reportGNB3['weighted avg']['f1-score']
    accuracy_GNB3 = reportGNB3['accuracy']

    # b)
    predictedBDT3 = bdtc(train_x, train_y, test_x)

    reportBDT3 = classification_report(test_y, predictedBDT3, output_dict=True, zero_division=0)
    macrof1_BDT3 = reportBDT3['macro avg']['f1-score']
    weightedf1_BDT3 = reportBDT3['weighted avg']['f1-score']
    accuracy_BDT3 = reportBDT3['accuracy']

    # c)
    clfTDT3 = tdtc(train_x, train_y)
    predictedTDT3 = clfTDT3.predict(test_x)

    reportTDT3 = classification_report(test_y, predictedTDT3, output_dict=True, zero_division=0)
    macrof1_TDT3 = reportTDT3['macro avg']['f1-score']
    weightedf1_TDT3 = reportTDT3['weighted avg']['f1-score']
    accuracy_TDT3 = reportTDT3['accuracy']

    # d)
    predictedP3 = pc(train_x, train_y, test_x)

    reportP3 = classification_report(test_y, predictedP3, output_dict=True, zero_division=0)
    macrof1_P3 = reportP3['macro avg']['f1-score']
    weightedf1_P3 = reportP3['weighted avg']['f1-score']
    accuracy_P3 = reportP3['accuracy']

    # e)
    predictedBMLP3 = bmlp(train_x, train_y, test_x)

    reportBMLP3 = classification_report(test_y, predictedBMLP3, output_dict=True, zero_division=0)
    macrof1_BMLP3 = reportBMLP3['macro avg']['f1-score']
    weightedf1_BMLP3 = reportBMLP3['weighted avg']['f1-score']
    accuracy_BMLP3 = reportBMLP3['accuracy']

    # f)
    clfTMLP3 = tmlp(train_x, train_y)
    predictedTMLP3 = clfTMLP3.predict(test_x)

    reportTMLP3 = classification_report(test_y, predictedTMLP3, output_dict=True, zero_division=0)
    macrof1_TMLP3 = reportTMLP3['macro avg']['f1-score']
    weightedf1_TMLP3 = reportTMLP3['weighted avg']['f1-score']
    accuracy_TMLP3 = reportTMLP3['accuracy']

    # *********************************************************************************************************

    # FOURTH RUN *********************************************************************************************

    # a)
    predictedGNB4 = gnb(train_x, train_y, test_x)

    reportGNB4 = classification_report(test_y, predictedGNB4, output_dict=True, zero_division=0)
    macrof1_GNB4 = reportGNB4['macro avg']['f1-score']
    weightedf1_GNB4 = reportGNB4['weighted avg']['f1-score']
    accuracy_GNB4 = reportGNB4['accuracy']

    # b)
    predictedBDT4 = bdtc(train_x, train_y, test_x)

    reportBDT4 = classification_report(test_y, predictedBDT4, output_dict=True, zero_division=0)
    macrof1_BDT4 = reportBDT4['macro avg']['f1-score']
    weightedf1_BDT4 = reportBDT4['weighted avg']['f1-score']
    accuracy_BDT4 = reportBDT4['accuracy']

    # c)
    clfTDT4 = tdtc(train_x, train_y)
    predictedTDT4 = clfTDT4.predict(test_x)

    reportTDT4 = classification_report(test_y, predictedTDT4, output_dict=True, zero_division=0)
    macrof1_TDT4 = reportTDT4['macro avg']['f1-score']
    weightedf1_TDT4 = reportTDT4['weighted avg']['f1-score']
    accuracy_TDT4 = reportTDT4['accuracy']

    # d)
    predictedP4 = pc(train_x, train_y, test_x)

    reportP4 = classification_report(test_y, predictedP4, output_dict=True, zero_division=0)
    macrof1_P4 = reportP4['macro avg']['f1-score']
    weightedf1_P4 = reportP4['weighted avg']['f1-score']
    accuracy_P4 = reportP4['accuracy']

    # e)
    predictedBMLP4 = bmlp(train_x, train_y, test_x)

    reportBMLP4 = classification_report(test_y, predictedBMLP4, output_dict=True, zero_division=0)
    macrof1_BMLP4 = reportBMLP4['macro avg']['f1-score']
    weightedf1_BMLP4 = reportBMLP4['weighted avg']['f1-score']
    accuracy_BMLP4 = reportBMLP4['accuracy']

    # f)
    clfTMLP4 = tmlp(train_x, train_y)
    predictedTMLP4 = clfTMLP4.predict(test_x)

    reportTMLP4 = classification_report(test_y, predictedTMLP4, output_dict=True, zero_division=0)
    macrof1_TMLP4 = reportTMLP4['macro avg']['f1-score']
    weightedf1_TMLP4 = reportTMLP4['weighted avg']['f1-score']
    accuracy_TMLP4 = reportTMLP4['accuracy']

    # *********************************************************************************************************

    # FIFTH RUN *********************************************************************************************

    # a)
    predictedGNB5 = gnb(train_x, train_y, test_x)

    reportGNB5 = classification_report(test_y, predictedGNB5, output_dict=True, zero_division=0)
    macrof1_GNB5 = reportGNB5['macro avg']['f1-score']
    weightedf1_GNB5 = reportGNB5['weighted avg']['f1-score']
    accuracy_GNB5 = reportGNB5['accuracy']

    # b)
    predictedBDT5 = bdtc(train_x, train_y, test_x)

    reportBDT5 = classification_report(test_y, predictedBDT5, output_dict=True, zero_division=0)
    macrof1_BDT5 = reportBDT5['macro avg']['f1-score']
    weightedf1_BDT5 = reportBDT5['weighted avg']['f1-score']
    accuracy_BDT5 = reportBDT5['accuracy']

    # c)
    clfTDT5 = tdtc(train_x, train_y)
    predictedTDT5 = clfTDT5.predict(test_x)

    reportTDT5 = classification_report(test_y, predictedTDT5, output_dict=True, zero_division=0)
    macrof1_TDT5 = reportTDT5['macro avg']['f1-score']
    weightedf1_TDT5 = reportTDT5['weighted avg']['f1-score']
    accuracy_TDT5 = reportTDT5['accuracy']

    # d)
    predictedP5 = pc(train_x, train_y, test_x)

    reportP5 = classification_report(test_y, predictedP5, output_dict=True, zero_division=0)
    macrof1_P5 = reportP5['macro avg']['f1-score']
    weightedf1_P5 = reportP5['weighted avg']['f1-score']
    accuracy_P5 = reportP5['accuracy']

    # e)
    predictedBMLP5 = bmlp(train_x, train_y, test_x)

    reportBMLP5 = classification_report(test_y, predictedBMLP5, output_dict=True, zero_division=0)
    macrof1_BMLP5 = reportBMLP5['macro avg']['f1-score']
    weightedf1_BMLP5 = reportBMLP5['weighted avg']['f1-score']
    accuracy_BMLP5 = reportBMLP5['accuracy']

    # f)
    clfTMLP5 = tmlp(train_x, train_y)
    predictedTMLP5 = clfTMLP5.predict(test_x)

    reportTMLP5 = classification_report(test_y, predictedTMLP5, output_dict=True, zero_division=0)
    macrof1_TMLP5 = reportTMLP5['macro avg']['f1-score']
    weightedf1_TMLP5 = reportTMLP5['weighted avg']['f1-score']
    accuracy_TMLP5 = reportTMLP5['accuracy']

    # *********************************************************************************************************

    # SIXTH RUN *********************************************************************************************

    # a)
    predictedGNB6 = gnb(train_x, train_y, test_x)

    reportGNB6 = classification_report(test_y, predictedGNB6, output_dict=True, zero_division=0)
    macrof1_GNB6 = reportGNB6['macro avg']['f1-score']
    weightedf1_GNB6 = reportGNB6['weighted avg']['f1-score']
    accuracy_GNB6 = reportGNB6['accuracy']

    # b)
    predictedBDT6 = bdtc(train_x, train_y, test_x)

    reportBDT6 = classification_report(test_y, predictedBDT6, output_dict=True, zero_division=0)
    macrof1_BDT6 = reportBDT6['macro avg']['f1-score']
    weightedf1_BDT6 = reportBDT6['weighted avg']['f1-score']
    accuracy_BDT6 = reportBDT6['accuracy']

    # c)
    clfTDT6 = tdtc(train_x, train_y)
    predictedTDT6 = clfTDT6.predict(test_x)

    reportTDT6 = classification_report(test_y, predictedTDT6, output_dict=True, zero_division=0)
    macrof1_TDT6 = reportTDT6['macro avg']['f1-score']
    weightedf1_TDT6 = reportTDT6['weighted avg']['f1-score']
    accuracy_TDT6 = reportTDT6['accuracy']

    # d)
    predictedP6 = pc(train_x, train_y, test_x)

    reportP6 = classification_report(test_y, predictedP6, output_dict=True, zero_division=0)
    macrof1_P6 = reportP6['macro avg']['f1-score']
    weightedf1_P6 = reportP6['weighted avg']['f1-score']
    accuracy_P6 = reportP6['accuracy']

    # e)
    predictedBMLP6 = bmlp(train_x, train_y, test_x)

    reportBMLP6 = classification_report(test_y, predictedBMLP6, output_dict=True, zero_division=0)
    macrof1_BMLP6 = reportBMLP6['macro avg']['f1-score']
    weightedf1_BMLP6 = reportBMLP6['weighted avg']['f1-score']
    accuracy_BMLP6 = reportBMLP6['accuracy']

    # f)
    clfTMLP6 = tmlp(train_x, train_y)
    predictedTMLP6 = clfTMLP6.predict(test_x)

    reportTMLP6 = classification_report(test_y, predictedTMLP6, output_dict=True, zero_division=0)
    macrof1_TMLP6 = reportTMLP6['macro avg']['f1-score']
    weightedf1_TMLP6 = reportTMLP6['weighted avg']['f1-score']
    accuracy_TMLP6 = reportTMLP6['accuracy']

    # *********************************************************************************************************

    # SEVENTH RUN *********************************************************************************************

    # a)
    predictedGNB7 = gnb(train_x, train_y, test_x)

    reportGNB7 = classification_report(test_y, predictedGNB7, output_dict=True, zero_division=0)
    macrof1_GNB7 = reportGNB7['macro avg']['f1-score']
    weightedf1_GNB7 = reportGNB7['weighted avg']['f1-score']
    accuracy_GNB7 = reportGNB7['accuracy']

    # b)
    predictedBDT7 = bdtc(train_x, train_y, test_x)

    reportBDT7 = classification_report(test_y, predictedBDT7, output_dict=True, zero_division=0)
    macrof1_BDT7 = reportBDT7['macro avg']['f1-score']
    weightedf1_BDT7 = reportBDT7['weighted avg']['f1-score']
    accuracy_BDT7 = reportBDT7['accuracy']

    # c)
    clfTDT7 = tdtc(train_x, train_y)
    predictedTDT7 = clfTDT7.predict(test_x)

    reportTDT7 = classification_report(test_y, predictedTDT7, output_dict=True, zero_division=0)
    macrof1_TDT7 = reportTDT7['macro avg']['f1-score']
    weightedf1_TDT7 = reportTDT7['weighted avg']['f1-score']
    accuracy_TDT7 = reportTDT7['accuracy']

    # d)
    predictedP7 = pc(train_x, train_y, test_x)

    reportP7 = classification_report(test_y, predictedP7, output_dict=True, zero_division=0)
    macrof1_P7 = reportP7['macro avg']['f1-score']
    weightedf1_P7 = reportP7['weighted avg']['f1-score']
    accuracy_P7 = reportP7['accuracy']

    # e)
    predictedBMLP7 = bmlp(train_x, train_y, test_x)

    reportBMLP7 = classification_report(test_y, predictedBMLP7, output_dict=True, zero_division=0)
    macrof1_BMLP7 = reportBMLP7['macro avg']['f1-score']
    weightedf1_BMLP7 = reportBMLP7['weighted avg']['f1-score']
    accuracy_BMLP7 = reportBMLP7['accuracy']

    # f)
    clfTMLP7 = tmlp(train_x, train_y)
    predictedTMLP7 = clfTMLP7.predict(test_x)

    reportTMLP7 = classification_report(test_y, predictedTMLP7, output_dict=True, zero_division=0)
    macrof1_TMLP7 = reportTMLP7['macro avg']['f1-score']
    weightedf1_TMLP7 = reportTMLP7['weighted avg']['f1-score']
    accuracy_TMLP7 = reportTMLP7['accuracy']

    # *********************************************************************************************************

    # EIGHTH RUN *********************************************************************************************

    # a)
    predictedGNB8 = gnb(train_x, train_y, test_x)

    reportGNB8 = classification_report(test_y, predictedGNB8, output_dict=True, zero_division=0)
    macrof1_GNB8 = reportGNB8['macro avg']['f1-score']
    weightedf1_GNB8 = reportGNB8['weighted avg']['f1-score']
    accuracy_GNB8 = reportGNB8['accuracy']

    # b)
    predictedBDT8 = bdtc(train_x, train_y, test_x)

    reportBDT8 = classification_report(test_y, predictedBDT8, output_dict=True, zero_division=0)
    macrof1_BDT8 = reportBDT8['macro avg']['f1-score']
    weightedf1_BDT8 = reportBDT8['weighted avg']['f1-score']
    accuracy_BDT8 = reportBDT8['accuracy']

    # c)
    clfTDT8 = tdtc(train_x, train_y)
    predictedTDT8 = clfTDT8.predict(test_x)

    reportTDT8 = classification_report(test_y, predictedTDT8, output_dict=True, zero_division=0)
    macrof1_TDT8 = reportTDT8['macro avg']['f1-score']
    weightedf1_TDT8 = reportTDT8['weighted avg']['f1-score']
    accuracy_TDT8 = reportTDT8['accuracy']

    # d)
    predictedP8 = pc(train_x, train_y, test_x)

    reportP8 = classification_report(test_y, predictedP8, output_dict=True, zero_division=0)
    macrof1_P8 = reportP8['macro avg']['f1-score']
    weightedf1_P8 = reportP8['weighted avg']['f1-score']
    accuracy_P8 = reportP8['accuracy']

    # e)
    predictedBMLP8 = bmlp(train_x, train_y, test_x)

    reportBMLP8 = classification_report(test_y, predictedBMLP8, output_dict=True, zero_division=0)
    macrof1_BMLP8 = reportBMLP8['macro avg']['f1-score']
    weightedf1_BMLP8 = reportBMLP8['weighted avg']['f1-score']
    accuracy_BMLP8 = reportBMLP8['accuracy']

    # f)
    clfTMLP8 = tmlp(train_x, train_y)
    predictedTMLP8 = clfTMLP8.predict(test_x)

    reportTMLP8 = classification_report(test_y, predictedTMLP8, output_dict=True, zero_division=0)
    macrof1_TMLP8 = reportTMLP8['macro avg']['f1-score']
    weightedf1_TMLP8 = reportTMLP8['weighted avg']['f1-score']
    accuracy_TMLP8 = reportTMLP8['accuracy']

    # *********************************************************************************************************

    # NINTH RUN *********************************************************************************************

    # a)
    predictedGNB9 = gnb(train_x, train_y, test_x)

    reportGNB9 = classification_report(test_y, predictedGNB9, output_dict=True, zero_division=0)
    macrof1_GNB9 = reportGNB9['macro avg']['f1-score']
    weightedf1_GNB9 = reportGNB9['weighted avg']['f1-score']
    accuracy_GNB9 = reportGNB9['accuracy']

    # b)
    predictedBDT9 = bdtc(train_x, train_y, test_x)

    reportBDT9 = classification_report(test_y, predictedBDT9, output_dict=True, zero_division=0)
    macrof1_BDT9 = reportBDT9['macro avg']['f1-score']
    weightedf1_BDT9 = reportBDT9['weighted avg']['f1-score']
    accuracy_BDT9 = reportBDT9['accuracy']

    # c)
    clfTDT9 = tdtc(train_x, train_y)
    predictedTDT9 = clfTDT9.predict(test_x)

    reportTDT9 = classification_report(test_y, predictedTDT9, output_dict=True, zero_division=0)
    macrof1_TDT9 = reportTDT9['macro avg']['f1-score']
    weightedf1_TDT9 = reportTDT9['weighted avg']['f1-score']
    accuracy_TDT9 = reportTDT9['accuracy']

    # d)
    predictedP9 = pc(train_x, train_y, test_x)

    reportP9 = classification_report(test_y, predictedP9, output_dict=True, zero_division=0)
    macrof1_P9 = reportP9['macro avg']['f1-score']
    weightedf1_P9 = reportP9['weighted avg']['f1-score']
    accuracy_P9 = reportP9['accuracy']

    # e)
    predictedBMLP9 = bmlp(train_x, train_y, test_x)

    reportBMLP9 = classification_report(test_y, predictedBMLP9, output_dict=True, zero_division=0)
    macrof1_BMLP9 = reportBMLP9['macro avg']['f1-score']
    weightedf1_BMLP9 = reportBMLP9['weighted avg']['f1-score']
    accuracy_BMLP9 = reportBMLP9['accuracy']

    # f)
    clfTMLP9 = tmlp(train_x, train_y)
    predictedTMLP9 = clfTMLP9.predict(test_x)

    reportTMLP9 = classification_report(test_y, predictedTMLP9, output_dict=True, zero_division=0)
    macrof1_TMLP9 = reportTMLP9['macro avg']['f1-score']
    weightedf1_TMLP9 = reportTMLP9['weighted avg']['f1-score']
    accuracy_TMLP9 = reportTMLP9['accuracy']

    # *********************************************************************************************************

    # TENTH RUN *********************************************************************************************

    # a)
    predictedGNB10 = gnb(train_x, train_y, test_x)

    reportGNB10 = classification_report(test_y, predictedGNB10, output_dict=True, zero_division=0)
    macrof1_GNB10 = reportGNB10['macro avg']['f1-score']
    weightedf1_GNB10 = reportGNB10['weighted avg']['f1-score']
    accuracy_GNB10 = reportGNB10['accuracy']

    # b)
    predictedBDT10 = bdtc(train_x, train_y, test_x)

    reportBDT10 = classification_report(test_y, predictedBDT10, output_dict=True, zero_division=0)
    macrof1_BDT10 = reportBDT10['macro avg']['f1-score']
    weightedf1_BDT10 = reportBDT10['weighted avg']['f1-score']
    accuracy_BDT10 = reportBDT10['accuracy']

    # c)
    clfTDT10 = tdtc(train_x, train_y)
    predictedTDT10 = clfTDT10.predict(test_x)

    reportTDT10 = classification_report(test_y, predictedTDT10, output_dict=True, zero_division=0)
    macrof1_TDT10 = reportTDT10['macro avg']['f1-score']
    weightedf1_TDT10 = reportTDT10['weighted avg']['f1-score']
    accuracy_TDT10 = reportTDT10['accuracy']

    # d)
    predictedP10 = pc(train_x, train_y, test_x)

    reportP10 = classification_report(test_y, predictedP10, output_dict=True, zero_division=0)
    macrof1_P10 = reportP10['macro avg']['f1-score']
    weightedf1_P10 = reportP10['weighted avg']['f1-score']
    accuracy_P10 = reportP10['accuracy']

    # e)
    predictedBMLP10 = bmlp(train_x, train_y, test_x)

    reportBMLP10 = classification_report(test_y, predictedBMLP10, output_dict=True, zero_division=0)
    macrof1_BMLP10 = reportBMLP10['macro avg']['f1-score']
    weightedf1_BMLP10 = reportBMLP10['weighted avg']['f1-score']
    accuracy_BMLP10 = reportBMLP10['accuracy']

    # f)
    clfTMLP10 = tmlp(train_x, train_y)
    predictedTMLP10 = clfTMLP10.predict(test_x)

    reportTMLP10 = classification_report(test_y, predictedTMLP10, output_dict=True, zero_division=0)
    macrof1_TMLP10 = reportTMLP10['macro avg']['f1-score']
    weightedf1_TMLP10 = reportTMLP10['weighted avg']['f1-score']
    accuracy_TMLP10 = reportTMLP10['accuracy']

    # *********************************************************************************************************

    with open('drugs-performance.txt', 'a') as f:

        f.write('FIRST RUN STATISTICS*****************************************************************************\n\n')

        f.write('a) Gaussian Naive Baye\'s - Default parameters **************************************************\n')

        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predictedGNB1), fmt='%d')
        f.write('\n')

        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predictedGNB1, zero_division=0))
        f.write('\n')

        f.write('a) Decision Tree - Default parameters **************************************************\n')

        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predictedBDT1), fmt='%d')
        f.write('\n')

        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predictedBDT1, zero_division=0))
        f.write('\n')

        f.write('a) Top Decision Tree - Criterion: gini or entropy'
                ' - Max Depth: 5 or 10 - min_samples_split: 2, 3, or 4\n')

        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predictedTDT1), fmt='%d')
        f.write('\n')

        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predictedTDT1, zero_division=0))
        f.write('\n')

        f.write('Best parameters found:\n')
        f.write(str(clfTDT1.best_params_))
        f.write('\n\n')

        f.write('a) Perceptron - Default parameters ********************************************************\n')

        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predictedP1), fmt='%d')
        f.write('\n')

        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predictedP1, zero_division=0))
        f.write('\n')

        f.write('a) Multilayer Perceptron - Default parameters ********************************************************\n')

        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predictedBMLP1), fmt='%d')
        f.write('\n')

        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predictedBMLP1, zero_division=0))
        f.write('\n')

        f.write('a) Top Multilayer Perceptron - Activation function: logistic, tanh, relu, or identity - '
                'Network architectures: (30, 50) or (10, 10, 10) - Solver: adam or sgd\n')

        f.write('b) Confusion Matrix\n')
        np.savetxt(f, confusion_matrix(test_y, predictedTMLP1), fmt='%d')
        f.write('\n')

        f.write('c) precision, recall, and F1 measure\n')
        f.write('d) accuracy, macro average F1, and weight average F1 of model\n')
        f.write(classification_report(test_y, predictedTMLP1, zero_division=0))
        f.write('\n')

        f.write('Best parameters found:\n')
        f.write(str(clfTMLP1.best_params_))
        f.write('\n')

        f.write('**********************************************************************************************\n\n')

        # Calculate averages
        f.write('Average accuracy of models (GNB, Base-DT, Top-DT, PER, Base-MLP, Top-MLP):\n')
        f.write(str((accuracy_GNB1 + accuracy_GNB2 + accuracy_GNB3 + accuracy_GNB4 + accuracy_GNB5 + accuracy_GNB6
                     + accuracy_GNB7 + accuracy_GNB8 + accuracy_GNB9 + accuracy_GNB10) / 10))
        f.write(', ')
        f.write(str((accuracy_BDT1 + accuracy_BDT2 + accuracy_BDT3 + accuracy_BDT4 + accuracy_BDT5 + accuracy_BDT6
                     + accuracy_BDT7 + accuracy_BDT8 + accuracy_BDT9 + accuracy_BDT10) / 10))
        f.write(', ')
        f.write(str((accuracy_TDT1 + accuracy_TDT2 + accuracy_TDT3 + accuracy_TDT4 + accuracy_TDT5 + accuracy_TDT6
                     + accuracy_TDT7 + accuracy_TDT8 + accuracy_TDT9 + accuracy_TDT10) / 10))
        f.write(', ')
        f.write(str((accuracy_P1 + accuracy_P2 + accuracy_P3 + accuracy_P4 + accuracy_P5 + accuracy_P6
                     + accuracy_P7 + accuracy_P8 + accuracy_P9 + accuracy_P10) / 10))
        f.write(', ')
        f.write(str((accuracy_BMLP1 + accuracy_BMLP2 + accuracy_BMLP3 + accuracy_BMLP4 + accuracy_BMLP5 + accuracy_BMLP6
                     + accuracy_BMLP7 + accuracy_BMLP8 + accuracy_BMLP9 + accuracy_BMLP10) / 10))
        f.write(', ')
        f.write(str((accuracy_TMLP1 + accuracy_TMLP2 + accuracy_TMLP3 + accuracy_TMLP4 + accuracy_TMLP5 + accuracy_TMLP6
                     + accuracy_TMLP7 + accuracy_TMLP8 + accuracy_TMLP9 + accuracy_TMLP10) / 10))
        f.write('\n\n')

        f.write('Average macro-average of models (GNB, Base-DT, Top-DT, PER, Base-MLP, Top-MLP):\n')
        f.write(str((macrof1_GNB1 + macrof1_GNB2 + macrof1_GNB3 + macrof1_GNB4 + macrof1_GNB5 + macrof1_GNB6
                     + macrof1_GNB7 + macrof1_GNB8 + macrof1_GNB9 + macrof1_GNB10) / 10))
        f.write(', ')
        f.write(str((macrof1_BDT1 + macrof1_BDT2 + macrof1_BDT3 + macrof1_BDT4 + macrof1_BDT5 + macrof1_BDT6
                     + macrof1_BDT7 + macrof1_BDT8 + macrof1_BDT9 + macrof1_BDT10) / 10))
        f.write(', ')
        f.write(str((macrof1_TDT1 + macrof1_TDT2 + macrof1_TDT3 + macrof1_TDT4 + macrof1_TDT5 + macrof1_TDT6
                     + macrof1_TDT7 + macrof1_TDT8 + macrof1_TDT9 + macrof1_TDT10) / 10))
        f.write(', ')
        f.write(str((macrof1_P1 + macrof1_P2 + macrof1_P3 + macrof1_P4 + macrof1_P5 + macrof1_P6
                     + macrof1_P7 + macrof1_P8 + macrof1_P9 + macrof1_P10) / 10))
        f.write(', ')
        f.write(str((macrof1_BMLP1 + macrof1_BMLP2 + macrof1_BMLP3 + macrof1_BMLP4 + macrof1_BMLP5 + macrof1_BMLP6
                     + macrof1_BMLP7 + macrof1_BMLP8 + macrof1_BMLP9 + macrof1_BMLP10) / 10))
        f.write(', ')
        f.write(str((macrof1_TMLP1 + macrof1_TMLP2 + macrof1_TMLP3 + macrof1_TMLP4 + macrof1_TMLP5 + macrof1_TMLP6
                     + macrof1_TMLP7 + macrof1_TMLP8 + macrof1_TMLP9 + macrof1_TMLP10) / 10))
        f.write('\n\n')

        f.write('Average weighted-average of models (GNB, Base-DT, Top-DT, PER, Base-MLP, Top-MLP):\n')
        f.write(str((weightedf1_GNB1 + weightedf1_GNB2 + weightedf1_GNB3 + weightedf1_GNB4
                     + weightedf1_GNB5 + weightedf1_GNB6 + weightedf1_GNB7 + weightedf1_GNB8
                     + weightedf1_GNB9 + weightedf1_GNB10) / 10))
        f.write(', ')
        f.write(str((weightedf1_BDT1 + weightedf1_BDT2 + weightedf1_BDT3 + weightedf1_BDT4
                     + weightedf1_BDT5 + weightedf1_BDT6 + weightedf1_BDT7 + weightedf1_BDT8
                     + weightedf1_BDT9 + weightedf1_BDT10) / 10))
        f.write(', ')
        f.write(str((weightedf1_TDT1 + weightedf1_TDT2 + weightedf1_TDT3 + weightedf1_TDT4
                     + weightedf1_TDT5 + weightedf1_TDT6 + weightedf1_TDT7 + weightedf1_TDT8
                     + weightedf1_TDT9 + weightedf1_TDT10) / 10))
        f.write(', ')
        f.write(str((weightedf1_P1 + weightedf1_P2 + weightedf1_P3 + weightedf1_P4
                     + weightedf1_P5 + weightedf1_P6 + weightedf1_P7 + weightedf1_P8
                     + weightedf1_P9 + weightedf1_P10) / 10))
        f.write(', ')
        f.write(str((weightedf1_BMLP1 + weightedf1_BMLP2 + weightedf1_BMLP3 + weightedf1_BMLP4
                     + weightedf1_BMLP5 + weightedf1_BMLP6 + weightedf1_BMLP7 + weightedf1_BMLP8
                     + weightedf1_BMLP9 + weightedf1_BMLP10) / 10))
        f.write(', ')
        f.write(str((weightedf1_TMLP1 + weightedf1_TMLP2 + weightedf1_TMLP3 + weightedf1_TMLP4
                     + weightedf1_TMLP5 + weightedf1_TMLP6 + weightedf1_TMLP7 + weightedf1_TMLP8
                     + weightedf1_TMLP9 + weightedf1_TMLP10) / 10))
        f.write('\n\n')

        # Calculate standard deviations
        accuracyGNB = np.array([accuracy_GNB1, accuracy_GNB2, accuracy_GNB3, accuracy_GNB4, accuracy_GNB5, accuracy_GNB6
                                , accuracy_GNB7, accuracy_GNB8, accuracy_GNB9, accuracy_GNB10])
        accuracyBDT = np.array([accuracy_BDT1, accuracy_BDT2, accuracy_BDT3, accuracy_BDT4, accuracy_BDT5, accuracy_BDT6
                                , accuracy_BDT7, accuracy_BDT8, accuracy_BDT9, accuracy_BDT10])
        accuracyTDT = np.array([accuracy_TDT1, accuracy_TDT2, accuracy_TDT3, accuracy_TDT4, accuracy_TDT5, accuracy_TDT6
                                , accuracy_TDT7, accuracy_TDT8, accuracy_TDT9, accuracy_TDT10])
        accuracyP = np.array([accuracy_P1, accuracy_P2, accuracy_P3, accuracy_P4, accuracy_P5, accuracy_P6
                              , accuracy_P7, accuracy_P8, accuracy_P9, accuracy_P10])
        accuracyBMLP = np.array([accuracy_BMLP1, accuracy_BMLP2, accuracy_BMLP3, accuracy_BMLP4
                                 , accuracy_BMLP5, accuracy_BMLP6, accuracy_BMLP7, accuracy_BMLP8
                                 ,accuracy_BMLP9, accuracy_BMLP10])
        accuracyTMLP = np.array([accuracy_TMLP1, accuracy_TMLP2, accuracy_TMLP3, accuracy_TMLP4
                                 , accuracy_TMLP5, accuracy_TMLP6, accuracy_TMLP7, accuracy_TMLP8
                                 , accuracy_TMLP9, accuracy_TMLP10])

        macrof1GNB = np.array([macrof1_GNB1, macrof1_GNB2, macrof1_GNB3, macrof1_GNB4, macrof1_GNB5, macrof1_GNB6
                               , macrof1_GNB7, macrof1_GNB8, macrof1_GNB9, macrof1_GNB10])
        macrof1BDT = np.array([macrof1_BDT1, macrof1_BDT2, macrof1_BDT3, macrof1_BDT4, macrof1_BDT5, macrof1_BDT6
                               , macrof1_BDT7, macrof1_BDT8, macrof1_BDT9, macrof1_BDT10])
        macrof1TDT = np.array([macrof1_TDT1, macrof1_TDT2, macrof1_TDT3, macrof1_TDT4, macrof1_TDT5, macrof1_TDT6
                               , macrof1_TDT7, macrof1_TDT8, macrof1_TDT9, macrof1_TDT10])
        macrof1P = np.array([macrof1_P1, macrof1_P2, macrof1_P3, macrof1_P4, macrof1_P5, macrof1_P6
                             , macrof1_P7, macrof1_P8, macrof1_P9, macrof1_P10])
        macrof1BMLP = np.array([macrof1_BMLP1, macrof1_BMLP2, macrof1_BMLP3, macrof1_BMLP4
                                , macrof1_BMLP5, macrof1_BMLP6, macrof1_BMLP7, macrof1_BMLP8
                                , macrof1_BMLP9, macrof1_BMLP10])
        macrof1TMLP = np.array([macrof1_TMLP1, macrof1_TMLP2, macrof1_TMLP3, macrof1_TMLP4
                                , macrof1_TMLP5, macrof1_TMLP6, macrof1_TMLP7, macrof1_TMLP8
                                , macrof1_TMLP9, macrof1_TMLP10])
        weightf1GNB = np.array([weightedf1_GNB1, weightedf1_GNB2, weightedf1_GNB3, weightedf1_GNB4
                                , weightedf1_GNB5, weightedf1_GNB6, weightedf1_GNB7, weightedf1_GNB8
                                , weightedf1_GNB9, weightedf1_GNB10])
        weightf1BDT = np.array([weightedf1_BDT1, weightedf1_BDT2, weightedf1_BDT3, weightedf1_BDT4
                                , weightedf1_BDT5, weightedf1_BDT6, weightedf1_BDT7, weightedf1_BDT8
                                , weightedf1_BDT9, weightedf1_BDT10])
        weightf1TDT = np.array([weightedf1_TDT1, weightedf1_TDT2, weightedf1_TDT3, weightedf1_TDT4
                                , weightedf1_TDT5, weightedf1_TDT6, weightedf1_TDT7, weightedf1_TDT8
                                , weightedf1_TDT9, weightedf1_TDT10])
        weightf1P = np.array([weightedf1_P1, weightedf1_P2, weightedf1_P3, weightedf1_P4
                              , weightedf1_P5, weightedf1_P6, weightedf1_P7, weightedf1_P8
                              , weightedf1_P9, weightedf1_P10])
        weightf1BMLP = np.array([weightedf1_BMLP1, weightedf1_BMLP2, weightedf1_BMLP3, weightedf1_BMLP4
                                 , weightedf1_BMLP5, weightedf1_BMLP6, weightedf1_BMLP7, weightedf1_BMLP8
                                 , weightedf1_BMLP9, weightedf1_BMLP10])
        weightf1TMLP = np.array([weightedf1_TMLP1, weightedf1_TMLP2, weightedf1_TMLP3, weightedf1_TMLP4
                                 , weightedf1_TMLP5, weightedf1_TMLP6, weightedf1_TMLP7, weightedf1_TMLP8
                                 , weightedf1_TMLP9, weightedf1_TMLP10])

        f.write('Standard Deviation of the accuracy of models (GNB, Base-DT, Top-DT, PER, Base-MLP, Top-MLP):\n')
        f.write(str(np.std(accuracyGNB)))
        f.write(', ')
        f.write(str(np.std(accuracyBDT)))
        f.write(', ')
        f.write(str(np.std(accuracyTDT)))
        f.write(', ')
        f.write(str(np.std(accuracyP)))
        f.write(', ')
        f.write(str(np.std(accuracyBMLP)))
        f.write(', ')
        f.write(str(np.std(accuracyTMLP)))
        f.write('\n\n')

        f.write('Standard Deviation of the macro-average of models (GNB, Base-DT, Top-DT, PER, Base-MLP, Top-MLP):\n')
        f.write(str(np.std(macrof1GNB)))
        f.write(', ')
        f.write(str(np.std(macrof1BDT)))
        f.write(', ')
        f.write(str(np.std(macrof1TDT)))
        f.write(', ')
        f.write(str(np.std(macrof1P)))
        f.write(', ')
        f.write(str(np.std(macrof1BMLP)))
        f.write(', ')
        f.write(str(np.std(macrof1TMLP)))
        f.write('\n\n')

        f.write('Standard Deviation of the weighted-average of models (GNB, Base-DT, Top-DT, PER, Base-MLP, Top-MLP):\n')
        f.write(str(np.std(weightf1GNB)))
        f.write(', ')
        f.write(str(np.std(weightf1BDT)))
        f.write(', ')
        f.write(str(np.std(weightf1TDT)))
        f.write(', ')
        f.write(str(np.std(weightf1P)))
        f.write(', ')
        f.write(str(np.std(weightf1BMLP)))
        f.write(', ')
        f.write(str(np.std(weightf1TMLP)))
        f.write('\n\n')

        f.close()
