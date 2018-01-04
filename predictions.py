from analysis import generate_bar_categorical
from analysis import generate_distplots_continuous
from analysis import last_call_duration_kde
from analysis import boxplot_continuous

from featurization import featurization_dataset
from featurization import category_ohe
from featurization import standardization_train_test_split
from featurization import tsne_embeddings
from featurization import model
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier 

def plots(data):
	# Set 1 to generate graphs and display them
	generate_bar_categorical(data,0)
	generate_distplots_continuous(data,0)
	boxplot_continuous(data,0)
	last_call_duration_kde(data,0)

def feature_creation(data):
	featurization_dataset(data)
	category_ohe(data)
	dataset = standardization_train_test_split(data)

	return dataset




def main():
	data = pd.read_csv('marketing-data.csv')

	#Generates all plots and stores in the same folder
	plots(data)
	dataset = feature_creation(data)
	
	x_train = dataset[0]
	x_test = dataset[1]
	y_train = dataset[2]
	y_test = dataset[3]

	#Instead of 0 use 1 for generating 2-D tSNE embedding graph(takes time to generate)
	if(0):
		tsne_embeddings(x_train)

	#Modelling

	#Logistic Regression
	clf = SGDClassifier(loss = 'log',alpha = 0.001,penalty = 'l2',class_weight = 'balanced')
	model(x_train,y_train,x_test,y_test,clf,'logistic')
	print "\n"
	#Linear SVM
	clf = SGDClassifier(loss = 'hinge',alpha = 0.0003,penalty = 'l1',class_weight = 'balanced')
	model(x_train,y_train,x_test,y_test,clf,'SVM')
	print "\n"
	#Random Forest Ensemble
	clf = RandomForestClassifier(n_estimators=12,class_weight = 'balanced',criterion = 'gini',\
	                            min_samples_split=4) 
	model(x_train,y_train,x_test,y_test,clf,'Random Forest')


if __name__ == "__main__":
	main()