#some things r self explanatory but still, i write comments.Anyone that reads can understand.

#importing dependencies
import pandas as pd
imort numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear.model import LogisticRegression
from sklearn.metricsimport accuracy_score

#loading dataset to a Panda DataFrame [/home/kamogelo/Downloads/project_10_credit_card_fraud_detection.py]
#credit_card_data = name for data
credit_card_data = pd.read_csv('/home/kamogelo/Downloads/project_10_credit_card_fraud_detection.py')

#head = Return first DataFrame rows
credit_card_data.head()

#tail = Return last DataFrame rows
credit_card_data.tail()

#info = Info on DataFrame
credit_card_data.info()

#is null = missing vales in each column
#sum , if iterable is empty,retrn start value
credit_card_data.isnull().sum()

#distribution pf legit and fraudulent transactions (2 output)
#value_counts()
credit_card_data['Class'].value_counts()

#DATASET is highly unbalanced
#output was:
# 0 = Normal Transactions(284315)
 # 1 = Fraudulent Transactions (492)
 
 #Seperating the two groups for further analysis.
 legit = credit_card_data[credit_card_data.Class  == 0]
 fraud = credit_card_data[credit_card_data.Class  == 1]
 
#shape =
print(legit.shape)
print(fraud.shape)
 
#describe legit card data
legit.Amount.describe()
 
#describe fraud card data
fraud.Amount.describe()
 
 #compare values from legit & fraud
 credit_card_data.groupby('Class').mean()
 
 
 #Under Sampling
 #Build a sample dataset containing similar distribution of normal transactions and fraudulent transactions
#Number of fraudulent transactoions -> 492

#sample = return sample of items 
legit_sample = legit.sample(n= 492)

#Concatening Two DataFrames
new_dataset = pd.concat([legit_sample, fraud] , axis = 1)

new_dataset.head()

new_dataset.tail()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

 #Splitting the data into Features and Targets
 #drop = 
 X = new_dataset.drop(columns = 'Class', axis = 1)
 Y = new_dataset['Class']
 
 print(X)
 print(Y)
 
 #Split the data into Training data and Testing data
 
 X-train , X_test , Y_train , = train_test_split(X,Y, test_size= 0.2, stratify, random_state=2)
 
 #output i (984,30) (787,30) (197,30)
 print(X_shape, X_test_shape, X_test_shape)
 
 #Model Training
 #Logistic Regression
 
 model = LogisticRegression()
 
 #training the logistic regression model with traing data
 model.fit(X_train,Y_train)
 
 #Model Evaluation
 #Accuracy Score
 
 #accuracy on training data
 X_train_predication = model.predict(X_train)
 training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
 
 print('Accuracy on training data: ',  training_data_accuracy )
 
#accuracy on test data
 X_test_predication = model.predict(X_test)
 test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
 
  print('Accuracy on testing data: ',  test_data_accuracy )
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 





 
 
 
 
 
 
 
