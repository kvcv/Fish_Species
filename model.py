import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('Fish.csv')
#print (df)

X=df.iloc[:,1:7]
y=df.iloc[:,0]
#print(y)
#print(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Implement Random Forest classifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

score=accuracy_score(y_test,y_pred)

print(score)

# Create a Pickle file
import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

#print(classifier.predict([[1,1,3,1,1,3]]))