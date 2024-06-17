import pandas as pd
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

import pickle 

def get_clean_data():
  data = pd.read_csv("dataset/data.csv")
  data = data.drop(['Unnamed: 32','id'],axis=1)
  data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
  return data 

def create_model(data):
  X = data.drop(['diagnosis'],axis=1)
  y = data['diagnosis']
  
  # scale the data 
  scalar =  StandardScaler()
  X = scalar.fit_transform(X)
  
  # split the data
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
  
  # train the model
  model = LogisticRegression()
  model.fit(X_train,y_train)
  
  # test the model
  y_pred = model.predict(X_test)
  print(f"Acurracy Model: {accuracy_score(y_test,y_pred)}")
  print(f"Classification Report:\n {classification_report(y_test,y_pred)}")
  return model,scalar


  

def main():
  ...
  data = get_clean_data()

  model,scalar = create_model(data)
  
  # export our ML model as binary file (why).
  with open('model/model.pkl','wb') as file:
    pickle.dump(model,file)
    
  with open('model/scalar.pkl','wb') as file:
    pickle.dump(scalar,file)
    
 
  
if __name__ == '__main__':
  main()