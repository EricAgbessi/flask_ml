import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

print("1")
url = "https://drive.google.com/file/d/1myIDkOCxSdN1nlRygkgz_RPoukKP0wzO/view?usp=share_link"
file_id = url.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?id=' + file_id
df = pd.read_csv(dwn_url)
print("finish")
#df = pd.read_csv(dwn_url)
df.drop_duplicates(inplace=True)

## Do Label Encoding for gender column
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

print("2")

# Convertir l'historique de tabagisme au format numérique
smoking_history_mapping = {'never': 0, 'No Info': -1, 'current': 2, 'former': 1, 'ever': 2, 'not current': 0}
df['smoking_history'] = df['smoking_history'].map(smoking_history_mapping)



df['age'] = df['age'].astype(int)



X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


print(X_train.shape)

from sklearn import preprocessing
import joblib

stand = preprocessing.StandardScaler()
X_train = stand.fit_transform(X_train)
X_test = stand.transform(X_test)

joblib.dump(stand, 'scaler.pkl')

print("3")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfc.fit(X_train, y_train)
print("4")

# Evaluate the model
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))



joblib.dump(rfc, 'model.pkl')

import joblib
joblib.dump(rfc, 'model.pkl')
print("Model dumped!")


rfc = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(df.drop("diabetes",axis=1).columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")


person_X = stand.transform([[1,23.0, 0,0,0,22.9,5.4, 108]])

print(rfc.predict(person_X)) #4th patient


print("finish")


person_predict = rfc.predict(person_X)
result=person_predict
print(result)

