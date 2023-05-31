import requests

url = 'http://localhost:5000/predict'
data = {
    "Pregnancies": 0,
    "Glucose": 137,
    "BloodPressure": 40,
    "SkinThickness": 35,
    "Insulin": 168,
    "BMI": 43.1,
    "DiabetesPedigreeFunction": 2.228,
    "Age": 33
}
headers = {'Content-type': 'application/json'}
response = requests.post(url, json=data, headers=headers)

print(response.json())
