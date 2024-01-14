import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = pd.read_csv('D:/ML/data.csv')

X = data[['Area', 'Rooms', 'Locality']]
y = data['Price']

X = X.copy()
imputer = SimpleImputer(strategy='mean')

le = LabelEncoder()
X['Locality'] = le.fit_transform(X['Locality'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train_imputed, y_train)

y_pred = model.predict(X_test_imputed)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse} \n')

model.feature_names_out_ = X_train_imputed.columns.tolist()


locality = input("Enter the locality in Chennai: ")
rooms = input("Enter the number of bedrooms: ")
area = float(input("Enter the area in square feet: "))

try:
    locality_encoded = le.transform([locality])
except ValueError:
    locality_encoded = [-1] 
new_data_point = pd.DataFrame([[area, rooms, locality_encoded[0]]], columns=X_train_imputed.columns)

predicted_price = model.predict(new_data_point)

print(f'Predicted Price : {predicted_price[0]:,.2f}','lakhs')