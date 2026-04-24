import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#loading and inspecting the data
df = pd.read_csv('dataset.csv')

print(df.head())

print(df['FaultClass'].value_counts())  # check class balance
print(df.isnull().sum())               # check for missing values

#seperating the feature cols and severity cols as the severity cols wont be used for the ML training
feature_cols = ['fpeak', 'pLow', 'pMid', 'pHigh', 'pKurtosis', 'qMean', 'qVar', 'qSkewness', 'qKurtosis', 'qPeak2Peak', 'qCrest', 'qRMS', 'qMAD', 'qCSRange']
severity_cols = ['LeakFault', 'BlockingFault','BearingFault', 'FaultClass']

X = df[feature_cols]
y = df['FaultClass']

#scaling the values
scaler = StandardScaler()
X_scaled=scaler.fit_transform(X)

#spilting into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, train_size=0.8, stratify=y, random_state=42)

#loading the model
model=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print(accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True))