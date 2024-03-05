import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df= pd.read_csv('static/teleCust1000t.csv')
print(df.head())
print(df['custcat'].value_counts())

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float) its out input data

y= df['custcat'].values # targer value(classes we want 2 predict)
y[0:5] # 5 rows from y

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=4)

print('Train set: ', X_train.shape, y_train.shape)
print('Test set: ', X_test.shape, y_test.shape)

k = 10
neigh = KNeighborsClassifier(n_neighbors= k).fit(X_train, y_train)
yhat= neigh.predict(X_test)
print(yhat[0:5])


from sklearn import metrics
Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1* std_acc, mean_acc+1*std_acc, alpha=0.10)
plt.fill_between(range(1, Ks), mean_acc - 3* std_acc, mean_acc+3*std_acc, alpha=0.10, color = 'green')
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (k)')
plt.tight_layout()


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Class')
plt.scatter(range(len(y_test)), yhat, color='red', label='Predicted Class')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('KNN Predicted vs. Actual Classes')
plt.legend()
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 