

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

strt = pd.read_csv("D:\\excelR\\Data science notes\\Neural networks\\asgmnt\\50_Startups.csv")
strt.info()
strt.describe()
strt.shape
strt.isnull().sum()  # No missing values 
strt.columns
strt.head()
strt.dtypes
strt.drop_duplicates(keep='first', inplace=True) # 
strt.shape

# Measure of Dispersion
np.var(strt)
np.std(strt)


# HIStogram

plt.hist(strt['R&D Spend']);plt.xlabel('R&D Spend');plt.ylabel('Frequency');plt.title('Histogram of R&D Spend')
plt.hist(strt['Administration']);plt.xlabel('Administration');plt.ylabel('Frequency');plt.title('Histogram of Administration')
plt.hist(strt['Marketing Spend']);plt.xlabel('Marketing Spend');plt.ylabel('Frequency');plt.title('Histogram of Marketing Spend')
plt.hist(strt['State']);plt.xlabel('State');plt.ylabel('Frequency');plt.title('Histogram of State')
plt.hist(strt['Profit']);plt.xlabel('Profit');plt.ylabel('Frequency');plt.title('Histogram of Profit')

# Barplot
sns.countplot(strt['State']).set_title('Count of State')
# Normal Q-Q plot
plt.plot(strt.drop('State',axis=1));plt.legend(['R&D Spend', 'Administration', 'Marketing Spend', 'Profit'])

RD = np.array(strt['R&D Spend'])
Admin = np.array(strt['Administration'])
MS = np.array(strt['Marketing Spend'])
profit = np.array(strt['Profit'])

from scipy import stats

stats.probplot(RD, dist='norm', plot=plt);plt.title('Probability Plot of RD')
stats.probplot(Admin, dist='norm', plot=plt);plt.title('Probability Plot of Admin')
stats.probplot(MS, dist='norm', plot=plt);plt.title('Probability Plot of MS')
stats.probplot(profit, dist='norm', plot=plt);plt.title('Probability Plot of profit')

## Boxplots
sns.boxplot(strt['R&D Spend'], orient='h', color='blue').set_title('Boxplot of RD')
sns.boxplot(strt['Administration'], orient='h', color='yellow').set_title('Boxplot of Admin')
sns.boxplot(strt['Marketing Spend'], orient='v', color='skyblue').set_title('Boxplot of MS')
sns.boxplot(strt['Profit'], orient='v', color='orange').set_title('Boxplot of profit')

sns.pairplot(strt)
# Heatmap
corr = strt.corr()
sns.heatmap(corr, annot=True)


# Creating Dummy variables for State
strt = pd.get_dummies(strt)
X = strt.drop(["Profit"],axis=1)
Y =strt["Profit"]
X=X.astype('int')
Y=Y.astype('int')

# Splitting the data
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(X,Y)

# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(3,3))

mlp.fit(trainX,trainY)
prediction_train=mlp.predict(trainX)
prediction_train
prediction_test = mlp.predict(testX)
prediction_test

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(testY,prediction_test))
np.mean(testY==prediction_test)
np.mean(trainY==prediction_train)

