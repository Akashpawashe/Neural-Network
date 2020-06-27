

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

conc = pd.read_csv("D:\\excelR\\Data science notes\\Neural networks\\asgmnt\\concrete.csv")
conc.info()
conc.describe()
conc.shape
conc.isnull().sum()  # No missing values 
conc.columns
conc.head()
conc.dtypes
conc.drop_duplicates(keep='first', inplace=True) # 25 duplicate rows are there
conc.shape
# Measure of Dispersion
np.var(conc)
np.std(conc)
# Skewness and Kurtosis
skew(conc)
kurtosis(conc)

# HIStogram

plt.hist(conc['cement']);plt.xlabel('cement');plt.ylabel('Frequency');plt.title('Histogram of cement')
plt.hist(conc['slag']);plt.xlabel('slag');plt.ylabel('Frequency');plt.title('Histogram of slag')
plt.hist(conc['ash']);plt.xlabel('ash');plt.ylabel('Frequency');plt.title('Histogram of ash')
plt.hist(conc['water']);plt.xlabel('water');plt.ylabel('Frequency');plt.title('Histogram of water')
plt.hist(conc['superplastic']);plt.xlabel('superplastic');plt.ylabel('Frequency');plt.title('Histogram of superplastic')
plt.hist(conc['coarseagg']);plt.xlabel('coarseagg');plt.ylabel('Frequency');plt.title('Histogram of coarseagg')
plt.hist(conc['fineagg']);plt.xlabel('fineagg');plt.ylabel('Frequency');plt.title('Histogram of fineagg')
plt.hist(conc['age']);plt.xlabel('age');plt.ylabel('Frequency');plt.title('Histogram of age')
plt.hist(conc['strength']);plt.xlabel('strength');plt.ylabel('Frequency');plt.title('Histogram of strength')
 
# Normal Q-Q plot
plt.plot(conc);plt.legend(list(conc.columns))


cement = np.array(conc['cement'])
slag = np.array(conc['slag'])
ash = np.array(conc['ash'])
water = np.array(conc['water'])
supplst = np.array(conc['superplastic'])
coaragg = np.array(conc['coarseagg'])
fineagg = np.array(conc['fineagg'])
age = np.array(conc['age'])
strength = np.array(conc['strength'])

from scipy import stats

stats.probplot(cement, dist='norm', plot=plt);plt.title('Probability Plot of Cement')
stats.probplot(slag, dist='norm', plot=plt);plt.title('Probability Plot of Slag')
stats.probplot(ash, dist='norm', plot=plt);plt.title('Probability Plot of Ash')
stats.probplot(water, dist='norm', plot=plt);plt.title('Probability Plot of Water')
stats.probplot(supplst, dist='norm', plot=plt);plt.title('Probability Plot of Superplastic')
stats.probplot(coaragg, dist='norm', plot=plt);plt.title('Probability Plot of Coarseagg')
stats.probplot(fineagg, dist='norm', plot=plt);plt.title('Probability Plot of Fineagg')
stats.probplot(age, dist='norm', plot=plt);plt.title('Probability Plot of Age')
stats.probplot(strength, dist='norm', plot=plt);plt.title('Probability Plot of Strength')

## Boxplots
sns.boxplot(conc['cement'])
sns.boxplot(conc['slag'], orient='h', color='coral').set_title('Boxplot of Slag')
sns.boxplot(conc['ash'], orient='v', color='skyblue').set_title('Boxplot of Ash')
sns.boxplot(conc['water'], orient='v', color='orange').set_title('Boxplot of Water')
sns.boxplot(conc['superplastic'], orient='v', color='red').set_title('Boxplot of SuperPlastic')
sns.boxplot(conc['coarseagg'], orient='v', color='brown').set_title('Boxplot of Coarseagg')
sns.boxplot(conc['fineagg'], orient='v', color='violet').set_title('Boxplot of Fineagg')
sns.boxplot(conc['age'], orient='v', color='purple').set_title('Boxplot of Age')
sns.boxplot(conc['strength'], orient='v', color='lightgreen').set_title('Boxplot of Strength')


sns.pairplot(conc)
# Heatmap
corr = conc.corr()
sns.heatmap(corr, annot=True)

X = conc.drop(["strength"],axis=1)
Y =conc["strength"]
X=X.astype('int')
Y=Y.astype('int')
conc.strength.value_counts()
# Splitting the data
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(X,Y)

# Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30))

mlp.fit(trainX,trainY)
prediction_train=mlp.predict(trainX)
prediction_train
prediction_test = mlp.predict(testX)
prediction_test


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(testY,prediction_test))
np.mean(testY==prediction_test)
np.mean(trainY==prediction_train)
