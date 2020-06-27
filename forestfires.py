



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

fire = pd.read_csv("D:\\excelR\\Data science notes\\Neural networks\\asgmnt\\forestfires.csv")
fire.info()
fire.describe()
fire.shape
fire.isnull().sum()  # No missing values 
fire.columns
fire.head()
fire.dtypes
fire.drop_duplicates(keep='first', inplace=True) # 8 duplicate rows are there
fire.shape
fire=fire.drop(['month','day','dayfri','daymon','daysat','daysun','daythu','daytue','daywed','monthapr','monthaug','monthdec','monthfeb','monthjan','monthjul','monthjun','monthmar','monthmay','monthnov','monthoct','monthsep'], axis=1)
fire['size_category'].value_counts()
# Measure of Dispersion
np.var(fire)
np.std(fire)
# Skewness and Kurtosis
skew(fire.drop('size_category',axis=1))
kurtosis(fire.drop('size_category',axis=1))

# HIStogram

plt.hist(fire['FFMC']);plt.xlabel('FFMC');plt.ylabel('Frequency');plt.title('Histogram of FFMC')
plt.hist(fire['DMC']);plt.xlabel('DMC');plt.ylabel('Frequency');plt.title('Histogram of DMC')
plt.hist(fire['DC']);plt.xlabel('DC');plt.ylabel('Frequency');plt.title('Histogram of DC')
plt.hist(fire['ISI']);plt.xlabel('State');plt.ylabel('Frequency');plt.title('Histogram of ICI')
plt.hist(fire['temp']);plt.xlabel('temp');plt.ylabel('Frequency');plt.title('Histogram of temp')
plt.hist(fire['RH']);plt.xlabel('RH');plt.ylabel('Frequency');plt.title('Histogram of RH')
plt.hist(fire['wind']);plt.xlabel('wind');plt.ylabel('Frequency');plt.title('Histogram of wind')
plt.hist(fire['rain']);plt.xlabel('rain');plt.ylabel('Frequency');plt.title('Histogram of rain')
plt.hist(fire['area']);plt.xlabel('area');plt.ylabel('Frequency');plt.title('Histogram of area')

# Normal Q-Q plot
plt.plot(fire.drop('size_category', axis=1));plt.legend(list(fire.columns))

ffmc = np.array(fire['FFMC'])
dmc = np.array(fire['DMC'])
dc = np.array(fire['DC'])
isi = np.array(fire['ISI'])
temp= np.array(fire['temp'])
rh=np.array(fire['RH'])
wind=np.array(fire['wind'])
rain=np.array(fire['rain'])
area=np.array(fire['area'])

from scipy import stats

stats.probplot(ffmc, dist='norm', plot=plt);plt.title('Probability Plot of FFMC')
stats.probplot(dmc, dist='norm', plot=plt);plt.title('Probability Plot of DMC')
stats.probplot(dc, dist='norm', plot=plt);plt.title('Probability Plot of Dc')
stats.probplot(isi, dist='norm', plot=plt);plt.title('Probability Plot of ISI')
stats.probplot(temp, dist='norm', plot=plt);plt.title('Probability Plot of temp')
stats.probplot(rh, dist='norm', plot=plt);plt.title('Probability Plot of RH')
stats.probplot(wind, dist='norm', plot=plt);plt.title('Probability Plot of wind')
stats.probplot(rain, dist='norm', plot=plt);plt.title('Probability Plot of rain')
stats.probplot(area, dist='norm', plot=plt);plt.title('Probability Plot of area')

## Boxplots

# Boxplot 
sns.boxplot(fire['FFMC'],orient='v').set_title('Boxplot of FFMC')
sns.boxplot(fire['DMC'], orient='v', color='coral').set_title('Boxplot of DMC')
sns.boxplot(fire['DC'], orient='v', color='skyblue').set_title('Boxplot of DC')
sns.boxplot(fire['ISI'], orient='v', color='orange').set_title('Boxplot of ISI')
sns.boxplot(fire['temp'], orient='v', color='teal').set_title('Boxplot of temp')
sns.boxplot(fire['RH'], orient='v', color='brown').set_title('Boxplot of RH')
sns.boxplot(fire['wind'], orient='v', color='violet').set_title('Boxplot of wind')
sns.boxplot(fire['rain'], orient='v', color='purple').set_title('Boxplot of rain')
sns.boxplot(fire['area'], orient='v', color='lightgreen').set_title('Boxplot of area')

sns.pairplot(fire)
# Heatmap
corr = fire.corr()
sns.heatmap(corr, annot=True)

# Convert the categorical to binary class 
# small=0 and large = 1
fire.loc[fire['size_category'] == 'small', 'size_category'] = 0
fire.loc[fire['size_category'] == 'large', 'size_category'] = 1

fire.size_category.value_counts()
X = fire.drop(["size_category"],axis=1)
Y =fire["size_category"]
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
mlp = MLPClassifier(hidden_layer_sizes=(9,9))

mlp.fit(trainX,trainY)
prediction_train=mlp.predict(trainX)
prediction_train
prediction_test = mlp.predict(testX)
prediction_test

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(testY,prediction_test))
np.mean(testY==prediction_test)
np.mean(trainY==prediction_train)

