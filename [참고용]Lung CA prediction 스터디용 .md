# Lung CA

학습일: 2025/09/19
선택: 프로젝트 진행중

| Contesnts | Contents |
| --- | --- |
| [Importing Libraries](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#1) | Correlation Heatmap |
| [About Dataset](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#2) | [Preprocessing For Classification](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#13) |
| [Basic Exploration](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#3) | [Logistic Regression Model](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#14) |
| [Dataset Summary](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#4) | [Gaussian Naive Bayes Model](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#15) |
| [Digging Deeper](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#5) | [Bernoulli Naive Bayes Model](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#16) |
| [Custom Palette For Visualization](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#6) | [Support Vector Machine Model](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#17) |
| [Positive Lung Cancer Cases](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#7) | [Random Forest Model](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#18) |
| [Positive Cases' Age Distribution](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#8) | [K Nearest Neighbors Model](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#19) |
| [Positive Cases' Gender Distribution](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#9) | [Extreme Gradient Boosting Model](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#20) |
| [Gender-wise Positive Cases' Reasons](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#10) | [Neural Network Architecture](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#21) |
| [Gender-wise Positive Cases' Symptoms](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4#11) |  |
- **GENDER :** M [Male] , F [Female]
- **AGE :** Age of patients
- **SMOKING :** 2 [Yes] , 1 [No]
- **YELLOW_FINGERS :** 2 [Yes] , 1 [No]
- **ANXIETY :** 2 [Yes] , 1 [No]
- **PEER_PRESSURE :** 2 [Yes] , 1 [No]
- **CHRONIC DISEASE :** 2 [Yes] , 1 [No]
- **FATIGUE :** 2 [Yes] , 1 [No]
- **ALLERGY :** 2 [Yes] , 1 [No]
- **WHEEZING :** 2 [Yes] , 1 [No]
- **ALCOHOL CONSUMING :** 2 [Yes] , 1 [No]
- **COUGHING :** 2 [Yes] , 1 [No]
- **SHORTNESS OF BREATH :** 2 [Yes] , 1 [No]
- **SWALLOWING DIFFICULTY :** 2 [Yes] , 1 [No]
- **CHEST PAIN :** 2 [Yes] , 1 [No]
- **LUNG_CANCER :** YES [Positive] , NO [Negative]

---

https://www.kaggle.com/code/sandragracenelson/lung-cancer-prediction

```python
#Importing Librariesimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#For ignoring warningimport warnings
warnings.filterwarnings("ignore")

df.shape #(309,16)

#Checking for Duplicates
df.duplicated().sum() # 33

#Removing Duplicates
df=df.drop_duplicates()

#Checking for null values
df.isnull().sum()

df.info() #

df.describe() #
```

```python
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['GENDER']=le.fit_transform(df['GENDER'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
df['SMOKING']=le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY']=le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
df['WHEEZING']=le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING']=le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])

#Let's check what's happened now
df
```

### **Male=1 & Female=0. Also for other variables, YES=1 & NO=0**

```python
df.infor()
```

![스크린샷 2025-09-20 15.25.46.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-09-20_15.25.46.png)

```
#Let's check the distributaion of Target variable.sns.countplot(x='LUNG_CANCER', data=df,)
plt.title('Target Distribution');

df['LUNG_CANCER'].value_counts()  # [1    238], [0     38]
																	# Name: LUNG_CANCER, dtype: int64
```

***We will handle this imbalance before applyig algorithm.**

**Now let's do some Data Visualizations for the better understanding of how the independent features are related to the target variable..**

```
# function for plottingdef plot(col, df=df):
    return df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))
```

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image.png)

```python
plot('AGE')
plot('SMOKING')
plot('YELLOW_FINGERS')
plot('ANXIETY')
```

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%201.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%202.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%203.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%204.png)

```python
plot('CHRONIC DISEASE')
plot('FATIGUE ')
plot('ALLERGY ')
plot('WHEEZING')

```

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%205.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%206.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%207.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%208.png)

```python
plot('ALCOHOL CONSUMING')
plot('COUGHING')
plot('SHORTNESS OF BREATH')
plot('SWALLOWING DIFFICULTY')
```

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%209.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%2010.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%2011.png)

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%2012.png)

```python
plot('CHEST PAIN')
```

![image.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/image%2013.png)

**From the visualizations, it is clear that in the given dataset, the features GENDER, AGE, SMOKING and SHORTNESS OF BREATH don't have that much relationship with LUNG CANCER. So let's drop those features to make this dataset more clean.**

```python
df_new=df.drop(columns=['GENDER','AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
df_new
```

![스크린샷 2025-09-20 15.30.44.png](Lung%20CA%2026fd44a085da80088616ed3511c422f9/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-09-20_15.30.44.png)

```python
#Finding Correlation
cn=df_new.corr()
cn
```

```python
#Correlation 
cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap=cmap,annot=True, square=True)
plt.show()
```

```
kot = cn[cn>=.40]
plt.figure(figsize=(12,8))
sns.heatmap(kot, cmap="Blues")
```
