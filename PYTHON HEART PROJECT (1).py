#!/usr/bin/env python
# coding: utf-8

# # Table Of Contents 
# * Libraries 
# * Dataset 
# * Exploratory Data Analysis 
# * Missing Data 
# * Split Train & Test 
# * Logistic Regression 
# * Metrics 
# 

# ## <a id='module1'><font color='green'>Libraries</font></a>

# In[1]:


# data analysis tools
import pandas as pd
import numpy as np

# importing ploting libraries
import matplotlib.pyplot as plt   

# importing seaborn for statistical plots
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Let us break the X and y dataframes into training set and test set. For this we will use
# Sklearn package's data splitting function which is based on random function
from sklearn.model_selection import train_test_split

# model implementation
from sklearn.linear_model import LogisticRegression

# calculate accuracy measures and confusion matrix
from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
# To enable plotting graphs in Jupyter notebook
# %matplotlib inline  # !pip install imblearn


# # <a id='module2'><font color='green'>Dataset</font></a>

# In[2]:


Data = pd.read_csv("heart.csv")
Data


# # <a id='module3'><font color='green'>Exploratory Data Analysis</font></a>

# In[3]:


Data.describe()


# In[4]:


Data.shape


# In[5]:


Data.size


# In[6]:


Data.info()


# In[7]:


Data.isnull().sum()


# In[8]:


Data.nunique()


# In[9]:


Data.duplicated().sum()


# In[10]:


Data.drop_duplicates(inplace = True)


# In[11]:


Data.duplicated().sum()


# In[12]:


Data.head(10)


# In[13]:


# Let us check whether any of the columns has any value other than numeric i.e. data is not corrupted such as a "?" instead of a number.
Data[~Data.applymap(np.isreal).all(1)]


# # Observations
# 
# * we use np.isreal a numpy function which checks each column for each row and returns a bool array,
# * where True if input element is real.
# * applymap is pandas dataframe function that applies the np.isreal function columnwise
# * Following line selects those rows which have some non-numeric value in any of the columns hence the ~ symbol

# In[14]:


# Pairplot using sns
sns.pairplot(Data , hue='target' , diag_kind = 'kde')


# In[15]:


Data.isnull().sum()


# ### Standardize the Data

# In[16]:


scaler = StandardScaler()
scaler.fit(Data.drop("target", axis =1))


# In[17]:


heart_scaled = scaler.transform(Data.drop("target", axis = 1))
heart_scaled = pd.DataFrame(heart_scaled, columns = Data.columns[:-1])


# In[18]:


X = heart_scaled
y = Data["target"]


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[20]:


Data.columns


# In[21]:


X = Data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = Data['target']


# In[ ]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[23]:


# Fit the model on original data i.e. before upsampling
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))
print(metrics.classification_report(y_test, y_predict))


# In[24]:


cm = metrics.confusion_matrix(y_test, y_predict)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Heart attack possiblity', 'Heart attack not possibility']
plt.title('Confusion Matrix - Test Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['G1', 'G2'], ['G1','G2']]
 
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

