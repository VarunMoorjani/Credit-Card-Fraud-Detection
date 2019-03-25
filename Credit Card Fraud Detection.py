#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing necessary libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('/Users/varunmoorjani/Downloads/creditcard.csv')
columns=df.columns
# The labels are in the last column ('Class'). Simply remove it to obtain features columns
features_columns=columns.delete(len(columns)-1)
features=df[features_columns]
labels=df['Class']


# In[3]:


print('The shape of the file is {}'.format(df.shape))
df.head()


# ## Visualizing the Dataset

# In[4]:


count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[5]:


plt.figure(figsize=(10,10))
pd.Series(df["Class"]).value_counts().plot(kind = "pie" , title = "Class" , autopct='%.2f')
plt.show()


# ## Visualizing " Number of Transactions VS Time(Secs)"

# In[6]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 50

ax1.hist(df.Time[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()


# ## Visualizing the Variance shown by different features via Histogram

# In[7]:


_features = df.ix[:,1:29].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(df[_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()


# ## I dropped these features as they showed similar variation and therefore I found that there is no point in keeping them.

# In[8]:


df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)


# In[9]:


df.head(5)


# In[10]:


features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels, 
                                                                            test_size=0.1, 
                                                                            random_state=1)


# ## In order to counter the imbalance which is present in the dataset, I used the approach of oversampling via SMOTE

# In[11]:


print('The count of the data before sampling {}'.format(labels_train.value_counts()))
oversampler=SMOTE(random_state=1)
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
print('The count of 0 is {} and 1 is {}'.format(len(os_labels[os_labels==0]),len(os_labels[os_labels==1])))


# ## Statistical Modelling using Random Forest and Logistic Regression

# In[12]:


## Random Forest


# In[13]:


clf=RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1)
clf.fit(os_features,os_labels)
actual=labels_test
predictions=clf.predict(features_test)


# In[14]:


#confusion_matrix(actual,predictions)
conf_mat = confusion_matrix(actual,predictions)
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(conf_mat, annot=True, fmt='d')
            #xticklabels=data.property_type.values, yticklabels=data.property_type.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[15]:


print(classification_report(actual,predictions))


# In[16]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print ('The roc_auc score is {}'.format(roc_auc))


# In[17]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[18]:


## Logistic Regression
lr = LogisticRegression()
lr.fit(os_features,os_labels)
actual = labels_test
predictions = lr.predict(features_test)


# In[19]:


conf_mat = confusion_matrix(actual,predictions)
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(conf_mat, annot=True, fmt='d')
            #xticklabels=data.property_type.values, yticklabels=data.property_type.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[20]:


print(classification_report(actual,predictions))


# In[21]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print ('The roc_auc score is {}'.format(roc_auc))


# In[22]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

