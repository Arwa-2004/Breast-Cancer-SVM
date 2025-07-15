#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[5]:


cancer = load_breast_cancer()


# In[6]:


cancer.keys()


# In[7]:


print(cancer['DESCR'])


# In[8]:


df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])


# In[9]:


df_feat.info()


# In[14]:


cancer['target']


# In[12]:


cancer['target_names']


# In[13]:


# EDA


# In[16]:


df_feat.head()


# In[17]:


df_feat.info()


# In[42]:


# Visualize the distribution of classes
sns.countplot(x=cancer['target'])
plt.xticks([0, 1], cancer['target_names'])


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X= df_feat
y= cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[28]:


from sklearn.svm import SVC


# In[29]:


model = SVC()


# In[30]:


model.fit(X_train, y_train)


# In[31]:


predictions = model.predict(X_test)


# In[33]:


from sklearn.metrics import classification_report, confusion_matrixmatrix


# In[34]:


print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


# In[36]:


from sklearn.model_selection import GridSearchCV


# In[37]:


param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid.best_params_)

# Evaluate again
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))


# In[ ]:




