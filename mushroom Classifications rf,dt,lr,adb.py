#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[3]:


data=pd.read_csv("mushrooms.csv")


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='class')
plt.title("Count of Mushroom Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# In[ ]:





# 
# 
# 
# 
# # Feature Engineering
# 

# In[8]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in data.columns:
    data[col] = label_encoder.fit_transform(data[col])


# In[9]:


# Split the data into features (X) and labels (y)
X = data.drop('class', axis=1)
y = data['class']


# # Model Building

# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


# In[35]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[36]:


X_train


# In[37]:


y_train


# In[38]:


# Creating and training a Random Forest classifier
rf_clf = RandomForestClassifier(max_depth = 5)
rf_clf.fit(X_train, y_train)


# In[39]:


y_pred = rf_clf.predict(X_test)


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[41]:


feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_clf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)


# In[42]:


print("Top 5 important features:")
print(feature_importances.head(5))


# In[43]:


# Creating and training a Logistic Regression classifier
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)


# In[44]:



# Creating and training a Decision Tree classifier
dt_clf = DecisionTreeClassifier(random_state = 0 , max_depth = 5)
dt_clf.fit(X_train, y_train)


# In[ ]:





# In[45]:


adb_clf = AdaBoostClassifier(base_estimator= dt_clf, n_estimators= 400, learning_rate = 1)


# In[46]:


adb_clf.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# # Model Testing

# In[47]:


from sklearn.metrics import accuracy_score


# In[48]:


# Making predictions on the test set using each classifier
rf_predictions = rf_clf.predict(X_test)
lr_predictions = lr_clf.predict(X_test)
dt_predictions = dt_clf.predict(X_test)
adb_predictions= adb_clf.predict(X_test)


# In[49]:


# Calculating accuracies of each classifier
rf_accuracy = accuracy_score(y_test, rf_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
adb_accuracy = accuracy_score(y_test, adb_predictions)
# Comparing the accuracies
print("Random Forest Accuracy:", rf_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("adaptive boosting accuracy:", adb_accuracy)


# In[50]:


# Defining the classifiers and accuracies
classifiers = ['Random Forest', 'Logistic Regression', 'Decision Tree','Adboosting']
accuracies = [rf_accuracy, lr_accuracy, dt_accuracy,adb_accuracy]

# Plotting the accuracies
plt.figure(figsize=(8, 6))
plt.bar(classifiers, accuracies)
plt.title('Classifier Accuracies')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1.0])
plt.show()


# In[ ]:

pickle.dump(adb_clf, open("model.pkl", "wb"))



# In[ ]:




