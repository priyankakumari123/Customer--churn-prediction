#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data= pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
import seaborn as sns
import scipy.stats as stats
from numpy.random import seed
seed(10)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[3]:


data.head(6)


# In[4]:


data.shape


# In[5]:


data.dtypes


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data['Churn']= data['Churn'].map({'Yes': 1, 'No': 0})


# In[9]:


def bargraph(feature):
    churn= data[data['Churn']==1][feature].value_counts()
    stay= data[data['Churn']==0][feature].value_counts()
    df= pd.DataFrame([churn,stay])
    df.index= ('churn','stay')
    df.plot(kind='bar',stacked= True)


# In[10]:


data['Churn'].value_counts()


# In[11]:


bargraph('Churn')


# In[12]:


bargraph('gender')


# In[13]:


bargraph('SeniorCitizen')


# In[14]:


bargraph('Dependents')


# In[15]:


bargraph('tenure')


# In[16]:


data.tenure[data.tenure <= 16] = 0
data.tenure[(data.tenure > 16)& (data.tenure<=26)] = 1
data.tenure[(data.tenure > 26)& (data.tenure<=36)] = 2
data.tenure[(data.tenure > 36)& (data.tenure<=62)] = 3
data.tenure[data.tenure > 62] = 4

    


# In[17]:


bargraph('tenure')


# In[18]:


bargraph('PhoneService')


# In[19]:


bargraph('MultipleLines')


# In[20]:


bargraph('InternetService')


# In[21]:


data['InternetService'].value_counts()


# In[22]:


data.InternetService[data.InternetService == 'Fiber optic'] = 0
data.InternetService[data.InternetService == 'DSL'] = 1
data.InternetService[data.InternetService == 'No'] = 2


# In[24]:


bargraph('OnlineSecurity')


# In[25]:


bargraph('OnlineBackup')


# In[26]:


bargraph('DeviceProtection')


# In[27]:


bargraph('TechSupport')


# In[28]:


bargraph('StreamingTV')


# In[29]:


bargraph('StreamingMovies')


# In[30]:


bargraph('Contract')


# In[31]:


bargraph('PaymentMethod')


# In[32]:


bargraph('PaperlessBilling')


# In[37]:


data.MonthlyCharges[data.MonthlyCharges <= 21] = 0
data.MonthlyCharges[(data.MonthlyCharges > 21)& (data.MonthlyCharges<=41)] = 1
data.MonthlyCharges[(data.MonthlyCharges > 41)& (data.MonthlyCharges<=70)] = 2
data.MonthlyCharges[(data.MonthlyCharges > 70)& (data.MonthlyCharges<=100)] = 3
data.MonthlyCharges[data.MonthlyCharges > 100] = 4


# In[38]:


bargraph('MonthlyCharges')


# In[34]:


data['gender'].value_counts()


# In[39]:


data['SeniorCitizen'].value_counts()

    


# In[40]:


data['Partner'].value_counts()


# In[41]:


data['Dependents'].value_counts()


# In[42]:


data['gender']= data['gender'].map({'Male': 1, 'Female': 0})


# In[43]:


data['Partner']= data['Partner'].map({'Yes': 1, 'No': 0})


# In[44]:


data['Dependents']= data['Dependents'].map({'Yes': 1, 'No': 0})


# In[45]:


data['PhoneService']= data['PhoneService'].map({'Yes': 1, 'No': 0})


# In[46]:


data['OnlineSecurity']= data['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service':2})


# In[47]:


data['DeviceProtection']= data['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service':2})


# In[48]:


data['TechSupport']= data['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service':2})


# In[49]:


data['StreamingMovies']= data['StreamingMovies'].map({'Yes': 1, 'No': 0,'No internet service':2 })


# In[50]:


data['StreamingTV']= data['StreamingTV'].map({'Yes': 1, 'No': 0,'No internet service':2 })


# In[51]:


data['PaperlessBilling']= data['PaperlessBilling'].map({'Yes': 1, 'No': 0})


# In[52]:


data['MultipleLines']= data['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service':2})


# In[53]:


data['OnlineBackup']= data['OnlineBackup'].map({'Yes': 1, 'No': 0})


# In[54]:


data['Contract'].value_counts()


# In[55]:


data.Contract[data.Contract == 'Month-to-month'] = 0
data.Contract[data.Contract == 'Two year'] = 2
data.Contract[data.Contract == 'One year'] = 1


# In[56]:


data['PaymentMethod'].value_counts()


# In[57]:


data.PaymentMethod[data.PaymentMethod == 'Electronic check'] = 0
data.PaymentMethod[data.PaymentMethod == 'Bank transfer (automatic)'] = 2
data.PaymentMethod[data.PaymentMethod == 'Mailed check'] = 1
data.PaymentMethod[data.PaymentMethod == 'Credit card (automatic)'] = 3


# In[58]:


data.head(4)


# In[59]:


data.hist(bins=10,figsize=(9,5),grid=False)


# In[60]:


corr=data.corr()
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features')


# In[61]:


X = data.drop(['customerID','TotalCharges','Churn'], axis=1)


# In[62]:


X.head(4)


# In[63]:


y = data['Churn']


# In[64]:


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


# In[65]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(X), y, test_size=0.30, random_state=123)


# In[66]:


# LogisticRegression classifier
from sklearn.linear_model import LogisticRegression

tuned_parameters_LR = [{'C': [ 10, 100,1000], 'max_iter': [100,150], 'multi_class':['ovr'], 'tol':[1e-4,1e-6],
                         'random_state':[19]}]
scores = ['accuracy']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
clf = GridSearchCV(LogisticRegression(), tuned_parameters_LR, cv=10, scoring='%s' % score)
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print("Detailed confusion matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.matshow(confusion_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('Churned')
plt.xlabel('Predicted')


# In[67]:


print("Accuracy Score: \n")
print(accuracy_score(y_true, y_pred))
 
# false positive and true positive rates
FPR, TPR, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
 
# AUC calculation

roc_auc = auc(FPR, TPR)
print ('ROC AUC: %0.3f' % roc_auc )
 
# ROC curve
plt.figure(figsize=(5,5))
plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
print()


# In[ ]:




