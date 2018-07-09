
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn import preprocessing as pre


# ## Import data

# In[14]:


gender_submission = pd.read_csv("gender_submission.csv")
print("GENDER_SUBMISSION: " + str(gender_submission.head()))
print(gender_submission.describe())
# What is gender_submission.csv even used for?


# In[15]:


train = pd.read_csv("train.csv")
print(train.describe())


# In[16]:


test = pd.read_csv("train.csv")
print(test.describe())
print(test.info())


# ## Clean and explore

# In[17]:


# Drop some columns that are probably not useful. Maybe explore these later.
# Ticket, Name, Embarked

# TODO: Convert cabin. NaNs=0; A=1; B=2; etc. Theory is cabins correspond to a location on the ship and 
#    survival is a function of location

print(train.dtypes)


# In[18]:


#obj_columns = train.select_dtypes(include=['object']).columns

train_clean = train.copy()

# Convert sex to integer. 0->Female, 1->Male
lenc = pre.LabelEncoder().fit(train_clean.Sex.unique())
train_clean.Sex = lenc.transform(train_clean.Sex)

# Also try converting object columns to one hot encoded columns
"""
train_hots = train.copy()
#cat_features = ['color', 'director_name', 'actor_2_name']
lenc = pre.LabelEncoder()
new_cat_features = new_cat_features.reshape(-1, 1) # Needs to be the correct shape
ohe = pre.OneHotEncoder(sparse=False) #Easier to read
print(ohe.fit_transform(new_cat_features))
# print(enc.transform(train_cats["Sex"]))

print(train_cats["Embarked"])
"""


# In[19]:


plt = train_clean.hist(bins=25, figsize=(20,15) )


# Not really much to glean from that...

# #### Other stuff

# In[20]:


corr_matrix = train_clean.corr()
corr = corr_matrix["Survived"].sort_values(ascending=False)
print(corr)

to_normalize = ["Age", "Fare", "Parch", "Pclass", "SibSp"]
f = lambda col: (col - col.mean())/(col.max() - col.min())
x_train_norm = train_clean.copy()
x_train_norm[to_normalize] = train_clean[to_normalize].apply(f, axis=1)


# In[21]:


# Drop columns with low correlation
train_clean = train_clean.drop(["PassengerId"], axis=1)


# Let's see how likely males are to survive compared to females...

# In[35]:


f = train_clean[train_clean['Sex'] == 1]
m = train_clean[train_clean['Sex'] == 0]
num_f = f.shape[0]
num_m = m.shape[0]

f_survive = f['Survived'].sum()/num_f
m_survive = m['Survived'].sum()/num_m

print("#m: {}; #f: {};\n%m: {}; %f: {}".format(num_f, num_m, f_survive, m_survive))


# In[60]:


train_clean["Pclass"].unique()


# In[61]:


train_clean[["Pclass", "Fare"]].corr()


# ^ These are strongly correlated. Group? 

# In[62]:


#from pandas.tools.plotting import scatter_matrix
scatter = pd.plotting.scatter_matrix(
    train_clean[["Survived", "Fare", "Pclass", "Age", "Parch", "Sex"]], 
    figsize=(12, 8), 
    alpha=0.1
)


# Did not show much. I just wanted to try it out. 
# 
# Normalize now?

# In[69]:


"""
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
"""
to_use = ["Sex", "Age", "Fare"]

# Are we missing in data in columns we care about? If so, fill with median
x_train = train_clean[to_use].apply(lambda col: col.fillna(col.median()), axis=0)

# Normalized inputs worked better
to_normalize = ["Age", "Fare"]#, "Parch", "Pclass", "SibSp"]
f = lambda col: (col - col.mean())/(col.max() - col.min()) # Normalize that ho
x_train_norm[to_normalize] = x_train[to_normalize].apply(f, axis=0)
print(x_train_norm.describe())

y_train = train_clean['Survived']


# In[81]:


from sklearn.linear_model import SGDClassifier
import sklearn.model_selection as ms 
import sklearn.metrics as metrics 

# Try stochastic gradient descent
sgd_classifier = SGDClassifier(random_state=1337)
results = ms.cross_val_predict(sgd_classifier, x_train_norm[["Sex"]], y_train, cv=3)
cm = metrics.confusion_matrix(y_train, results)
print(cm)
print("Precision: {}".format(cm[1][1]/(cm[1][1] + cm[0][1])))
print("Recall: {}".format(cm[1][1]/(cm[1][1] + cm[1][0])))
print(metrics.precision_score(y_train, results))
print(metrics.recall_score(y_train, results))
print(metrics.f1_score(y_train, results)) # Harmonic mean of Precision and Recall


# In[83]:


from sklearn.ensemble import RandomForestClassifier

# TODO: abstract shared code between this cell and the previous one into a function
# Try Random Forest classifier
forest_classifier = RandomForestClassifier(random_state=1337)
results = ms.cross_val_predict(forest_classifier, x_train_norm[["Sex"]], y_train, cv=3)
cm = metrics.confusion_matrix(y_train, results)
print(cm)
print("Precision: {}".format(cm[1][1]/(cm[1][1] + cm[0][1])))
print("Recall: {}".format(cm[1][1]/(cm[1][1] + cm[1][0])))
print(metrics.precision_score(y_train, results))
print(metrics.recall_score(y_train, results))
print(metrics.f1_score(y_train, results)) # Harmonic mean of Precision and Recall

