
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

#gender_submission = pd.read_csv("gender_submission.csv")
#print("GENDER_SUBMISSION: " + str(gender_submission.head()))
#print(gender_submission.describe())
# What is gender_submission.csv even used for?
# Answer: It's just an example submission file.


# In[15]:


train = pd.read_csv("train.csv")
print(train.describe())

df = train.groupby(['Pclass', 'Survived'], as_index=False) \
          .count() \
          [['Pclass', 'Survived', 'PassengerId']]


# In[16]:


test = pd.read_csv("test.csv")
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

# Also:
# Use transformation on highly-skewed variables to make them be more normally distributed.
# Simply taking the log is often a good rough approximation, look up the Box-Cox test
# for info on how to apply a better transformation, tuned to the specific data at hand.
plt.hist(train_clean.Fare)
plt.hist(train_clean.Fare.apply(lambda x: 0 if x <= 0 else np.log(x)))


# In[19]:


plt = train_clean.hist(bins=25, figsize=(20,15) )


# Not really much to glean from that...

# #### Other stuff

# In[20]:


corr_matrix = train_clean.corr()
corr = corr_matrix["Survived"].sort_values(ascending=False)
print(corr)


# TODO: Use scikit's built-in for this.
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


# I would do this.
df = train_clean.groupby(['Sex', 'Survived'], as_index=False).count()[['Sex', 'Survived', 'Pclass']].rename(columns={'Pclass': 'PassengerCount'})
df['TotalProportion'] = df.PassengerCount/df.PassengerCount.sum()
females = df[df['Sex'] == 0]['PassengerCount'].sum()
males = df[df['Sex'] == 1]['PassengerCount'].sum()
df['WithinSexProportion'] = df.apply(lambda x: x.PassengerCount/females if x.Sex == 0 else x.PassengerCount/males, axis=1)
df
""" 



# In[60]:


train_clean["Pclass"].unique()


# In[61]:


train_clean[["Pclass", "Fare"]].corr()


# ^ These are strongly correlated. Group? 

# This makes sense since 


# In[62]:


#from pandas.tools.plotting import scatter_matrix
scatter = pd.plotting.scatter_matrix(
    train_clean[["Survived", "Fare", "Pclass", "Age", "Parch", "Sex"]], 
    figsize=(12, 8), 
    alpha=0.1
)

# Scatter matrix won't be as useful here since we're mostly plotting continuous vs
# categorical variables. Boxplots would be my go-to tool, and plotting only the
# continuous vs continuous variable interactions with a scatter matrix.
scatter = pd.plotting.scatter_matrix(
    train_clean[["Survived", "Fare", "Age"]], 
    figsize=(12, 8), 
    alpha=0.3
)

train_clean.boxplot(by='Survived', figsize=(20,20))
train_clean.boxplot(column='Age', by='Survived', figsize=(20,20))
train_clean.boxplot(column='Parch', by='Survived', figsize=(20,20))
train_clean.boxplot(column='SibSp', by='Survived', figsize=(20,20))

# Boxplots aren't doing a good job of revealing patterns in categorical vs categorical.
def stacked_bar(df, x, y):
    df2 = train_clean.groupby([x, y])[y].count().unstack(y)
    df2.plot(kind='bar', stacked=True)

stacked_bar(train_clean, 'Pclass', 'Survived')
stacked_bar(train_clean, 'Parch', 'Survived')
stacked_bar(train_clean, 'Sex', 'Survived')


# Did not show much. I just wanted to try it out. 
# Normalize now?

# When you filter to just the continuous vs continuous dimensions 
# the skew in Fare becomes apparent.
# Also, Fare and Age are positively correlated.
# Age also seems skewed, and there's a clear jump in count when age > 18,
# indicating a potential mixed distribution around children vs. adults.

# Experimenting with plydata & plotnine
from plydata import *
from plotnine import *
(train_clean >>
  group_by('Survived', 'Pclass') >>
  summarize(NumPassengers='len(Survived)') >>
  define(PercentOfTotal='NumPassengers / sum(NumPassengers)')
) >> (
ggplot() +
  geom_bar(aes(x='Pclass', y='NumPassengers', fill='Survived'), stat='identity')
)

# Same plot but with Survived as a category. Looks like visualization adapts
# to use a qualitative color mapping, as I would expect from ggplot in R.
(train_clean >>
  define(Survived = 'Survived.astype("category")') >>
  group_by('Survived', 'Pclass') >>
  summarize(NumPassengers='len(Survived)') >>
  define(PercentOfTotal='NumPassengers / sum(NumPassengers)')
) >> (
ggplot() +
  geom_bar(aes(x='Pclass', y='NumPassengers', fill='Survived'), stat='identity')
)

# Define a function to do it, call it on a few columns to see what's up.
def stacked_bar(df, x, y, ratio=False):
    return (df >>
      define(x_cat = '{0}.astype("category")'.format(x),
             y_cat = '{0}.astype("category")'.format(y)) >>
      group_by('x_cat', 'y_cat') >>
      summarize(Count='len(y_cat)')
    ) >> (
    ggplot() +
      geom_bar(aes(x='x_cat', y='Count', fill='y_cat'), stat='identity') + 
      labs(title='Count of {0} by {1}'.format(y, x), x=x, fill=y)

    )

# Mostly men died.
stacked_bar(train_clean, 'Sex', 'Survived', True) # Greater in-group proportion of them
stacked_bar(train_clean, 'Survived', 'Sex') # Also make up vast majority of total deaths

# 3rd, 2nd, and 1st class, in that order, were the most deadly.
stacked_bar(train_clean, 'Pclass', 'Survived') # Largest in-group proportion
stacked_bar(train_clean, 'Survived', 'Pclass') # 3rd class passengers make up ~75% of total deaths

# Technically Parch and Sibsp aren't categoricals, so this is a bit wonky.
# These two should probably be combined into a new, separate feature.
# For example: FamilySize = Parch + Sibsp
# Looks like it's harder for people with more family members onboard to survive.
# Perhaps families died together? Could we pull out surname from Name field and use
# that as a feature? Might not generalize well, but worth exploring.
stacked_bar(train_clean, 'Parch', 'Survived')
stacked_bar(train_clean, 'Survived', 'Parch') # Basically no-one with 3 or more parents/children aboard survived

stacked_bar(train_clean, 'SibSp', 'Survived')
stacked_bar(train_clean, 'Survived', 'SibSp') # Basically no-one with 3 or more sibling/spouses aboard survived

train_clean['FamiliySize'] = train_clean['Parch'] + train_clean['SibSp']
test['FamiliySize'] = test['Parch'] + test['SibSp']
stacked_bar(train_clean, 'FamiliySize', 'Survived')
stacked_bar(train_clean, 'Survived', 'FamiliySize') # Yeah, basically 3+ family members = death.

train_clean['BigFamily'] = train_clean['FamiliySize'].apply(lambda x: x > 3)
test['BigFamily'] = test['FamiliySize'].apply(lambda x: x > 3)
stacked_bar(train_clean, 'BigFamily', 'Survived')
stacked_bar(train_clean, 'Survived', 'BigFamily') # Yeah, basically 3+ family members = death.

# A few more features
train_clean['Age'].hist(bins=50)
train_clean['AgeGroup'] = train_clean['Age'].apply(lambda x: 'Child' if x <= 18 else 'Adult')
test['AgeGroup'] = test['Age'].apply(lambda x: 'Child' if x <= 18 else 'Adult')
stacked_bar(train_clean, 'AgeGroup', 'Survived')
stacked_bar(train_clean, 'Survived', 'AgeGroup')

train_clean["AgeGroup2"] = pd.qcut(train_clean["Age"], 10)
stacked_bar(train_clean, "AgeGroup2", "Survived")

# Embarked point could relate to passenger class, maybe different points
# have different levels of affluence, different cultures/norms that could
# influence behavior.
# May shed extra light on Pclass vs Survived dimension.
stacked_bar(train_clean, 'Embarked', 'Survived')
stacked_bar(train_clean, 'Survived', 'Embarked')
stacked_bar(train_clean, 'Pclass', 'Embarked')
stacked_bar(train_clean, 'Embarked', 'Pclass')
stacked_bar(train_clean, 'Pclass', 'Embarked')



# TODO:
# 0) Exploration & Feature Generation
#    - Surname, Title (Dr., Mrs., Ms.)
#    - CabinLetter & CabinNumber parsed out as separate columns
#    - FamilySize = Parch + SibSp
#    - FamilySize -> categorical
#    - Ratio of Parch or SibSp to Family size
#    - AgeGroup = Age categorical
# 1) Cleanup
#    - Normalize skewed variables (Box-Cox transform)
#    - Scale
#    - Dummy variables
#    - Impute/fill missing Age (maybe using Fare?), Cabin, Embarked, etc.
# 2) Split train_clean into train and validation sets.
# 3) Train model(s) on the train set.
#    - Use cross validation, minimize appropriate error metric.
# 4) Assess model generalizability on validation set.

train_clean = train.copy()



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
# Need to use more features to get better performance here.
# E.g. Pclass seems pretty relevant to survival rate for example.
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





