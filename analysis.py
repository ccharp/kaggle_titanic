#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

print("Finished importing")

#%% Read training data
gender_submission = pd.read_csv("gender_submission.csv")
print("GENDER_SUBMISSION: " + str(gender_submission.head()))
print(gender_submission.describe())
train = pd.read_csv("train.csv")
print("TRAIN: " + str(train.head()))
print(train.describe())
test = pd.read_csv("test.csv")
print("TEST: " + str(test.head()))
print(test.describe())

print("End read output")

#%% Clean


#%% Explore
train.hist(bins=50, figsize=(20,15))

#%% Train

#%% Test


