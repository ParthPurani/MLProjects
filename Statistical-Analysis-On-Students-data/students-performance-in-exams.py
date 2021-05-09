'''
this is the doc
'''
print(__doc__)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
csv_loc = 'E:/kaggle_students-performance-in-exams/StudentsPerformance.csv'
df = pd.read_csv(csv_loc)
'''
at this point we have catagorical data on which we can not do linear regression directly
so we have to convert the catagorical data into to corresponding numeric data

'''
encoding_df = { 'gender': {'male' : 0, 'female' : 1},
                'race/ethnicity': {'group A' : 0,'group B' : 1,'group C' : 2,'group D' : 3,'group E' : 4},
                'parental level of education': {'high school' : 0, 'some high school' : 1, 'some college' : 2, "bachelor's degree" : 3, "associate's degree" : 4, "master's degree" : 5},
                'lunch': {'standard' : 0, 'free/reduced' : 1 },
                'test preparation course': {'none' : 0 , 'completed' : 1}
               }
df.replace(encoding_df, inplace=True)

#features
x = np.array(df.drop(['math score', 'reading score', 'writing score'], axis=1)) #here axis=1 corresponds to column
#labels
y = np.array(df[['math score', 'reading score', 'writing score']])                

x = preprocessing.scale(x)


x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3) 

lrm = LinearRegression()
lrm.fit(x_train, y_train)
accuracy = lrm.score(x_test, y_test)
print(accuracy)

