# Statistical-Analysis-On-Students-data

Analyzing(elementray and advance statistical analysis) the student performance data provided by kaggle compition.

## Data Descriptin

data is given in a CSV format from Kaggle.
```
pd_read_csv.describe() #describe the initial analysis
```
```
       math score  reading score  writing score
count  1000.00000    1000.000000    1000.000000
mean     66.08900      69.169000      68.054000
std      15.16308      14.600192      15.195657
min       0.00000      17.000000      10.000000
25%      57.00000      59.000000      57.750000
50%      66.00000      70.000000      69.000000
75%      77.00000      79.000000      79.000000
max     100.00000     100.000000     100.000000
```

```
pd_read_csv.dtypes #this ouputs the all data's types
```
```
gender                         object
race/ethnicity                 object
parental level of education    object
lunch                          object
test preparation course        object
math score                      int64
reading score                   int64
writing score                   int64
```


