#Reading the dataset
#First Step, Let’s view our data and interpret the attributes in the dataset.
#Step 1: Importing Basic Libraries
#Importing Basic Libraries 
#Importing Basic Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Step 2: Reading The Dataset
#We read the CSV file in a pandas dataframe and view the head() of our dataframe.
#Reading The Dataset

df = pd.read_csv('1700271417138_Bank_Personal_Loan_Modelling.csv')
df


#Step 3: Viewing the columns’ info
#Viewing the columns’ info
df.info()

#Step 4: View some quick statistical measures for the continuous attributes
df[['Age','Income','CCAvg']].describe()
#We can conclude:
#The average age of 45 years with a deviation of 11 years.
#The average income of 73,000$ with the maximum applicant with 224,000$ for income.
#75% of the customers have an income of less than 98,000$.
#Average credit card spending of 1900$.

#Step 5: Quick Measures with Filters

#It might be helpful to understand the behavior of those who previously applied for a loan to target those. So, we filter for those who accepted the personal loan in the last campaign (Personal_Loan = 1).
df[df['Personal Loan']==1][['Age','Income','CCAvg']].describe()

# Missing Values in the attributes
df.isna().sum()
# Duplicate Values in the attributes
sum(df.duplicated(subset = ['ID']))
#Drop Unneeded columns
df = df.drop(['ID'],axis = 1)
df

#Find Histogram Distribution of Attribute
fig = plt.figure(figsize = (6,6))
plt.hist(df['Age'], color = 'r')
plt.title('Age Histogram')
plt.xlabel('Age')
plt.ylabel('Counts')



#Box Plot for the 5 point summary and the outliers if any

fig = plt.figure(figsize = (6,6))
sns.boxplot(x=df['Age'])
plt.title('Age Box plot')
plt.xlabel('Age')
plt.ylabel('Box Representation')



#Plot Violin Plot to check the distribution of the data
fig = plt.figure(figsize = (6,6))
sns.violinplot(x=df['Age'])
plt.title('Age Violin plot')
plt.xlabel('Age')
plt.ylabel('Distribution')



#Education counts plot
sns.countplot(x='Education',data=df)




#Filter Education for only those who previously took a loan
sns.countplot(x='Education',data=df[df['Personal Loan']==1]);


#We can check the relationship using the crosstab function()
pd.crosstab(df['Education'],df['Personal Loan']).plot(kind="bar",figsize=(6,6))
plt.legend(["No Loan","Loan"])
plt.xlabel('Education')
plt.ylabel('Number of Occurences')
plt.show()


# Plot Correlation between Features
fig = plt.figure(figsize = (10,10))
sns.heatmap(df.corr(),annot = True)



#Pairplot the relationships between numerical columns
sns.pairplot(df[['Age','Income','Experience','CCAvg']])





















