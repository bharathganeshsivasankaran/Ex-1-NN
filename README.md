<H3>ENTER YOUR NAME : Bharathganesh. S</H3>
<H3>ENTER YOUR REGISTER NO : 212222230022</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:

### dataset:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/b17f240d-066d-4af4-a635-4453bb564340)

### DROPPING THE UNWANTED DATASET:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/43b9989e-313d-453f-a4d9-69ea00f6885d)

### CHECKING NULL VALUES:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/7d3b00c6-8b84-481a-b1f1-6d5bb1cea045)

### CHECKING FOR DUPLICATION:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/e8ada4d0-e089-4c38-8974-3d27a65d8cc6)

### DESCRIBING THE DATASET:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/97dc6222-f6a7-4d56-8761-bc226d901ee6)

### SCALING THE DATASET:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/f772f457-c0d7-4482-9325-c066c68b2782)

### X FEATURES:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/12dd95ea-5587-49bb-9596-bea55802f892)

### Y FEATURES:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/c757e0cd-c249-49b8-bb1d-61981408f825)

### SPLITTING THE TRAINING AND TESTING DATASET:

![image](https://github.com/bharathganeshsivasankaran/Ex-1-NN/assets/119478098/66dfca68-8330-47f2-ae26-6b30f8cdf411)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


