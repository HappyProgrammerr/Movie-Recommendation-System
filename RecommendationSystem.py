# Import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing
from scipy.sparse import hstack
import pandas_profiling


# Load dataset
df = pd.read_csv("MoviesOnStreamingPlatforms_updated.csv")
df = df.iloc[:,1:]  # removing in unnamed index column

# print(df.head())
# print(df.info())
# print(df.Type.unique())


#Finding Missing values in all columns
miss = pd.DataFrame(df.isnull().sum())
miss = miss.rename(columns={0:"miss_count"})
miss["miss_%"] = (miss.miss_count/len(df.ID))*100
#print(miss)

# Dropping values with missing % more than 50%
df.drop(['Rotten Tomatoes', 'Age'], axis = 1, inplace=True)
# Dropping Na's from the following columns
df.dropna(subset=['IMDb','Directors', 'Genres', 'Country', 'Language', 'Runtime'],inplace=True)
df.reset_index(inplace=True,drop=True)
# Converting into object type
df.Year = df.Year.astype("object")


#print(df.info())

#checking Distribution of years
plt.figure(figsize=(20,5))
sns.displot(df['Year'])
plt.show()



