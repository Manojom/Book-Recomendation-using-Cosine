#!/usr/bin/env python
# coding: utf-8

# In[2]:


### important libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[4]:


df = pd.read_csv("C:\\Users\\Admin\\Downloads\\book (2).csv")


# In[6]:


df.head


# In[12]:


#### data preprocessing

features = ['User.ID', 'Book.Rating', 'Book.Title',]
for feature in features:
    df[feature] = df[feature].fillna('')


# In[32]:


####Combining Relevant Features into a Single Feature


def combined_features(row):
    return row['User.ID']+" "+row['Book.Rating']+" "+row['Book.Title']
df["combined_features"] = df.apply(combined_features, axis =1)


# In[33]:


###Extracting Features


cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
print("Count Matrix:", count_matrix.toarray())


# In[35]:


####Using the Cosine Similarity

cosine_sim = cosine_similarity(count_matrix)


# In[40]:


### User rating
book_user_likes = "Book.Rating"
def get_index_from_title(title):
    return df[df.Title == Title]["index"].values[0]
book_index = get_index_from_Book.Title(book_user_likes)


# In[42]:


####Generating the Similar BOOK Matrix

similar_book = list(enumerate(cosine_sim[Book_index]))


# In[44]:


####Sorting the Similar Books List in Descending Order

sorted_similar_Book = sorted(similar_Book, key=lambda x:x[1], reverse=True)


# In[46]:


####Printing the Similar Books
def get_title_from_index(index):
    return df[df.index == index]["Book.Title"].values[0]
i=0
for movie in sorted_similar_books:
    print(get_title_from_index(book[0]))
    i=i+1
    if i>15:
        break


# In[ ]:




