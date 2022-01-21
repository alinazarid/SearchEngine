#!/usr/bin/env python
# coding: utf-8

# # Project 01_Relevance Ranking
# Compute for each resource (i.e., document) in CR its corresponding relevance score 
# with respect to q.
# 
# * where q is the query
# * d is a document in CR
# * w is a non-stop stemmed/lemmatized word in q
# * freq(w, d) is the number of times w appears in d
# * maxd  is the number of times the most frequently-occurred term appears in d (which is constant for each document)
# * N is the number of documents in DC
# * and nw is the number of documents in DC in which w appears at least once. 

# ## Setup

# In[1]:


from nltk import word_tokenize, PorterStemmer
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import numpy as np
from collections import Counter as ctr
from time import time
from functools import reduce
from simplejson import dump, load
from itertools import compress
from collections import Counter
from nltk.tokenize import sent_tokenize

#pd.set_option('display.max_colwidth', -1)    # Avoids truncating the results
#pd.options.display.max_rows = 999


# ## Load Data

# In[2]:


# number documents in the collection
global N
N=10000


# In[3]:


# Load stop words
with open(r'./data/stop_words.txt', 'r') as filehandle:
    stop_words = load(filehandle)
lm = WordNetLemmatizer()
ps = PorterStemmer()


# In[5]:


# Load indices
ixDir =  r'./Data/index_{}K.pkl'.format(10000//1000)
indx = pd.read_pickle(ixDir, compression= 'gzip')


# In[6]:


# Load the CR
crDir = r'./data/CandidResources.pkl'
crDF = pd.read_pickle(crDir)


# In[7]:


# Load the query
Q='iran tehran'


# ### Text Precessing

# In[8]:


# Lower casing, lemmatizing and stemming the query
q = word_tokenize(Q.lower())
q = pd.Series(q)
q = q[~q.isin(stop_words)].map(lm.lemmatize).map(ps.stem)
qLen = len(q)


# In[9]:


# Lower casing, lemmatizing and stemming the content of documents
crDF['tokens'] = crDF['content'].apply(lambda d: str(d).lower()).apply(word_tokenize)
crDF['tokens'] = crDF['tokens'].apply(lambda c: list(compress( c, ~pd.Series(c).isin(stop_words).values )))
crDF['tokens'] = crDF['tokens'].apply(lambda c: pd.Series(c).map(lm.lemmatize).map(ps.stem).values)


# ### Relevance Score
# * maxd  is the number of times the most frequently-occurred term appears in d (which is constant for each document)
# * N is the number of documents in DC
# * and nw is the number of documents in DC in which w appears at least once. 

# In[10]:


# max(d)
crDF['maxd'] = crDF['tokens'].apply(lambda r : Counter(r).most_common(1)[0][1])


# In[11]:


# create the query dataframe with 
qDF = pd.DataFrame({'nw':[len(indx.loc[t][0]) for t in q]}, index=q)
qDF['IDF'] =qDF.apply(lambda nw : np.log2(np.divide(N,nw)))


# In[21]:


crDF ['score'] = crDF.apply(lambda r: sum(r['f_'+w]/r['maxd']*qDF['IDF'].loc[w] for w in q), axis=1)
crDF.sort_values('score', ascending=False,inplace=True)


# ### Generating Snippets
# For each selected result, create the corresponding snippet. The snippet of each (top-ranked) document d should include:    
# 
# * a. The title of d 
# 
# * b. The two sentences in d that have the highest cosine similarity with respect to q; with TF-IDF as the term weighting scheme

# #### Title in the snippet

# In[24]:


snip = crDF[:5].reset_index(drop=True)
snip['title']


# #### Calculate TF_IDF for each word in a sentence

# In[18]:


# # Sentence tokenizing
# results['frq_w'] = results['tokens'].apply(Counter)
# results['sent'] = results['content'].apply(sent_tokenize)
# results = results.explode('sent')
# results['sent'] = results['sent'].apply(lambda d: str(d).lower()).apply(word_tokenize)
# # truncate the sentence and keep the element that multiply to the query!
# results['sent'] =results['sent'].apply(lambda c: list(compress( c, ~pd.Series(c).isin(stop_words).values ))[:qLen])
# results['sent']=results['sent'].apply(lambda c: pd.Series(c).map(lm.lemmatize).map(ps.stem).values)


# In[16]:


# # calculating the tf_idf weight for each word in a sentence
# def fun(r):
#     return list(
#         pd.Series(r['sent']).apply(lambda w: r['frq_w'][w]/r['maxd'] * 
#                               np.log2(np.divide(N, len(indx.loc[w][0]) ) 
#                                      )
#                                   )
#                 )
# results ['d_weights'] = results.apply(fun, axis=1)
# #results.head(3)

