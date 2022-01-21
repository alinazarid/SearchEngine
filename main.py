# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:48:08 2019

@author: alina
"""
# # Project 01_Identifying candidate resources
# Given a query q, create a set of documents CR comprised of all the documents in DC that contain each of the terms in q.
# ## Setup

# In[1]:



from nltk import word_tokenize, PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize 
import pandas as pd
import numpy as np
#from collections import Counter as ctr
from functools import reduce
#from simplejson import dump, load
from simplejson import load
from itertools import compress
from collections import Counter
#from nltk.tokenize import sent_tokenize
#pd.set_option('display.max_colwidth', -1)    # Avoids truncating the results
#pd.options.display.max_rows = 999
class SearchDone(Exception): pass
pd.set_option('mode.chained_assignment', None)

# ## Load Data

# In[2]:
# number documents in the collection
global N, stop_words, data, lm, ps, indx
N=100000

# Load stop words
with open(r'./data/stop_words.txt', 'r') as filehandle:
    stop_words = load(filehandle)
lm = WordNetLemmatizer()
ps = PorterStemmer()


# In[3]:


# Load the data
dataDir= r'./Data/data_{}k.pkl'.format(N//1000)
data= pd.read_pickle(dataDir, compression = 'gzip')
# Reverting from multiindex to single index 
data.reset_index(level=[0], inplace = True)

# Load the query log
logDir = r'data/logs.pkl'
logs = pd.read_pickle(logDir)

# In[4]:


# Load indices
ixDir =  r'./Data/index_{}K.pkl'.format(N//1000)
indx= pd.read_pickle(ixDir, compression = 'gzip')

def candidRetriever(Q):
    # ## Processing Query
    
    # In[6]:
    
    
    # Lower casing, lemmatizing and stemming the query
    q = word_tokenize(Q.lower())
    q = pd.Series(q)
    q = q[~q.isin(stop_words)]
    global q_
    q_ = q.map(lm.lemmatize).map(ps.stem)
    qLen = len(q)
    
    
    # ### Creating sorted candidate resources
    # In this section, a dataframe with each term as a field is created along with the term's count. Then the documents are sorted based on two critera:
    
    
    # In[7]:
    
    
    # Create DataFrame for each term in the query
    try :
        dfLt = [pd.DataFrame(indx.loc[q_[t]].values[0],
                             columns=['doc',('f_'+q[t])]) for t in range(qLen)]
    except KeyError:                              
        print('\nCan not find the term(s) in the collection')
        raise SearchDone

    ############################################################### Run the main program again
        
    # Merge the list of dataframes to create a candidate documents
    docCR = reduce(lambda df1,df2: pd.merge(df1,df2,on='doc'
                                            , how='outer'
                                            , sort=True)
                   , dfLt)
    
    docCR['#notNA'] , docCR['fSum']= docCR.iloc[:,1:].count(axis=1) , docCR.iloc[:,1:].sum(axis=1)
    # consider docs with |q| terms or |q|-1
    docCR = docCR[docCR['#notNA'] >= (qLen-1)]
    # Sort the candids
    docCR.sort_values(by = ['#notNA', 'fSum']
                     , ascending = False
                     , inplace =True)
    # Adding the document text to the resources
    candids = docCR[:50].merge(data
                ,how = 'inner'
                , left_on = 'doc'
                , right_on = 'id'
               , sort=True).set_index('id')
    candids.drop(['doc','#notNA', 'fSum'],axis=1, inplace=True)
    candids.fillna(value = 0, inplace=True)
    
    
    # In[9]:
    
    # # Project 01_Relevance Ranking
    # Compute for each resource (i.e., document) in CR its corresponding relevance score 
    # with respect to q.
    
    # ### Text Precessing
    
    # In[9]:
    
    
    # Lower casing, lemmatizing and stemming the content of documents
    candids['tokens'] = candids['content'].apply(lambda d: str(d).lower()).apply(word_tokenize)
    candids['tokens'] = candids['tokens'].apply(lambda c: list(compress( c, ~pd.Series(c).isin(stop_words).values )))
    candids['tokens'] = candids['tokens'].apply(lambda c: pd.Series(c).map(lm.lemmatize).map(ps.stem).values)
    
    
    # ### Relevance Score
    
    # In[10]:
    
    
    # max(d)
    candids['maxd'] = candids['tokens'].apply(lambda r : Counter(r).most_common(1)[0][1])
    
    
    # In[11]:
    
    
    # create the query dataframe with 
    qDF = pd.DataFrame({'nw':[len(indx.loc[t][0]) for t in q_]}, index=q)
    qDF['IDF'] =qDF.apply(lambda nw : np.log2(np.divide(N,nw)))
    
    
    # In[21]:
    
    
    #candids ['score'] = candids.apply(lambda r: sum(r['f_'+w]/r['maxd']*qDF['IDF'].loc[w] for w in set(q)), axis=1)
    candids ['score'] = candids.apply(
            lambda r: sum(r.filter(like='f_'+w)[0]/r['maxd']*qDF['IDF'].loc[w] for w in q )
            , axis=1)
    candids['qTF'] = candids.apply(    # this is the sum of squred of each element for snippet generation equation
            lambda r: sum(
                    (r.filter(like='f_'+w)[0]/r['maxd']*qDF['IDF'].loc[w])**2 for w in q 
                    ), axis=1)
    candids.sort_values('score', ascending=False,inplace=True)
    return candids

def snipGenerator(candids):
    # ### Generating Snippets
    # For each selected result, create the corresponding snippet. The snippet of each (top-ranked) document d should include:    
    
    # #### Title in the snippet
    
    # In[24]:
    results = candids[:5]
    # The numinator of weights for query and document
    results.loc[:,'sent'] = results['content'].apply(sent_tokenize)
    results = results.explode('sent')
    results.loc[:,'counts'] = results['tokens'].apply(lambda r: Counter(r))
    results.loc[:,'tokens'] = results['sent'].apply(lambda d: str(d).lower()).apply(word_tokenize)
    results.loc[:,'tokens'] = results['tokens'].apply(lambda c: list(compress( c, ~pd.Series(c).isin(stop_words).values )))
    results.loc[:,'tokens'] = results['tokens'].apply(lambda c: pd.Series(c).map(lm.lemmatize).map(ps.stem).values)
    # Equation for document weights in the denominator amd mumerator
    results.loc[:,'numer'] = results.apply(lambda ds : sum((
                                                np.log2(N/len(indx.loc[t][0])    # IDF term
                                                )*(
                                                ds['counts'][t]/ds['maxd'])      # TF: frquency of "t"/maxd of that document
                                                )**2 if (t in indx.index) &  (t in q_.values) else 1e-8 
                                                for t in ds['tokens']), axis=1)

    results.loc[:,'denom'] = results.apply(lambda ds : sum((
                                                np.log2(N/len(indx.loc[t][0])    # IDF term
                                                )*(
                                                ds['counts'][t]/ds['maxd'])      # TF: frquency of "t"/maxd of that document
                                                )**2 if t in indx.index else 1e-8 for t in ds['tokens']), axis=1)

    #Final denominator
    results.loc[:,'snipScore'] = results.apply(lambda r:  r['numer'] / (r['denom']*r['qTF'])**0.5 \
                                           if r['denom']*r['qTF'] !=0 else 1e-8 , axis=1)
    snip = results.groupby('id', as_index=False).apply(lambda df: df.nlargest(2,'snipScore'))
    # Process to print the snippets
    snip.sort_values('score',ascending=False, inplace=True)
    snip.reset_index(level=1,drop=True, inplace=True)
    snip['sent'] = snip.groupby(snip.index,sort=False)['sent','title'].apply(lambda s: '\n'.join(s['sent']))
    snip = snip[['title','sent']].drop_duplicates().reset_index(drop=True)
    snip.apply(lambda s: print(
             '#{0}\n## {1}\n{2}\n'.format(
                 s.name+1
                 , s['title']
                 , s['sent']
                                         )), axis=1)

def qScore(row):
    """ Function to calculate the ranking score for the suggested queries"""
    frq = row['qFrq']
    mod = row['mode']
    time = row['timeRank']
    return (frq+mod+time)/ (1 - min([frq,time, mod]))

def qSuggestion(Q):
    """ Uses a query log to suggest three query to the user"""
    # Lower casing splitting the query
    query = Q.lower()
    qLen    = len(query.split())
    
    
    # ## Identify Candidates
    # In[ ]:
    
    
    # We need to find sessons that include the accutal query
    # Then the next query is larger in size
    # and it is containing the query
    query_filters = logs [ 
                       (logs['nxtL']> qLen)]
    query_filters = query_filters[
                        (query_filters['nxtQ'].str.startswith(query))]
    
    # ## Rank Suggestion Candidates
    
    # In[ ]:
    # Total number of sessions in QL in which q’ appears
    # Calculate the total number of sessions that the string of query is one of the queries searched in that session   
    qMode = query_filters[query_filters['Query']==query][['Query','nxtQ']]
    totalSessions = qMode.index.nunique()
    qMode.loc[:,'mode'] = qMode.groupby('nxtQ')['nxtQ'].transform('count')/totalSessions
    query_filters = query_filters.merge(qMode
                                    , how='left'
                                    , on=['Id','Query','nxtQ'])
    query_filters['mode'].fillna(0,inplace=True)
    
    # ### Scoring
    
    # In[ ]:
    
    if not query_filters.empty:
        query_filters['score'] = query_filters.apply(lambda r: qScore(r), axis=1)
        query_filters.sort_values('score',ascending=False, inplace=True)
        query_filters.drop_duplicates(subset ="nxtQ",inplace = True) 
        return query_filters[:3]['nxtQ'].reset_index(drop=True)       
    else:
        return '\nNo suggestion found!'
        # this means there was no such a query that has transformed into an other query

# In[5]:

if __name__ == '__main__':
    search =True
    while search:# Load the query
        global Q
        Q=input("""\nPlease enter a query.
                \nFinish with "..." for suggestions: \n\n""")
        # Loop to get suggestions
        while Q.endswith('...'):
            print(qSuggestion(Q[:-3]))
            Q=input("""\nEnter the new query!
                    \nFinish with "..." for suggestions: \n\n""")
        # Performs the search
        try:
            candids = candidRetriever(Q)
            snipGenerator(candids)
            d = int(input ('Enter the number you are interested: '))-1
            print( candids['content'].iloc[d] )
        except SearchDone:
            pass
        nxt = input('\nWould you like to run another query: (y/n)\n' )
        search = True if nxt=='y' else False
    print('Your search session has ended!')

