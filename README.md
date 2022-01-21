# SearchEngine
A search engine that runs locally on a machine and searches a large corpus.

## Introduction
This is a Windows Command Prompt app that takes a query (English term(s)) and returns the top 5 relevant documents in Wikipedia for it. 
The Wikipedia documents dataset has to be downloaded. For the sake of performance, only the first 100,000 documents are selected, but the application design allows easily to divide the input data into chunks. A log of old queries and a indexed vocabulary are required for the program to execute.
This project was a part Information Retrieval course project back in Fall 2019 at Boise State University.

The purpose of this project is to put into practice the different IR strategies related to the search process. Per project instruction the goal is to 
•	demonstrate the understanding of core IR concepts along with the ability to translate IR theory into practice
•	Develop and apply techniques required to handle large text collections
The search engine UI is decided to be Windows Command Prompt for its simplicity and speed. 

### Steps to Run
1- Download the data at [here](https://drive.google.com/drive/folders/1P7d2TqUnmzF_CzzbN_FZzOkzz8fN-Huu?usp=sharing)

2- Open Windows Command Prompt and run >> Python main.py

3- Enter the query (English term(s))

The program should return the top five documents and a snipped for each one. For example, in the following picture the top five results are compared with actual results of the same search in Wikipedia.org.

![thanksgiving](Results_for_Thanksgiving.jfif)

### Dependencies

Python 3.6
Libraries:
- nltk
- pandas
- numpy
- functools
- simplejson
- itertools


### Statistics on Document Collection (DC)
The collection consists of first 100,000 documents of the English Wikipedia collection provided by the instructor. I decided to include only 100,000 documents to improve the performance and efficiency of the search engine. If more time would be available, the Search Engine source code is structured such that it is relatively straightforward to divide the entire data collection into ten chunks of 100,000 documents and process them individually then merge the resulted index from each chunk into a final index table.
I used Python 3.6 programming language for this project. Specifically, Pandas library was used throughout this project due to its convenient and speed. Other libraries are used for statistical and text processing purposes which are mentioned in their corresponding sections.  

## Methodology
### Text Processing
Processing the documents was the first step of the project. This step consists of following tasks:
1.	Converting all the upper-case terms into lower-case
2.	Tokenizing the documents
3.	Removing the stopwords from the documents
4.	Lemmatizing the terms
5.	Stemming the terms
The Natural Language Tool Kit (NLTK) library was used to tokenize the documents into terms due to its convenient and simplicity and efficient methods. I decided to use both NLTK’s lemmatizer and stemmer to create a more generic and inclusive index so the possibility of an index being found in a document increases considering the relatively small size of the collection.
The list of stopwords consists of NLTK list of stopwords in addition to stopwords from SciKit learn library and punctuations.  The effort was to be as thorough as possible to decrease the size of the index while not compromising the task of the search engine which is returning relevant results to the user’s query. The stopwords list is stored in json format using SimpleJson library; hence it would be possible to use it in future steps of the project such as processing the queries.
The next step was to create the inverted index. I used Pandas DataFrame structure to store the terms and create the inverted index by setting the terms to the index of the data frame. A list of tuples is mapped to every term. Each tuple contains the document’s Id where the term is found and the term’s frequency in the document.
The size of index for 100,000 documents is 81977. This index includes numbers and terms in non-English characters as well. It would be a good idea to remove these terms from the index except for years in future improvements. 
The inverted index data frame is stored in Pickle format. The data types are attached to the data fields in this format and it is quite fast to reload and use the file in other scripts.

### Query Suggestions
A set of query logs were provided in .csv format by the instructor. These files were merged and processes prior to incorporating in the source code. The query logs were stored in a Pandas DataFrame. In this data frame, the column containing the queries was duplicated and shifted one single step in time to represent next query coming after original query. 
The equation below was proposed the score for query suggestions in the search engine. 
 
To improve the efficiency of the search engine all the variables in this equation were pre-computed and mapped to each query in the query log except for Mod (CQ, q’). Then the query log with the new information was saved as a pickle file to be used in the search engine.
The search engine loads the new query log as a data frame and finds the queries that start with the triggering query. This step of the process does not require any text processing for the triggering document neither for the queries in the log. The search engine uses the original terms with no modification in step. 
After computing the Mod (CQ, q’)¸ the engine has all the parameters to calculate the score for each candid query and returns the top three suggestions for the triggering query. It assigns zero value to the Mod variable where there is no instance of q’ being modified to candid query.

### Identify possible candidates:
In this step, the SE creates a data frame for each query term. A data frame has a document’s Id where the query term is found and the term’s frequency in the document. If the term is not found in the index, the SE raise an error stating “Cannot find the term(s) in the collection”.
The data frames corresponding to each query term are merged. The resulted data frame fields are named after the query term, therefore a query with duplicates will cause an error in the program. The documents are sorted based on the following two criteria to determine the candidates:
•	The presence of the term in a document
•	The sum of all the query term's frequency found in a document
A document will not be considered in the list of candidates if it misses more than one of the query terms. The search engine keeps the top 50 candidates and maps every document’s content from the original data to its Id in the data frame.

###  Relevance Ranking
It is required to carry on the exact text processing steps performed in creating the index on the content of the candidate document. The above variables were computed for each query term and mapped to every document. The documents were sorted base on their score and the top 5 were selected for the purpose of the project.

### Snippet Generation
In this step, the search engine tokenizes the original document’s content into sentences and creates a data frame structure with each sentence as an instance of the data frame. The TF-IDF is calculated for each term of every sentence. The similarity of the query and each sentence is computed.  The top two sentences for every document are selected to accompany the document’s title as the snippet in the results. The user would be able to read the snippets and select the number in which he/she is interested, and the search engine will retrieve that document for the user.
If a query term was not found in the document, the TF-IDF value would be zero. No smoothing method was used to avoid the division by zero error. The workaround was to assign 1e-8 to the term’s weight.
