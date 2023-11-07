#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install colorama


# In[ ]:


pip install tld


# In[ ]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

from colorama import Fore  #Colorama is a module to color the python outputs

from urllib.parse import urlparse
# This module defines a standard interface to break Uniform Resource Locator (URL)
# strings up in components (addressing scheme, network location, path etc.),
# to combine the components back into a URL string,
# and to convert a “relative URL” to an absolute URL given a “base URL.”

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tld import get_tld, is_tld


# In[ ]:


data = pd.read_csv('malicious_phish.csv')
data.head()


# In[ ]:


##Meta information of Dataframe
data.info()
##null(NAN) value check
data.isnull().sum()


# In[ ]:


#counting the types
count = data.type.value_counts()
print(count,"\n")
x=count.index
print(x)


# In[ ]:


sns.barplot(x=count.index, y=count)
plt.xlabel('Types')
plt.ylabel('Count');


# In[ ]:


data['url'] = data['url'].replace('www.', '', regex=True)
data


# In[ ]:


#catagories datas
rem = {"Category": {"benign": 0, "defacement": 1, "phishing":2, "malware":3}}
data['Category'] = data['type']
data = data.replace(rem)
data.head(25)


# In[ ]:


#Feature Extraction=== string count of the dataset
data['url_len'] = data['url'].apply(lambda x: len(str(x)))
data.head()


# In[ ]:


def process_tld(url):
    try:
        res = get_tld(url, as_object = True, fail_silently=False,fix_protocol=True)
        pri_domain= res.parsed_url.netloc
    except :
        pri_domain= None
    return pri_domain


# In[ ]:


data['domain'] = data['url'].apply(lambda i: process_tld(i))
data.head()


# In[ ]:


feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
for j in feature:
    data[j] = data['url'].apply(lambda i: i.count(j))
data.head()


# In[ ]:


def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0





# urlparse():This function parses a URL into six components, returning a 6-tuple.
# This corresponds to the general structure of a URL. Each tuple item is a string.
# The components are not broken up in smaller parts
#(for example, the network location is a single string), and % escapes are not expanded.


# In[ ]:


data['abnormal_url'] = data['url'].apply(lambda i: abnormal_url(i))
data.head(10)


# In[ ]:


sns.countplot(x='abnormal_url', data=data);


# In[ ]:


def httpSecure(url):
    htp = urlparse(url).scheme #It supports the following URL schemes: file , ftp , gopher , hdl ,
                               #http , https ... from urllib.parse
    match = str(htp)
    if match=='https':
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


# In[ ]:


data['https'] = data['url'].apply(lambda i: httpSecure(i))
data.head(20)


# In[ ]:


sns.countplot(x='https', data=data);


# # Counts the number of digit characters in a URL

# In[ ]:


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


# In[ ]:


data['digits']= data['url'].apply(lambda i: digit_count(i))
data.head()


# # Counts the number of letter characters in a URL

# In[ ]:


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

# The isalpha() method returns True if all the characters are alphabet letters (a-z).
# Example of characters that are not alphabet letters: (space)!


# In[ ]:


data['letters']= data['url'].apply(lambda i: letter_count(i))
data.head()


# In[ ]:


plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), linewidths=.5)


# # Data Cleaning

# In[ ]:


X = data.drop(['url','type','Category','domain'],axis=1)#,'type_code'
y = data['Category']


# In[ ]:


X


# In[ ]:


y


# # Train & Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


y_train


# In[ ]:


y_test


# # Training Models

# In[ ]:


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve


# In[ ]:


#test dataset
models = [DecisionTreeClassifier,RandomForestClassifier,KNeighborsClassifier,GaussianNB]
accuracy_test=[]
for m in models:
    print('#############################################')
    print('######-Model =>\033[07m {} \033[0m'.format(m))
    model_ = m()
    model_.fit(X_train, y_train)
    pred = model_.predict(X_test)
    acc = accuracy_score(pred, y_test)
    accuracy_test.append(acc)
    print('Test Accuracy :\033[32m \033[01m {:.2f}% \033[30m \033[0m'.format(acc*100))
    print('\033[01m              Classification_report \033[0m')
    print(classification_report(y_test, pred))
    print('\033[01m             Confusion_matrix \033[0m')
    cf_matrix = confusion_matrix(y_test, pred)
    plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap='Blues', annot=True,fmt= '0.2%')
    plt.show()
    print('\033[31m###################- End -###################\033[0m')


# In[ ]:


#Final Report
output = pd.DataFrame({"Model":['Decision Tree Classifier','Random Forest Classifier','KNeighborsClassifier','Gaussian NB'],
                      "Accuracy":accuracy_test})
output


# In[ ]:


plt.figure(figsize=(10, 5))
plots = sns.barplot(x='Model', y='Accuracy', data=output)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

plt.xlabel("Models", size=14)
plt.xticks(rotation=20);
plt.ylabel("Accuracy", size=14)
plt.show()

