# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <h1> Email Classifier </h1> 

# %%
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# %%
#the spam and valid emails are stored in 2 folders called "easy_ham" and "spam"
#we want to read all the files in these 2 folders and pu the filenames into a list of spam and not spam
ham_filenames = [filename for filename in sorted(os.listdir(os.path.join(os.getcwd(),"easy_ham"))) if len(filename) > 20]
spam_filenames = [filename for filename in sorted(os.listdir(os.path.join(os.getcwd(),"spam"))) if len(filename) > 20]


# %%
len(ham_filenames)


# %%
len(spam_filenames)

# %% [markdown]
# To parse emails we will use the email library of Python:
# 
# How the email library works for parsing:
# 
# The parser takes a serialized version of the email message(a stream of bytes) and converts it to a tree of EmailMessage objects.  The generator takes an EmailMessage and turns it back into a serialized byte stream.
# 
# There are 2 parser interfaces available, the Parser API and FeedParser API. The Parser API is most useful when you have the entire text of the message in memory or if the entire message lives in a file on the file system. FeedParser API is useful when you are reading the message from a stream which might block your waiting(reading from a url itself)

# %%
#collecting all the parsed ham messages
ham_path = os.path.join(os.getcwd(),"easy_ham")
spam_path = os.path.join(os.getcwd(),"spam")


# %%
import email
from email import policy


# %%
ham_messages = list()
for filename in ham_filenames:
    with open(os.path.join(ham_path,filename),'rb') as f:
        ham_messages.append(email.parser.BytesParser(policy = email.policy.default).parse(f))


# %%
spam_messages = list()
for filename in spam_filenames:
    with open(os.path.join(spam_path,filename),'rb') as f:
        spam_messages.append(email.parser.BytesParser(policy = email.policy.default).parse(f))


# %%
print(ham_messages[0].get_content().strip())

# %% [markdown]
# Emails can have different parts to it, with images, attachments. And these attachments can have emails in them. 

# %%
def get_email_structure(email):
    if isinstance(email,str):
        return email
    #get_payload() returns a list if the email is multipart and .is_multipart() = True
    payload = email.get_payload()
    if isinstance(payload,list):
        result = "multipart({})".format(', '.join([get_email_structure(sub_email) for sub_email in payload]))
        return result
    else:
        return email.get_content_type()


# %%
from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


# %%
structure_counts_ham = structures_counter(ham_messages)
structure_counts_spam = structures_counter(spam_messages)


# %%
structure_counts_ham.most_common()


# %%
structure_counts_spam.most_common()


# %%
type_list = list()
temp = [[type_list.append(i) for i in structures] for structures in [structure_counts_ham, structure_counts_spam]]
ham_type_counts = [structure_counts_ham[i] for i in set(type_list)]
spam_type_counts = [structure_counts_spam[i] for i in set(type_list)]


# %%
type_counts_df = pd.DataFrame({'Email Type':list(set(type_list)), 'Ham Count':ham_type_counts, 'Spam Count':spam_type_counts})
type_counts_df

# %% [markdown]
# Most valid(ham) emails are text/plain and contain a PGP(Pretty Good Privacy) signature, while Spam emails have a higher amount of HTML messages. 

# %%
#create a list of type of each email 
ham_email_type = [get_email_structure(email) for email in ham_messages]
spam_email_type = [get_email_structure(email) for email in spam_messages]


# %%
def FindEmailSender(email):
    try:
        return dict(email.items())['From']
    except:
        return "N/A"


# %%
#creating list of senders of each email
ham_email_senders = [FindEmailSender(email) for email in ham_messages]
spam_email_senders = [FindEmailSender(email) for email in spam_messages]


# %%
#combining the dataset to create a complete dataset for splitting into train and test sets
import numpy as np
X = np.array(ham_messages + spam_messages, dtype = object)
y = np.array([0] * len(ham_messages) + [1]*len(spam_messages))


# %%
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42, test_size=0.2)


# %%
#turning html in emails into tags we want
#turn head tags into ''
#turn anchor tags into hyper links
#turn all html tags into ''
import re
from html import unescape
def html_to_text(html):
    text = re.sub(r'<head.*?>.*?</head>','',html,flags = re.I|re.M|re.S)
    text = re.sub(r'<a.*?>.*?</a>',' HYPERLINK ',text,flags = re.I|re.M|re.S)
    text = re.sub(r'<.*?>','',text,flags=re.M|re.I|re.S)
    text = re.sub(r'(\s*\n)+','\n',text,flags = re.M|re.I|re.S)
    return unescape(text)


# %%
idx_html = [i for i in range(len(X_train)) if get_email_structure(X_train[i]) == 'text/html']


# %%
html_to_text(X_train[idx_html[5]].get_content().strip())


# %%
#check if any EmailMessage objects have more tha one text/html type occuring
html_check = list()
for j in range(len(X_train)):
    i = 0
    for part in X_train[j].walk():
        if part.get_content_type() == 'text/plain':
            i += 1
        if i == 2:
            html_check.append(j)
    
html_check


# %%
def email_to_text(email):
    html = None
    for sub_email in email.walk():
        content_type = email.get_content_type()
        if content_type not in ('text/html','text/plain'):
            continue
        try:
            content = sub_email.get_content()
        except:
            content = str(sub_email.get_payload())
        if content_type == 'text/plain':
            return content
        else:
            html = content
    if html:
        return html_to_text(html)


# %%
#converts emails in the body to "EMAIL"
import re
def convert_email_tags(email_text):
    return re.sub(r'([a-zA-Z0-9\._-]+@[a-zA-Z0-9\._-]+\.[a-zA-Z0-9\._-]+)',' EMAIL ',email_text,flags = re.I|re.S|re.M)


# %%
#remove stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(email_text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(email_text.strip())
    return [i for i in words if i not in stop_words]


# %%
#stem the words to get the root of words
from nltk.stem import PorterStemmer

def stemmer(email_word_list):
    ps = PorterStemmer()
    return [ps.stem(w) for w in email_word_list]


# %%
from sklearn.base import BaseEstimator,TransformerMixin
import urlextract
import re
class EmailToWordCounts(BaseEstimator,TransformerMixin):
    def __init__(self,lower_case = True, remove_email = True, remove_punctuation = True, remove_urls = True, stemming = True, remove_stopwords = True, remove_numbers = True):
         self.lower_case =lower_case
         self.remove_email = remove_email
         self.remove_punctuation = remove_punctuation
         self.remove_urls = remove_urls
         self.stemming = stemming
         self.remove_stopwords = remove_stopwords
         self.remove_numbers = remove_numbers
    def fit(self,X,y = None):
        return self
    def transform(self,X,y = None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.remove_urls:
                url_extractor = urlextract.URLExtract()
                urls = url_extractor.find_urls(text)
                for url in urls:
                    text = text.replace(url,' URL ')
            if self.remove_punctuation:
                text = re.sub(r'[^a-zA-Z0-9_]',' ', text, flags = re.M|re.S|re.I)
            if self.remove_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text, flags = re.I|re.S|re.M)
            if self.remove_email:
                text = convert_email_tags(text)
            if self.remove_stopwords:
                #words without stemming
                word_list = remove_stopwords(text)
            if self.stemming:
                stemmed_list = stemmer(word_list)
            X_transformed.append(stemmed_list)
        return X_transformed
    


# %%
emailtowords = EmailToWordCounts()
test = emailtowords.fit_transform(X_train[:100])


# %%
test[41]


# %%
X_train[41].get_payload()[2].get_content_type()


# %%
print(email_to_text(X_train[41]))


# %%
import urlextract
url_extractor = urlextract.URLExtract()
print(url_extractor.find_urls("facebook.com and reddit.com and blah.come and https://youtu.be/7Pq-S557XQU?t=3m32s"))


# %%
exp = "I'm @ bashs & euedh leh_ss"
res = re.sub(r'[^a-zA-Z0-9]+',' punc ',exp,flags = re.I|re.M|re.S)
res2 = re.sub(r'\W+',' punc ',exp,flags = re.I|re.M|re.S)


# %%
res2


# %%
html_idx = [idx for idx in range(len(X_train)) if X_train[idx].get_content_type() == 'text/html']


# %%
import nltk


# %%
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# %%
stop_words = set(stopwords.words("english"))


# %%
html_to_text(X_train[html_idx[0]].get_content()).strip()


# %%
[w for w in word_tokenize(html_to_text(X_train[html_idx[4]].get_content()).strip()) if w not in stop_words]


# %%



