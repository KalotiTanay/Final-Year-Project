# Import Statements
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
import matplotlib.pyplot as plt
nltk.download('stopwords')


# Reading Data Files from Storage

test_file = 'test.csv'
train_file = 'train.csv'
valid_file = 'valid.csv'

train_news = pd.read_csv(train_file)
test_news = pd.read_csv(test_file)
valid_news = pd.read_csv(valid_file)


# In this section we observe the dataset while creating distribution in order to
# evenly distribute data between classes and performing an optional integrity check.


def data_observe():
    print("DataSet size:")
    print(train_news.shape)
    print(train_news.head(10))


def create_distribution(dataFile, Label):
    sb.countplot(x='Label', data=dataFile, palette='hls').set_title(Label)
    plt.show()


create_distribution(train_news, 'Training Distribution')
create_distribution(test_news, 'Testing Distribution')
create_distribution(valid_news, 'Valid Distribution')


def data_check():
    print("Checking Quality of Training Data...")
    train_news.isnull().sum()
    train_news.info()
    print("Checking Completed.")
    data_observe()


data_check()
# "Module ready" Use the // data_check() // command in order to perform checking.

# STEMMING SECTION

eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))


def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed


# processing data

def process_data(data, exclude_stopword=True, stem=True):
    tokens = [w.lower() for w in data]
    tokensStemmed = tokens
    tokensStemmed = stem_tokens(tokens, eng_stemmer)
    tokensStemmed = [w for w in tokensStemmed if w not in stopwords]
    return tokensStemmed


# Creating ngrams
# UNIGRAM
def create_unigram(words):
    assert type(words) == list
    return words


# BIGRAM
def create_bigram(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 1:
        lst = []
        for i in range(Len - 1):
            for k in range(1, skip + 2):
                if i + k < Len:
                    lst.append(join_str.join([words[i], words[i + k]]))
    else:  # set as unigram
        lst = create_unigram(words)
    return lst


# PORTER SECTION
porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
