import os
import random
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

random.seed(1)

MDA_FILES = os.listdir("./data/mda/")
MDA_FILES = [file for file in MDA_FILES if file != ".DS_Store"]

random_file_idx = random.randint(1, len(MDA_FILES))
random_file = MDA_FILES[random_file_idx]

def clean_text(mda_file):
    stopwords_nltk = set(stopwords.words('english'))
    with open(mda_file) as file:
        mda_text = file.read()
    words = mda_text.split(" ")
    alphabetic_only = [word.lower() for word in words if word.isalpha()]
    cleaned_words = [word for word in alphabetic_only if word not in stopwords_nltk]
    return cleaned_words

positive_lexicon = set(opinion_lexicon.positive())
negative_lexicon = set(opinion_lexicon.negative())

sentiment_scores = {}
error_files = []
mda_text = {}
for file in MDA_FILES:
    cik = file.split('_')[0]
    filing_date = file.split('_')[2].split('.')[0]
    cik_filing_date = cik + '_' + filing_date
    with open('./data/mda' +file) as f:
            text = f.read()
    mda_text[cik_filing_date]= text

df_text = pd.DataFrame(mda_text,index=['text']).T
df_text.reset_index(inplace=True)
df_text.columns = ['cik_filing_date','mda_text']

df_text['cik'] = df_text['cik_filing_date'].apply(lambda x: x.split('_')[0])
df_text['cik'] = pd.to_numeric(df_text['cik'])

df_text['filing_date'] = df_text['cik_filing_date'].apply(lambda x: x.split('_')[1])
df_text['filing_date'] = pd.to_datetime(df_text['filing_date'],format='%Y-%m-%d')

df_text.drop(columns=['cik_filing_date'],inplace=True)
df_text = df_text[['cik','filing_date','mda_text']]
df_text['mda_text_clean'] = df_text['mda_text'].apply(clean_text)
df_text['num_cleaned_words'] = df_text['mda_text_clean'].apply(lambda x: len(x))
df_text['mda_text_clean_str'] = df_text['mda_text_clean'],apply(lambda x: " ".join(x))

count_vec = CountVectorizer(vocabulary=positive_lexicon)
dtm_pos_words = count_vec.fit_transform(df_text['mda_text_clean_str'])
df_text['positive_count'] = df_dtm_pos_words.sum(axis=1)

#
#     cleaned_words = clean_text(f"./data/mda/{file}")
#     positive_sentiment = 0
#     negative_sentiment = 0
#
#     if len(cleaned_words) >= 100:
#         for word in cleaned_words:
#             if word in positive_lexicon:
#                 positive_sentiment += 1
#             elif word in negative_lexicon:
#                 negative_sentiment += 1
#
#         phi_pos = positive_sentiment / len(cleaned_words)
#         phi_neg = negative_sentiment / len(cleaned_words)
#
#         phi_npt = (phi_pos - phi_neg) / (phi_pos + phi_neg)
#         sentiment_scores[file] = [phi_pos,phi_neg,phi_npt,len(cleaned_words)]
#
#     else:
#         pass
#
# df_scores = pd.DataFrame(sentiment_scores).T
# df_scores.reset_index(inplace=True)
# df_scores.columns = ['file_name','phi_pos','phi_neg','phi_npt','num_cleaned_words']
# df_scores['phi_npt'].hist(figsize=(12,8),bins=100)
# plt.show()

