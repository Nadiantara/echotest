#basic
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import zscore
import scipy.stats as stats
#scraping
import requests
import json
from datetime import datetime,timedelta, date
from distutils.version import LooseVersion
import math
from tqdm import tqdm
from google_play_scraper import app, reviews, reviews_all, Sort
# for NLP preprocessing
import nltk
import spacy
import pickle5 as pickle
import gensim 
# for main model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,classification_report


# Echo apple scrapper only being run in app.py
def apple_scrapper():

  APPID = 1031922175
  COUNTRY = "GB"

  STOREURL = f'http://apps.apple.com/{COUNTRY}/app/id{APPID}'
  res = requests.get(STOREURL)
  if res.status_code == 200:
      appname = re.search('(?<="name":").*?(?=")', res.text).group(0)
      print(appname)

  #extracting from appstore
  def extract_itunes(app_id, country="US", pages=1, save=True):
      for i in range(pages):
          URL = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/page={i+1}/sortBy=mostRecent/json"
          res = requests.get(URL)
          if res.status_code == 200:
              entries = res.json()['feed']['entry']
              if i == 0:
                  df = pd.DataFrame(columns=list(entries[1].keys())[0:6], index=range(len(entries)))

              for j, entry in enumerate(entries):
                  for column in df.columns:  
                      try:
                          df.loc[j, column] = entry[column]['name']['label']
                      except:
                          df.loc[j, column] = entry[column]['label']

              df.set_index('id', inplace=True, drop=False)
      
      df.drop('id', axis=1, inplace=True)
      
      if save:
          datenow = datetime.today().strftime('%d/%m/%Y').replace('/', '')
          filename = f'{app_id}_{country}_{datenow}_AAS.csv'
          df.to_csv(f"{filename}")
      
      df.index = df.index.astype('int64')
      apple_df = df
      return apple_df
          
  #Have to make the date suitable first. Make sure to adjust according to what you have,
  #E.g if today's date is 23/8/2020, and your old date is 18/8/2020, then make sure timedelta =5
  apple_df = extract_itunes(APPID, COUNTRY, pages=10)
  old_date=datetime.today() - timedelta(days=1)#specifiy the file name of existing file
  old_date=old_date.strftime('%d/%m/%Y').replace('/', '')
  existing_file=f'{APPID}_{COUNTRY}_overall_{old_date}.csv' 

  def updating_apple(existing_file, apple_df):
    datenow = datetime.today().strftime('%d/%m/%Y').replace('/', '')
    filename=f'{APPID}_{COUNTRY}_overall_{datenow}.csv'

    existing = pd.read_csv(existing_file, index_col="id")
    if apple_df.index.dtype == existing.index.dtype:
        # perform an SQL upsert equivalent
        newrecords = sum(~apple_df.index.isin(existing.index))
        existing = pd.concat([apple_df[~apple_df.index.isin(existing.index)],existing])
        existing.to_csv(f"{filename}")
    return filename

  #use the function to update   
  filename = updating_apple(existing_file, apple_df)
  def apple_plot(filename):
    apple_df=pd.read_csv(f"{filename}")
    pd.crosstab(apple_df['im:rating'], apple_df['im:version'], margins=True)
    ct = pd.crosstab(apple_df['im:version'], apple_df['im:rating'])
    plot = ct.tail(15).plot(kind="bar",figsize=(15, 3))
    plot.set_xlabel("version")
    plot.set_ylabel("rating")
    return plot
  
  apple_plot(filename)


# Echo google scrapper, only being run in app.py

def google_scrapper():
  PLAYSTORE_ID = 'echo.co.uk'
  COUNTRY = 'gb'
  BATCH_SIZE = 50
  MAX_REVIEWS = 50000
  appinfo = app(PLAYSTORE_ID,lang='en',country=COUNTRY)
  AVAIL_REVIEWS = appinfo.get('reviews')
  TOFETCH_REVIEWS = min(AVAIL_REVIEWS, MAX_REVIEWS)
  t = tqdm(total=TOFETCH_REVIEWS)

  for i in range(TOFETCH_REVIEWS//BATCH_SIZE):
      if i == 0:
          result, continuation_token = reviews(PLAYSTORE_ID, 
                                              count=BATCH_SIZE,
                                              country=COUNTRY
                                              )         
      res, continuation_token = reviews(PLAYSTORE_ID, count=BATCH_SIZE, continuation_token=continuation_token)
      result.extend(res)
      t.update(BATCH_SIZE)
  t.close()
  df_google_ps = pd.DataFrame(result)
  df_google_ps.drop_duplicates('reviewId', inplace=True)
  df_google_ps = df_google_ps.set_index('reviewId')
  datenow = datetime.today().strftime('%d/%m/%Y').replace('/', '')
  filename = f'{PLAYSTORE_ID}_{COUNTRY}_{datenow}_GPS.csv'
  df_google_ps.to_csv(f"{filename}") 

  #update oldfile
  old_date=datetime.today() - timedelta(days=1)#specifiy the file name of existing file
  old_date=old_date.strftime('%d/%m/%Y').replace('/', '')
  existing_file=f'{PLAYSTORE_ID}_{COUNTRY.upper()}_overall_{old_date}.csv'
  existing = pd.read_csv(existing_file, index_col='reviewId')

  newfilename= f'{PLAYSTORE_ID}_{COUNTRY.upper()}_overall_{datenow}.csv' 

  #Basically filename of the df_google_ps after upsert
  if df_google_ps.index.dtype == existing.index.dtype:
    print("if statement ran")
    # perform an SQL upsert equivalent
    newrecords = sum(~df_google_ps.index.isin(existing.index))
    existing = pd.concat([df_google_ps[~df_google_ps.index.isin(existing.index)],existing])
    print(f'{newrecords} new records added. {existing.shape} rows post upsert.')

    existing.to_csv(f"{newfilename}")

  #get the plot
  def google_plot(filename):
    google_df=pd.read_csv(f"{filename}")
    pd.crosstab(google_df['score'], google_df['reviewCreatedVersion'], margins=True)
    ct = pd.crosstab(google_df['reviewCreatedVersion'], google_df['score'])
    plot = ct.tail(15).plot(kind="bar",figsize=(15, 3))
    plot.set_xlabel("version")
    plot.set_ylabel("rating")
    return plot

  google_plot(newfilename)


#loading data, optional if scrapper isnt called by scraper
df = pd.read_csv("labeled_negative_reviews_with_versions_ratings_type.csv")
df_apple = pd.read_csv("1031922175_GB_overall_28082020.csv")
df_google = pd.read_csv("echo.co.uk_GB_overall_28082020.csv")
df_apple["content"] = df_apple["title"].astype(str)+" "+df_apple["content"].astype(str)

#for this better make a pd df of combined reviews first
review_rating_dict={
  'review': list(df_apple['content'].values) + list(df_google['content'].values),
  'rating': list(df_apple['im:rating'].values) + list(df_google['score'].values)
}
combined_df= pd.DataFrame(review_rating_dict)
combined_df.drop_duplicates(['review'],inplace=True)
combined_df.reset_index(inplace=True,drop=True)

combined_df.to_pickle("combined_df.pkl")
#get similiar index
similar_index=combined_df[combined_df['review'].isin(df['review'])].index

# get words length of each review (nadi)
raw_lengths=combined_df['review'].apply(lambda x: len(x.split(" "))).values

#removing emoji
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text=regrex_pattern.sub(r'',text)
    text=text.replace('\n',' ')
    text=re.sub(' +', ' ', text)
    
    return text

# contractions dictionary
contractions = { 
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i shall have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have"
    }

# expand contraction with this function
def expand_contractions(entry):
    entry=entry.lower()
    entry=re.sub(r"’","'",entry)
    entry=entry.split(" ")

    for idx,word in enumerate(entry):
        if word in contractions:
            
            entry[idx]=contractions[word]
    return " ".join(entry)

# remove punctuation with this function
def remove_punctuation(entry):
  entry=re.sub(r"[^\w\s]"," ",entry)
  return entry


from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
 
stop_words = stopwords.words('english')


# remove stopwords with this function
def remove_stop_words(sentence):
    dummy=sentence.split(" ")
    out=[i for i in dummy if i not in stop_words]
    return " ".join(out)

# remove short words with this function
def remove_short_words(sentence):
  dummy=sentence.split(" ")
  out=[i for i in dummy if len(i)>3 ]
  return " ".join(out)

# spacy method for lemmatization
nlp = spacy.load('en', disable=['parser', 'ner'])

# lemmatizer function
def lemmatize(reviews):
    doc = nlp(reviews)
    
    return " ".join([token.lemma_ for token in doc])

# combine all functions above to one "preprocess_data" function below
def preprocess_data(df):
  df=df.copy()
  df['review']=df['review'].apply(lambda x: x.lower()) #lowering case
  df['review']=df['review'].apply(deEmojify)           #removing emoji
  df['review']=df['review'].apply(expand_contractions)   
  df['review']=df['review'].apply(remove_stop_words)   
  df['review']=df['review'].apply(remove_punctuation)  
  df['review']=df['review'].apply(lambda s: re.sub(r"[^a-zA-Z]"," ",s)) #keep only alphabetical words
  df['review']=df['review'].apply(lambda s:re.sub(' +', ' ', s))   #remove + sign
  df['review']=df['review'].apply(lemmatize) #apply lemmatizer
  df['review']=df['review'].apply(remove_short_words)
  df.drop(df[df['review'].apply(lambda x: len(x)==0)].index,inplace=True)
  #df.reset_index(inplace=True,drop=True) 

  return df

combined_df_preprocessed=preprocess_data(combined_df)

#NGram analysis

#keep the index this way, will use the indexes later
pos=combined_df_preprocessed[combined_df_preprocessed['rating'].isin([4,5])] #positive reviews
neg=combined_df_preprocessed[combined_df_preprocessed['rating'].isin([1,2,3])] #negative reviews

#tokenizing the separated reviews
pos_lemma=[val.split(" ") for val in pos['review'].values]
neg_lemma=[val.split(" ") for val in neg['review'].values]

#construct negative bigram with minimal frequency = 3, and minimum score(threshold)= 6
#construct positive bigram with minimal frequency = 3, and minimum score(threshold)= 10
def bigram(neg_lemma, pos_lemma):
  neg_bigrams=gensim.models.Phrases(neg_lemma,min_count=3,threshold=6)
  neg_bigram_reviews=neg_bigrams[neg_lemma] 
  pos_bigrams=gensim.models.Phrases(pos_lemma,min_count=3,threshold=10)
  pos_bigram_reviews=pos_bigrams[pos_lemma] 
  return pos_bigrams, pos_bigram_reviews, neg_bigrams, neg_bigram_reviews

pos_bigrams, pos_bigram_reviews, neg_bigrams, neg_bigram_reviews = bigram(neg_lemma, pos_lemma)



#construct positive trigram with minimal frequency = 2, and minimum score(threshold)= 15
def trigram(pos_bigram_reviews, neg_bigram_reviews):
  pos_trigrams=gensim.models.Phrases(pos_bigram_reviews,min_count=2,threshold=15)
  pos_trigram_reviews=pos_trigrams[pos_bigram_reviews]
  neg_trigrams=gensim.models.Phrases(neg_bigram_reviews,min_count=2,threshold=8)
  neg_trigram_reviews=neg_trigrams[neg_bigram_reviews]
  return pos_trigrams, pos_trigram_reviews, neg_trigrams, neg_trigram_reviews


pos_trigrams, pos_trigram_reviews, neg_trigrams, neg_trigram_reviews = trigram(pos_bigram_reviews, neg_bigram_reviews)

# replace initial positive review with "grammed" positive review
# replace initial negative review with "grammed" negative review
def replace_gram(combined_df_preprocessed, neg_trigram_reviews, pos_trigram_reviews, neg_index, pos_index):
  dummy=[" ".join(tri) for tri in neg_trigram_reviews]
  pointer=0
  for idx in neg_index:
    combined_df_preprocessed['review'][idx]=dummy[pointer]
    pointer+=1

  dummy2=[" ".join(tri) for tri in pos_trigram_reviews]
  pointer=0
  for idx2 in pos_index:
    combined_df_preprocessed['review'][idx2]=dummy2[pointer]
    pointer+=1
  return  combined_df_preprocessed

combined_df_preprocessed = replace_gram(combined_df_preprocessed, neg_trigram_reviews, pos_trigram_reviews, neg.index, pos.index)

# map rating into positive (1) and negative (0) values for later logistic regression analysis
mapping={4:1,
         5:1,
         3:0,
         2:0,
         1:0
        }
        
combined_df_preprocessed['rating']=combined_df_preprocessed['rating'].map(lambda x: mapping[x])

#put to pickle for later use
combined_df_preprocessed.to_pickle("data_clean_and_mapped.pkl")



# split dataset into train and test set
df_train, df_test=train_test_split(combined_df_preprocessed,test_size=0.05,
                                  stratify=combined_df_preprocessed.rating.values,random_state=2020)

# assigning each train and test axis
x_train=df_train['review'].values
x_test=df_test['review'].values
train_index=df_train['review'].index
test_index=df_train['review'].index
y_train=df_train['rating'].values
y_test=df_test['rating'].values

# transforming every train and test reviews based on its frequency
cv=CountVectorizer()
x_train_count=cv.fit_transform(x_train)
x_test_count=cv.transform(x_test)
feature_names = cv.get_feature_names()
# Vectorize words in each review
tfidf=TfidfTransformer()
x_train_tfidf=tfidf.fit_transform(x_train_count)
x_test_tfidf=tfidf.transform(x_test_count)

# put it to pickle for later use in webapp
def pickle_save(y_train, y_test, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, feature_names):
  pickle.dump(y_train, open("y_train.pkl", "wb"))
  pickle.dump(y_test, open("y_test.pkl", "wb"))

  pickle.dump(x_train_count, open("x_train_count.pkl", "wb"))
  pickle.dump(x_test_count, open("x_test_count.pkl", "wb"))

  pickle.dump(x_train_tfidf, open("x_train_tfidf.pkl", "wb"))
  pickle.dump(x_test_tfidf, open("x_test_tfidf.pkl", "wb"))
  pickle.dump(feature_names, open("feature_names.pkl", "wb"))

pickle_save(y_train, y_test, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, feature_names)

# Logistic regression
#This applies a heuristic class balancing, because we have imbalance towards the number of 1s
log_r=LogisticRegression(random_state=2020,class_weight='balanced',solver='liblinear') 
log_r.fit(x_train_tfidf,y_train)
log_r_pred=log_r.predict(x_test_tfidf)

#plot confusion matrix no need this for webapp
#rcParams['figure.figsize'] = 6.4,4.8
#plot_confusion_matrix(log_r,x_test_tfidf,y_test,values_format='.3g',cmap='cividis')
#plt.show()

#assign linear regression coefficient of each word as individual word score and put it into a dictionary and dataframe 
vocab_size=x_train_count.shape[1]
features=cv.get_feature_names()
coeffs=log_r.coef_
result_dict=dict(tuple([(features[i],coeffs[0][i]) for i in range(vocab_size)]))
result_df=pd.DataFrame(sorted(result_dict.items(),key=lambda x:x[1]),columns=['word','coeff'])

#For some reason the bigrams and trigrams did not work on the negative words, it might be because 
#the threshold I set was a little high
#note, the index here != index in the original dataframe

dummy=result_df[result_df['word'].apply(lambda x: len(x.split("_"))==3)]
dummy[dummy['coeff']<0]

#Function for finding outliers for length
def find_outliers(array):
  lower_outlier=int(np.quantile(array,0.25)-stats.iqr(array)*1.5)
  upper_outlier=int(stats.iqr(array)*1.5+np.quantile(array,0.75))
  return lower_outlier, upper_outlier

#z-statistic will be calculated from the processed combined df
lengths=combined_df_preprocessed['review'].apply(lambda x: len(x.split(" "))).values

#find the lower and upper outlier
lower_outlier,upper_outlier=find_outliers(lengths)

# add 2 columns to previous combined dataframes which contain "zscore" and its review length
combined_df_preprocessed['z_score']=zscore(lengths)
combined_df_preprocessed['lengths']=lengths

# constructing another one column to previous combined dataset which contain its "zscore" probability
zscores=combined_df_preprocessed['z_score'].values
combined_df_preprocessed['probabilities']=[0]*len(combined_df_preprocessed)
combined_df_preprocessed['probabilities']=combined_df_preprocessed['z_score'].apply(lambda x:1-stats.norm.cdf(x)) 

#df should be the original dataframe
#ref should be the reference/dictionary used so pass in result_dict
def lookup(df,ref):

  val=df['review'].values
  score_list=[]
  for i in val:
    score=0
    for j in i.split(" "):
      if j in ref:
        score+=ref[j]
      else:
        score+=0
  
    score_list.append(score)
  return score_list 


#take a look for the score and assign it to a new dataframe
score=lookup(combined_df_preprocessed,result_dict)
log_r_scored_df=combined_df_preprocessed.copy()
log_r_scored_df['score']=score


#There are two scorings in Jupyter notebook, here we just take 1 scoring method
#that more suitable (Ideas 2).


#Here should include checking if lower outlier <0, if yes then ignore this and do not do scaling
def outlier_scaling(df, upper, lower):
  #Function assumes that lengths have been added to sentences
  longg=df[df['lengths']>upper].index
  df['score'].loc[longg]=df['score'].loc[longg]/abs(df['z_score'].loc[longg])
  if lower>0:
    #so if outlier is lets say 1.6, rounded to 1, we find and scale for those sentences with lengths <2
    small=df[df['lengths']<lower+1].index
    df['score'].loc[small]=df['score'].loc[small]*abs(df['z_score'].loc[small])
  return df

# apply the function to the initial score dataframe
log_r_scored_df_scaled=outlier_scaling(log_r_scored_df,upper_outlier,lower_outlier)

#Matching the results

#replace the combined_df score with previous scaled score(Idea 2, trial 2)
combined_df['score']=log_r_scored_df_scaled['score'] 

# merge the review and score with labeled negative topic
final_df=pd.merge(combined_df.loc[similar_index][['review','score']],df,how='inner',on='review')

#get summed scores
def get_summed_scores(df,col):
  """ Inputs: 
  df--dataframe to get sum of scores from. (should be scaled)
  col--list of column names

  Outputs:
  Dictionary containing the mappings of the column names, and the severance score values """
  sum_array=[]
  for c in col:
    sum=abs(np.sum(df[df[c]==1]['score'].values))
    sum_array.append((c,sum))
  sum_dict=dict(tuple(sum_array))
  return sum_dict


# fill Null score with zero
final_df.fillna(0,inplace=True)

#pickle it
final_df.to_pickle("final_df.pkl")


