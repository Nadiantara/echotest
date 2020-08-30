import numpy as np
# check your pickle compability, perhaps its pickle not pickle5
import pickle5 as pickle
import pandas as pd
import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import zscore
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,classification_report

#I will add more buttons and input value

#sidebar header
st.sidebar.header('User Input Features')

# This webb app will contain analysis from ECHO and Pharmacy2U, right now only ECHO available
selected_APP = st.sidebar.selectbox('APP Name', ['Echo-NHS', 'Pharmacy2U' ])
selected_store = st.sidebar.selectbox('OS Name', ['Apple', 'Android' ])
selected_solver = st.sidebar.selectbox('Solver', ['liblinear', 'lbfgs', 'saga', 'newton-cg', 'sag' ])


#header description
st.write("""
# Topic Severance Predictor App
""")

st.write("""
### Rating for each version
""")

# Display version plot for each OS
def version_plot(selected_store):
  if selected_store == 'Apple':
    from echoreview import apple_plot
    apple_plot()
    st.pyplot()
  else:
    from echoreview import google_plot
    google_plot()
    st.pyplot()

version_plot(selected_store)


#default echo function
def main_echo():
  #reading dataset
  #pickle dataframe type
  df = pd.read_csv("labeled_negative_reviews_with_versions_ratings_type.csv")
  combined_df = pd.read_pickle('combined_df.pkl')
  combined_df_preprocessed = pd.read_pickle('data_clean_and_mapped.pkl')

  #pickle non dataframe type
  x_train_count = pickle.load(open("x_train_count.pkl", "rb"))
  x_test_count = pickle.load(open("x_test_count.pkl", "rb"))

  x_train_tfidf = pickle.load(open("x_train_tfidf.pkl", "rb"))
  x_test_tfidf = pickle.load(open("x_test_tfidf.pkl", "rb"))

  y_train = pickle.load(open("y_train.pkl", "rb"))
  y_test = pickle.load(open("y_test.pkl", "rb"))
  #get similiar index
  similar_index=combined_df[combined_df['review'].isin(df['review'])].index

  # our default solver is 'liblinear'
  if selected_solver == 'liblinear':
    final_df = pd.read_pickle('final_df.pkl')
    from echoreview import get_summed_scores

    #assign each sum result with their label into a dictionary
    sum_dict=get_summed_scores(final_df,df.columns[1:7].values.astype(str))
    #Sort the value 
    sum_dict={k: v for k, v in sorted(sum_dict.items(), key=lambda item: item[1])}
    sum_df = pd.DataFrame.from_dict([sum_dict])
    st.dataframe(sum_df)
    # plot the labeled topic severance with barplot
    st.header('Feature Severance')
    plot_dict(sum_dict)
    st.pyplot(bbox_inches='tight')
  else:
    log_r=LogisticRegression(random_state=2020,class_weight='balanced',solver=selected_solver) 
    log_r.fit(x_train_tfidf,y_train)
    log_r_pred=log_r.predict(x_test_tfidf)

    #assign linear regression coefficient of each word as individual word score and put it into a dictionary and dataframe 
    vocab_size=x_train_count.shape[1]
    features=pickle.load(open("feature_names.pkl", "rb"))
    coeffs=log_r.coef_
    result_dict=dict(tuple([(features[i],coeffs[0][i]) for i in range(vocab_size)]))
    result_df=pd.DataFrame(sorted(result_dict.items(),key=lambda x:x[1]),columns=['word','coeff'])
    #find outlier
    from echoreview import find_outliers
    lengths=combined_df_preprocessed['review'].apply(lambda x: len(x.split(" "))).values
    lower_outlier,upper_outlier=find_outliers(lengths)
    combined_df_preprocessed['z_score']=zscore(lengths)
    combined_df_preprocessed['lengths']=lengths

    # constructing another one column to previous combined dataset which contain its "zscore" probability
    zscores=combined_df_preprocessed['z_score'].values
    combined_df_preprocessed['probabilities']=[0]*len(combined_df_preprocessed)
    combined_df_preprocessed['probabilities']=combined_df_preprocessed['z_score'].apply(lambda x:1-stats.norm.cdf(x))

    #take a look for the score and assign it to a new dataframe
    from echoreview import lookup
    score=lookup(combined_df_preprocessed,result_dict)
    log_r_scored_df=combined_df_preprocessed.copy()
    log_r_scored_df['score']=score

    #scaling: divide outlier with its zscore
    from echoreview import outlier_scaling
    log_r_scored_df_scaled=outlier_scaling(log_r_scored_df,upper_outlier,lower_outlier)


    #get the final dataframe
    combined_df['score']=log_r_scored_df_scaled['score'] 
    final_df=pd.merge(combined_df.loc[similar_index][['review','score']],df,how='inner',on='review')
    # fill null with zero
    final_df.fillna(0,inplace=True)

    from echoreview import get_summed_scores
    #assign each sum result with their label into a dictionary
    sum_dict=get_summed_scores(final_df,df.columns[1:7].values.astype(str))
    #Sort the value 
    sum_dict={k: v for k, v in sorted(sum_dict.items(), key=lambda item: item[1])}
    sum_df = pd.DataFrame.from_dict([sum_dict])
    st.dataframe(sum_df)
    # plot the labeled topic severance with barplot
    st.header('Feature Severance')
    plot_dict(sum_dict)
    st.pyplot(bbox_inches='tight')

# for Pharmacy2U

def main_pharmacy2u():
  #will added later
  pass

# function for plotting topic severance
def plot_dict(dict_value):
  sns.set(style="whitegrid", context="poster",font_scale=0.6)
  fig,ax=plt.subplots(figsize=(15,8))
  bar=sns.barplot(x=list(dict_value.keys()),y=list(dict_value.values()),ax=ax,palette='rocket')
  bar.set(xlabel='Categories',ylabel='Severance Score')
  plt.show()
  return bar


#######################################################################################
if __name__=='__main__':
    main_echo()