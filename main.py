import nltk
import streamlit as st
nltk.download('punkt')
from nltk.util import ngrams
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
import string
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import io

#create memory buffer for download button
buffer = io.BytesIO()


st.set_page_config(page_title="Inlinking Opportunities with K-Mean Categories",page_icon="⛓💡", initial_sidebar_state="collapsed")

st.title("⛓ Inlinking with K-Mean")

st.markdown("![Alt Text](https://c.tenor.com/x5JdEdbPPkIAAAAC/link-link-thinking.gif)")

#replaced below variable with a cached wrapper
#embedder = SentenceTransformer('all-MiniLM-L6-v2')

@st.experimental_memo
def embedder_cache():
  embedder = SentenceTransformer('all-MiniLM-L6-v2')
  return embedder


def extract_ngrams(data, num):
  n_grams = ngrams(nltk.word_tokenize(data), num)
  gram_list = [ ' '.join(grams) for grams in n_grams]
  return gram_list

def getname(cluster):
  data = ''
  data = ' '.join(cluster)
  keywords = extract_ngrams(data, 1)
  stop_words = set(stopwords.words('english'))
  cluster_name = [x.lower() for x in keywords]
  cluster_name = [x for x in cluster_name if not x in stop_words]
  cluster_name = [x for x in cluster_name if x not in string.punctuation]
  cluster_name = list(Counter(cluster_name).most_common(1))
  return cluster_name

df2 = pd.DataFrame(columns = ['cluster', 'title', 'url'])

st.write(
    """
Upload your internal_html.csv file
"""
)

uploaded_file = st.file_uploader("Upload CSV", type=".csv")

if uploaded_file:

  df = pd.read_csv(uploaded_file)

#ammended dropna to include the how='all' argument for better compatibility with SF internal_html files.

  df.dropna(inplace=True,how='all')

  df['Title 1'] = df['Title 1'].replace({' \| efront':'', ' \| eFront':''}, regex=True)

  corpus = df["Title 1"].tolist()

  print(corpus)


  corpus_embeddings = embedder_cache().encode(corpus)

  # adjust this as needed
  num_clusters = 15
  clustering_model = KMeans(n_clusters=num_clusters)
  clustering_model.fit(corpus_embeddings)
  cluster_assignment = clustering_model.labels_

  clustered_sentences = [[] for i in range(num_clusters)]

  for sentence_id, cluster_id in enumerate(cluster_assignment):
      clustered_sentences[cluster_id].append(corpus[sentence_id])

  for i, cluster in enumerate(clustered_sentences):
      cluster_name = getname(cluster)
      for x in cluster:
        geturl = df[df['Title 1']==x]['Address'].values[0]
        getdict = {'cluster':cluster_name[0][0],'title':x,'url':geturl}
        print(getdict)
        df2 = df2.append(getdict, ignore_index = True)

#this bit writes df2 to the memory buffer, ready for our download button.
  with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    df2.to_excel(writer, sheet_name='Sheet1')

    writer.save()

  st.success('Done!')

  st.download_button(
    label="Download Excel worksheets",
    data=buffer,
    file_name="categorised_urls.xlsx",
    mime="application/vnd.ms-excel"
  )
