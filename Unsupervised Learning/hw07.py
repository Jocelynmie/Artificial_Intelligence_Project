import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
nltk.download('stopwords')
from sklearn.cluster import KMeans

# Read Data
true_news = pd.read_csv('/Users/yangwenyu/Desktop/CS5100/Hw07/True.csv')
fake_news = pd.read_csv('/Users/yangwenyu/Desktop/CS5100/Hw07/Fake.csv')
true_news['label'] = 1  
fake_news['label'] = 0  
news_df = pd.concat([true_news, fake_news], ignore_index=True)

# Preprocessing 
def preprocess_text(text):
    return str(text).lower()

news_df['text'] = news_df['text'].apply(preprocess_text)
print(news_df.shape)
print("\nFrist 5 rows:")
print(news_df.head())

# LDA topic modeling
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(news_df['text'])

# train (k=10)
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda_output = lda.fit_transform(doc_term_matrix)

# dispaly keywords
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-20-1:-1]]
    print(f"\nTopic {topic_idx}:")
    print(", ".join(top_words))

# randomly select 5 real news examples and 5 fake news
# examine the topic distributions for each document
real_samples = news_df[news_df['label'] == 1].sample(5, random_state=42)
fake_samples = news_df[news_df['label'] == 0].sample(5, random_state=42)

print("\nReal news sample topic distribution:")
real_sample_vectors = lda.transform(vectorizer.transform(real_samples['text']))
print(real_sample_vectors)

print("\nFake news sample topic distribution:")
fake_sample_vectors = lda.transform(vectorizer.transform(fake_samples['text']))
print(fake_sample_vectors)



#Split training and test sets
X = lda_output  
y = news_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train logistic regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Model evaluation
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)
print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# Analyze topic importance

# Positive values: more likely to be real news
# Negative: more likely to be fake news
# The larger the abs : the more important the impact
for topic_idx, coef in enumerate(lr.coef_[0]):
    print(f"\nTopic {topic_idx} coefficient: {coef:.3f}")
    top_words = [feature_names[i] for i in lda.components_[topic_idx].argsort()[:-10-1:-1]]
    print(f"Keywords: {', '.join(top_words)}")




#Choice fake and real news
real_news_indices = news_df[news_df['label'] == 1].index
real_news_vectors = lda_output[real_news_indices]

# K-means k=10
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(real_news_vectors)

# for each cluster
for cluster_id in range(10):
    # get 5 new doc
    cluster_indices = [i for i, x in enumerate(clusters) if x == cluster_id][:5]
    print(f"\nCluster {cluster_id}:")
    for idx in cluster_indices:
        print(news_df.iloc[real_news_indices[idx]]['title'])