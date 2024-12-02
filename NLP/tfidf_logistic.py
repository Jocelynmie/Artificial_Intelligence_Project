import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

class TfidfVectorizer:
    def __init__(self, min_df=2):
        self.min_df = min_df
        self.vocabulary = {}  # vocabulary mapping
        self.doc_freq = {}    # document frequency
        self.n_docs = 0       # total number of documents
        
    def tokenize(self, text):
        return [w.lower() for w in re.split(r'\W+', str(text)) if w]
    
    def fit(self, documents):
        #Count document frequency for each word
        word_doc_freq = defaultdict(int)
        
        #Iterate through all documents
        self.n_docs = len(documents)
        for doc in documents:
            #Get unique words 
            words = set(self.tokenize(doc))
            for word in words:
                word_doc_freq[word] += 1
        
        #Build vocabulary 
        vocab_words = [word for word, freq in word_doc_freq.items() 
                      if freq >= self.min_df]
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab_words))}
        self.doc_freq = {word: freq for word, freq in word_doc_freq.items() 
                        if word in self.vocabulary}
        
        return self
    
    def transform(self, documents):
        X = np.zeros((len(documents), len(self.vocabulary)))
        
        for doc_idx, doc in enumerate(documents):
            #TF
            word_counts = Counter(self.tokenize(doc))
            doc_len = sum(word_counts.values())
            
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    #TF
                    tf = count / doc_len
                    #IDF
                    idf = np.log(self.n_docs / (self.doc_freq[word] + 1)) + 1
                    #TF-IDF
                    X[doc_idx, word_idx] = tf * idf
        
        return X

def main():
    #Load data
    df = pd.read_csv('/Users/yangwenyu/Desktop/CS5100/Hw08/Data.csv')
    
    #Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'], 
        df['Category'], 
        test_size=0.2, 
        random_state=42
    )
    
    #Create and train
    vectorizer = TfidfVectorizer(min_df=2)
    X_train_tfidf = vectorizer.fit(X_train).transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    #Train logistic regression model
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_tfidf, y_train)
    
    #Predict
    y_pred = clf.predict(X_test_tfidf)
    
    #Calculate evaluation metrics
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    

    for true, pred, text in zip(y_test[:5], y_pred[:5], X_test[:5]):
        print(f"Text: {text[:50]}...")
        print(f"True Label: {true}, Predicted Label: {pred}\n")
    

    print(f"Vocabulary Size: {len(vectorizer.vocabulary)}")
    print(f"Training Set Size: {len(X_train)}")
    print(f"Test Set Size: {len(X_test)}")

if __name__ == "__main__":
    main()