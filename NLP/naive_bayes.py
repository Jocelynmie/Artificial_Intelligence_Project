import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# read data
df = pd.read_csv('/Users/yangwenyu/Desktop/CS5100/Hw08/Data.csv')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# calculate prior probabilities
N = len(train_df)
Pham = len(train_df[train_df['Category'] == 'ham']) / N
Pspam = len(train_df[train_df['Category'] == 'spam']) / N

# initialize word frequency dictionaries
word_count = {}      
spam_word_count = {} 
ham_word_count = {}  

# raining phase
for index, row in train_df.iterrows():
    document = row['Message'] #get email content 
    category = row['Category'] #email category 
    tokenized_document = re.split(r"\W+", document.lower())  #lowercase
    #count frequencies  
    for word in tokenized_document:
        if word:  #ignore empty str
            word_count[word] = word_count.get(word, 0) + 1
            if category == 'spam':
                spam_word_count[word] = spam_word_count.get(word, 0) + 1
            else:
                ham_word_count[word] = ham_word_count.get(word, 0) + 1

#count size 
V = len(word_count)
spam_total_words = sum(spam_word_count.values())
ham_total_words = sum(ham_word_count.values())

#testing 
predictions = []
true_labels = []

for index, row in test_df.iterrows():
    document = row['Message']
    true_category = row['Category']
    true_labels.append(true_category)
    
    # log 
    log_p_spam = np.log(Pspam)
    log_p_ham = np.log(Pham)
    
    # for each word ,calculate conditional prob
    for word in re.split(r"\W+", document.lower()):
        if word:  
            # Laplace
            p_word_spam = (spam_word_count.get(word, 0) + 1) / (spam_total_words + V)
            p_word_ham = (ham_word_count.get(word, 0) + 1) / (ham_total_words + V)
            
            #log 
            log_p_spam += np.log(p_word_spam)
            log_p_ham += np.log(p_word_ham)
    
    #predict category 
    predicted_category = 'spam' if log_p_spam > log_p_ham else 'ham'
    predictions.append(predicted_category)

#calculate evaluation metrics
precision = precision_score(true_labels, predictions, pos_label='spam')
recall = recall_score(true_labels, predictions, pos_label='spam')
f1 = f1_score(true_labels, predictions, pos_label='spam')


print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


for i in range(min(5, len(predictions))):
    print(f"True category : {true_labels[i]}, predicted category: {predictions[i]}")