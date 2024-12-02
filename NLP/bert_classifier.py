import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts   #input text data 
        self.labels = labels # labels
        self.tokenizer = tokenizer # BERT
        self.max_length = max_length # max sequence length 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # check if  index is out of range 
        if idx >= len(self.texts):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.texts)}")
            
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # return processed data 
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad() #zero gradients
        
        # forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item() # accumulate loss
        
        loss.backward() #backward pass
        optimizer.step() # update 
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    true_labels = []
    predictions = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_with_early_stopping(model, train_loader, val_loader, optimizer, device, max_epochs=20, patience=4):
    best_val_loss = float('inf') # best validation loss
    no_improve_count = 0 # counter for no improvement 
    
    
    # train loop
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}")
        #epoch train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"Validation metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")

        #early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return best_val_loss

def prepare_data(file_path, sample_size=0.1):
 
    #Read data
    df = pd.read_csv(file_path)
    
    # ham/spam  - > 0/1
    df['Category'] = (df['Category'] == 'spam').astype(int)
    
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    sampled_df = df.groupby('Category', group_keys=False).apply(
        lambda x: x.sample(frac=sample_size, random_state=42)
    ).reset_index(drop=True)
    

    # print(f"Original dataset size: {len(df)}")
    # print(f"Sampled dataset size: {len(sampled_df)}")
    # print(f"Label distribution :\n{sampled_df['Category'].value_counts()}")
    
    return sampled_df

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    #hyperparameters
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    MAX_EPOCHS = 20
    LEARNING_RATES = [2e-5, 3e-5, 5e-5]
    N_SPLITS = 5
    
 
    try:
        df = prepare_data('/Users/yangwenyu/Desktop/CS5100/Hw08/Data.csv', sample_size=0.2)  
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # split train sets
    train_df, test_df = train_test_split(df, test_size=0.01, random_state=42)
    
    #  initialize
    train_texts = train_df['Message'].values
    train_labels = train_df['Category'].values
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # cross-validation 
    print("Starting cross-validation for learning rate selection...")
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    best_lr = None
    best_avg_loss = float('inf')
    
    # test each learning rate 
    for lr in LEARNING_RATES:
        print(f"\nTesting learning rate: {lr}")
        fold_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_texts), 1):
            print(f"\nFold {fold}")
            
            fold_train_texts, fold_val_texts = train_texts[train_idx], train_texts[val_idx]
            fold_train_labels, fold_val_labels = train_labels[train_idx], train_labels[val_idx]

            train_dataset = TextDataset(fold_train_texts, fold_train_labels, tokenizer, MAX_LENGTH)
            val_dataset = TextDataset(fold_val_texts, fold_val_labels, tokenizer, MAX_LENGTH)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

            # initialize model 
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2
            ).to(device)

            optimizer = AdamW(model.parameters(), lr=lr)

            #train (early stop )
            best_val_loss = train_with_early_stopping(
                model, train_loader, val_loader, optimizer, device,
                max_epochs=MAX_EPOCHS, patience=4
            )
            
            fold_losses.append(best_val_loss)
        
        avg_loss = np.mean(fold_losses)
        print(f"\nAverage loss for learning rate {lr}: {avg_loss:.4f}")
        
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_lr = lr
    
    print(f"\nBest learning rate found: {best_lr}")
    
    

    final_train_texts, final_val_texts, final_train_labels, final_val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )

    train_dataset = TextDataset(final_train_texts, final_train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(final_val_texts, final_val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    final_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    ).to(device)
    
    optimizer = AdamW(final_model.parameters(), lr=best_lr)
    
    # final model 
    train_with_early_stopping(
        final_model, train_loader, val_loader, optimizer, device,
        max_epochs=MAX_EPOCHS, patience=4
    )
    
    #evaluate on test set 
    print("\nEvaluating on test set...")
    test_dataset = TextDataset(test_df['Message'].values, test_df['Category'].values, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    test_metrics = evaluate(final_model, test_loader, device)
    print("\nTest set metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()