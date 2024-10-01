# %%
import pandas as pd
import json

def load_jsonl_to_df(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            json_line = json.loads(line)
            data.append(json_line)
    df = pd.DataFrame(data)
    return df

# %%
train_df = load_jsonl_to_df('/kaggle/input/hindi-headline/hi_train.jsonl')
val_df = load_jsonl_to_df('/kaggle/input/hindi-headline/hi_dev.jsonl')
test_df = load_jsonl_to_df('/kaggle/input/hindi-headline/hi_test.jsonl')

# %%
train_df

# %%
print("Training Set Size:", train_df.shape)
print("Validation Set Size:", val_df.shape)
print("Test Set Size:", test_df.shape)

# %%
original_train_size = 208091
original_val_size = 44718
original_test_size = 44475
original_total_size = original_train_size + original_val_size + original_test_size

train_size = 8000

train_ratio = original_train_size / original_total_size
val_ratio = original_val_size / original_total_size
test_ratio = original_test_size / original_total_size

val_size = int(train_size * val_ratio / train_ratio)
test_size = int(train_size * test_ratio / train_ratio)

print("New Training Set Size:", train_size)
print("New Validation Set Size:", val_size)
print("New Test Set Size:", test_size)

# %%
random_seed = 42

train_df = train_df.sample(n=train_size, random_state=random_seed)
val_df = val_df.sample(n=val_size, random_state=random_seed)
test_df = test_df.sample(n=test_size, random_state=random_seed)

# %%
import re

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df['Document'] = train_df['Document'].apply(preprocess_text)
train_df['Title'] = train_df['Title'].apply(preprocess_text)


# %%
train_df

# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm

tokenizer = T5Tokenizer.from_pretrained('t5-base')

class HindiNewsDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        document = self.dataframe.iloc[idx]['Document']
        title = self.dataframe.iloc[idx]['Title']
        source = tokenizer("summarize: " + document, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        target = tokenizer(title, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        return source.input_ids.squeeze(), target.input_ids.squeeze()

train_dataset = HindiNewsDataset(train_df)
val_dataset = HindiNewsDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained('t5-base')
model = model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(3): 
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    print(f"Total Loss: {total_loss / len(train_loader)}")

# %%
model.eval()
progress_bar = tqdm(val_loader, desc="Evaluating")
for batch in progress_bar:
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    outputs = model.generate(input_ids=input_ids)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    actual = tokenizer.decode(labels[0], skip_special_tokens=True)
    
    progress_bar.set_description(f"Generated: {generated} Actual: {actual}")

# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm


model.eval()
predictions, references = [], []

progress_bar = tqdm(val_loader, desc="Evaluating")
for batch in progress_bar:
    input_ids, labels = batch
    input_ids = input_ids.to(device)

    outputs = model.generate(input_ids=input_ids)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    actual_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    predictions.extend(generated_texts)
    references.extend(actual_texts)

formatted_references = [[ref] for ref in references]

# %%
pip install pynlpl

# %%
model.eval()
predictions, references = [], []

progress_bar = tqdm(val_loader, desc="Evaluating")
for batch in progress_bar:
    input_ids, labels = batch
    input_ids = input_ids.to(device)

    outputs = model.generate(input_ids=input_ids)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    actual_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    predictions.extend(generated_texts)
    references.extend(actual_texts)

true_dict = {i: [line] for i, line in enumerate(references)}
pred_dict = {i: [line] for i, line in enumerate(predictions)}


# %%
pip install bert-score

# %%
pip install sacrebleu

# %%
pip install pynlpl

# %%
import sacrebleu
from bert_score import score

formatted_references = [[ref] for ref in references]

chrf = sacrebleu.corpus_chrf(predictions, formatted_references)

P, R, F1 = score(predictions, references, lang='en', verbose=True)

print(f"chrF score: {chrf.score}")
print(f"BERTScore F1: {F1.mean().item()}")

# %% [markdown]
# # BERT-base

# %%
from transformers import BertTokenizer, EncoderDecoderModel
import torch
from torch.utils.data import DataLoader, Dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


def prepare_data(df):
    input_encodings = tokenizer(df['Document'].tolist(), padding="max_length", truncation=True, max_length=128)
    target_encodings = tokenizer(df['Title'].tolist(), padding="max_length", truncation=True, max_length=32)

    return input_encodings, target_encodings


class HindiNewsDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __len__(self):
        return len(self.input_encodings.input_ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        item['labels'] = torch.tensor(self.target_encodings.input_ids[idx])
        return item

# Prepare datasets
train_input_encodings, train_target_encodings = prepare_data(train_df)
val_input_encodings, val_target_encodings = prepare_data(val_df)
test_input_encodings, test_target_encodings = prepare_data(test_df)

train_dataset = HindiNewsDataset(train_input_encodings, train_target_encodings)
val_dataset = HindiNewsDataset(val_input_encodings, val_target_encodings)
test_dataset = HindiNewsDataset(test_input_encodings, test_target_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# %%
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 'bert-base-multilingual-cased')
model.to(device)

# %%

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.encoder.config.pad_token_id = tokenizer.pad_token_id
model.decoder.config.pad_token_id = tokenizer.pad_token_id

model.config.decoder_start_token_id = tokenizer.cls_token_id 


# %%
from transformers import AdamW
from tqdm import tqdm


optimizer = AdamW(model.parameters(), lr=5e-5)


model.train()
num_epochs = 3  

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    print(f"Training loss: {loss.item()}")

# %%
from torch.utils.data import DataLoader
import torch

model.eval()  

total_loss = 0
total_batches = 0

with torch.no_grad(): 
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        total_loss += loss.item()
        total_batches += 1

average_loss = total_loss / total_batches
print(f"Validation Loss: {average_loss:.4f}")


# %%
import pandas as pd

model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,  
            num_beams=5,    
            repetition_penalty=2.5,  
            length_penalty=1.0,  
            early_stopping=True,  
            decoder_start_token_id=tokenizer.bos_token_id  
        )

       
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(generated_texts)


if len(predictions) == len(test_df):
    test_df['Generated_Headline'] = predictions
else:
    print("The number of predictions does not match the number of entries in the test dataset.")


print(test_df[['Document','Title', 'Generated_Headline']])


test_df.to_csv("test_with_headlines.csv", index=False)

# %%

if tokenizer.bos_token_id is None:
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    model.resize_token_embeddings(len(tokenizer)) 

model.config.decoder_start_token_id = tokenizer.bos_token_id


model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,  
            num_beams=5,    
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            decoder_start_token_id=tokenizer.bos_token_id  
        )

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(generated_texts)


for i, text in enumerate(predictions):
    print(f"Generated headline {i+1}: {text}")


# %%



