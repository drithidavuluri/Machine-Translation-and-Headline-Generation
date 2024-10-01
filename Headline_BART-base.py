# %%

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'hindi-headline:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F4843372%2F8180842%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240421%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240421T104642Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D962bea6a5230df94fbfd2a4e386fe6a69d29a762f6243007100fc8175bb562714be80127da9b93fac0cd664519ffa4ded9d5999433976276545d45828df6403ea780adfdad2b2b1e75c09110b37f35abf7bddcceeca16a8ca245c850310e7972ac5b4f8f3d4ffc89fad162b4d3130b388e8cc6c886b2fc5bc0db29cc9eafd70398b46af0d28c2dbd98bec9e454f6f9351af5e41482d2cd3492ddbdcc0c641ba64474066065663dc9f38663675ce779908b20285fdb2432d3c24f63ccd5883e8a70ec364b26b73d63d969db1fe6c2d18800a0b033d3dca9d22b964eb7325ee7406fe36762d80f6035360825eb69a2a0d341d8fa1f4743c4b49d06f05ac7c5b02d'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')


# %%
import pandas as pd
import json

# %% [markdown]
# ### Conversting into dataframe

# %%
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

# %% [markdown]
# ### Reducing the size of dataset

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

# %% [markdown]
# # BART-base

# %%
import re

from transformers import BartForConditionalGeneration, BartTokenizer

from datasets import Dataset

from transformers import BartTokenizer

from transformers import Trainer, TrainingArguments

# %%
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

article_text = "कनाडा अमेरिका और यूरोपीय संघ का अनुसरण करते हुए ईरान पर लगा प्रतिबंध हटाएगा। यह बात देश के विदेश मंत्री स्टेफाने डियोन ने कही। संसद में एक सवाल के जवाब में डियोन ने कहा, 'कनाडा प्रतिबंध हटाएगा।' उन्होंने कहा, 'हम इस नीति में बदलाव लाएंगे। प्रतिबंध किसी के लिए अच्छा नहीं है।' इसके लिए कोई समयसीमा नहीं दी गई है। कनाडा ने ईरान और विश्व के प्रमुख देशों के बीच हुई सहमति लागू होने के कुछ हफ्तों के बाद कल यह घोषणा की। इस सहमति से अमेरिका और यूरोपीय संघ द्वारा ईरान के परमाणु कार्यक्रम के संबंध में लगाए गए प्रतिबंध को हटाने का रास्ता साफ हुआ।"
input_ids = tokenizer.encode(article_text, return_tensors='pt')

summary_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
headline = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"Generated Headline: {headline}")

# %%
train_df_downsampled = train_df.sample(n=8000, random_state=42)
val_df_downsampled = val_df.sample(n=1719, random_state=42)
test_df_downsampled = test_df.sample(n=1709, random_state=42)

# %%
def preprocess_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df_downsampled['Document'] = train_df_downsampled['Document'].apply(preprocess_text)
train_df_downsampled['Title'] = train_df_downsampled['Title'].apply(preprocess_text)


# %%
!pip install datasets

# %%
train_dataset = Dataset.from_pandas(train_df_downsampled[['Document', 'Title']])
val_dataset = Dataset.from_pandas(val_df_downsampled[['Document', 'Title']])
test_dataset = Dataset.from_pandas(test_df_downsampled[['Document', 'Title']])

# %%
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def tokenize_function(examples):
    
    model_inputs = tokenizer(examples['Document'], max_length=1024, truncation=True, padding="max_length")
   
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['Title'], max_length=128, truncation=True, padding="max_length")

    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['Document', 'Title'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['Document', 'Title'])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['Document', 'Title'])

# %%
!pip install accelerate -U
!pip install transformers[torch] -U


# %%
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# %%
model.save_pretrained('./results/checkpoint-last')
tokenizer.save_pretrained('./results/checkpoint-last')

# %%
from tqdm.auto import tqdm

# def generate_headline(document_text, max_length=50):
    
#     input_ids = tokenizer(document_text, return_tensors='pt', max_length=1024, truncation=True).input_ids
#     input_ids = input_ids.to(model.device) 

#     summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)

#     headline = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return headline

# generated_headlines = []

# for _, row in tqdm(test_df_downsampled.iterrows(), total=5, desc="Generating Headlines"):
#     document_text = row['Document']
#     generated_headline = generate_headline(document_text)
#     generated_headlines.append(generated_headline)

# test_df_downsampled['Generated_Headline'] = generated_headlines

# test_df_downsampled[['Document', 'Title', 'Generated_Headline']]

model_path = "./results/checkpoint-last"

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

def generate_headline(text, model, tokenizer):
    
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=50,
        num_beams=4,
        early_stopping=True
    )

    headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline

test_df['Generated_Headline'] = test_df['Document'].apply(lambda x: generate_headline(x, model, tokenizer))

test_df[['Document', 'Title','Generated_Headline']]

# %% [markdown]
# ## Evaluation Metrics

# %%
from pycocoevalcap.cider.cider import Cider

def calculate_cider(pred_dict, true_dict):
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(true_dict, pred_dict)
    return score

true_dict = {i: [line] for i, line in enumerate(true_headlines)}
pred_dict = {i: [line] for i, line in enumerate(pred_headlines)}

cider_score = calculate_cider(pred_dict, true_dict)
print("CIDEr Score:", cider_score*100)

# %%
pip install pynlpl

# %%
from sacrebleu.metrics import CHRF

def calculate_chrf(predictions, references):
    chrf = CHRF(word_order=2)  
    scores = chrf.corpus_score(predictions, [references])
    return scores.score

chrf_score = calculate_chrf(pred_headlines, true_headlines)
print("chrF Score:", chrf_score)

# %%
pip install bert-score

# %%
from bert_score import score

def calculate_bertscore(predictions, references):
    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    return F1.mean().item()

bertscore = calculate_bertscore(pred_headlines, true_headlines)
print("BERTScore F1:", bertscore)

# %%


# %%


# %%


# %%



