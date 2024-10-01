# %%
import pandas as pd
import json
import re
from tqdm.auto import tqdm
from datasets import Dataset

# %%
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

en_lines = load_data('/kaggle/input/transl/v2/en-hi/train.en')
hi_lines = load_data('/kaggle/input/transl/v2/en-hi/train.hi')

# %%
train_data = {'input_text': en_lines, 'target_text': hi_lines}
df = pd.DataFrame(train_data)

# %%
df = df.sample(n=4000, random_state=42)
df

# %%
def preprocess_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['input_text'] = df['input_text'].apply(preprocess_text)
df['target_text'] = df['target_text'].apply(preprocess_text)

# %%
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# %%
train_df

# %%
train_dataset = Dataset.from_pandas(train_df[['input_text', 'target_text']])
val_dataset = Dataset.from_pandas(val_df[['input_text', 'target_text']])
test_dataset = Dataset.from_pandas(test_df[['input_text', 'target_text']])

# %% [markdown]
# # BART-base

# %%
from transformers import BartForConditionalGeneration, BartTokenizer

# %%
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# %%
def tokenize_function(examples):
    inputs = tokenizer(examples['input_text'], max_length=1024, truncation=True, padding="max_length")
    outputs = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    
    labels = outputs.input_ids
    labels = [[label if label != tokenizer.pad_token_id else -100 for label in labels_example] for labels_example in labels]
    
    inputs["labels"] = labels
    return inputs

# %%
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# %%
from transformers import Trainer, TrainingArguments

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

# %%
trainer.train()

# %%
model.save_pretrained("./bartbase-model")
tokenizer.save_pretrained("./bartbase-model")

# %%
def translate_text(text, model, tokenizer, device):
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


# %%
test_df

# %%
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

first_entry = test_df.iloc[0]['input_text']
print("Original text:", first_entry)

translated_text = translate_text(first_entry, model, tokenizer, device)
print("Translated text:", translated_text)

# %%
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_df['translated_text'] = ''

for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    translated_text = translate_text(row['input_text'], model, tokenizer, device)
    test_df.at[index, 'translated_text'] = translated_text

print(test_df[['input_text', 'target_text', 'translated_text']])

# %%
test_df

# %%
test_df['translations_new'] = test_translations

# %% [markdown]
# ### Bleu Score

# %%
import nltk
from nltk.translate.bleu_score import corpus_bleu

nltk.download('wordnet')
nltk.download('punkt')

references = [[ref.split()] for ref in test_df['target_text'].tolist()]
translated_texts = [translation.split() for translation in test_df['translated_text'].tolist()]

bleu1 = corpus_bleu(references, translated_texts, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(references, translated_texts, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu(references, translated_texts, weights=(0.33, 0.33, 0.33, 0))
bleu4 = corpus_bleu(references, translated_texts, weights=(0.25, 0.25, 0.25, 0.25))

print(f"BLEU-1: {bleu1*100}")
print(f"BLEU-2: {bleu2*100}")
print(f"BLEU-3: {bleu3*100}")
print(f"BLEU-4: {bleu4*100}")

# %% [markdown]
# ### Rouge-score

# %%
!pip install rouge-score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for i in range(len(test_df)):
    reference = test_df['target_text'].iloc[i]
    translation = test_df['translated_text'].iloc[i]

    scores = scorer.score(reference, translation)

    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

print(f"Average ROUGE-1: {average_rouge1*100}")
print(f"Average ROUGE-2: {average_rouge2*100}")
print(f"Average ROUGE-L: {average_rougeL*100}")

# %% [markdown]
# ### ChrF Score

# %%
!pip install sacrebleu
import sacrebleu

references = [[ref] for ref in test_df['target_text'].tolist()]  # List of lists
translations = test_df['translated_text'].tolist()

chrf_scores = []

for reference, translation in zip(references, translations):
    chrf = sacrebleu.sentence_chrf(translation, reference)  # Note the order: (hypothesis, reference)
    chrf_scores.append(chrf.score)

average_chrf_score = sum(chrf_scores) / len(chrf_scores)

print(f"Average ChrF Score: {average_chrf_score}")


# %% [markdown]
# # M2M-100 is a multilingual model

# %%
!pip install transformers


# %%
pip install transformers sentencepiece


# %%
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)


# %%
def translate_text(text, model, tokenizer):

    tokenizer.src_lang = "en"
    model.config.forced_bos_token_id = tokenizer.get_lang_id("hi")

    encoded = tokenizer(text, return_tensors="pt")

    generated_tokens = model.generate(**encoded, forced_bos_token_id=model.config.forced_bos_token_id)

    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text

sample_text = "Hello world!"
translated_text = translate_text(sample_text, model, tokenizer)
print("Translated text:", translated_text)

# %%
top10_translations = []
for index, row in tqdm(test_df.head(10).iterrows(), total=10):
    translation = translate_text(row['input_text'], model, tokenizer)
    top10_translations.append(translation)

test_df.head(10)

# %%



