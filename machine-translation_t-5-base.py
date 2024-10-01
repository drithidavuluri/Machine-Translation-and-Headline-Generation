# %%
import pandas as pd
import json
import re
from tqdm.auto import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer
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

# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer

# %%
model_name = 't5-base'  # You can use 't5-base', 't5-large', 't5-3b', or 't5-11b' depending on your requirements
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# %%
def tokenize_function(examples):
    inputs = tokenizer(
        examples['input_text'],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    outputs = tokenizer(
        examples['target_text'],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    labels = outputs['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    
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
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    warmup_steps=500,
    weight_decay=0.01,
    logging_first_step=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",  # Changed to match evaluation_strategy
)


# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# %%
trainer.train()

# %%
model.save_pretrained("./t5-model")
tokenizer.save_pretrained("./t5-model")

# %%
def translate_text(text, model, tokenizer, device):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# %%
import torch
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_translations = []

for text in tqdm(test_df['input_text'].tolist()):
    translation = translate_text(text, model, tokenizer, device)
    test_translations.append(translation)


# %%
test_df['translations_new'] = test_translations

# %% [markdown]
# ### Bleu Score

# %%
import nltk
from nltk.translate.bleu_score import corpus_bleu

# %%
nltk.download('wordnet')
nltk.download('punkt')

# %%
references = [[ref.split()] for ref in test_df['target_text'].tolist()]
translated_texts = [translation.split() for translation in test_df['translations_new'].tolist()]

# %%
bleu1 = corpus_bleu(references, translated_texts, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(references, translated_texts, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu(references, translated_texts, weights=(0.33, 0.33, 0.33, 0))
bleu4 = corpus_bleu(references, translated_texts, weights=(0.25, 0.25, 0.25, 0.25))

# %%
print(f"BLEU-1: {bleu1}")
print(f"BLEU-2: {bleu2}")
print(f"BLEU-3: {bleu3}")
print(f"BLEU-4: {bleu4}")

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
    translation = test_df['translations_new'].iloc[i]

    scores = scorer.score(reference, translation)

    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

print(f"Average ROUGE-1: {average_rouge1}")
print(f"Average ROUGE-2: {average_rouge2}")
print(f"Average ROUGE-L: {average_rougeL}")


# %% [markdown]
# ### ChrF Score

# %%
import sacrebleu

references = [[ref] for ref in test_df['target_text'].tolist()]  # List of lists
translations = test_df['translations_new'].tolist()

chrf_scores = []

for reference, translation in zip(references, translations):
    chrf = sacrebleu.sentence_chrf(translation, reference)  # Note the order: (hypothesis, reference)
    chrf_scores.append(chrf.score)

average_chrf_score = sum(chrf_scores) / len(chrf_scores)

print(f"Average ChrF Score: {average_chrf_score}")


# %%



