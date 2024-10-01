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
from transformers import BartForConditionalGeneration, BartTokenizer

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
import re

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df_downsampled['Document'] = train_df_downsampled['Document'].apply(preprocess_text)
train_df_downsampled['Title'] = train_df_downsampled['Title'].apply(preprocess_text)


# %%
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df_downsampled[['Document', 'Title']])
val_dataset = Dataset.from_pandas(val_df_downsampled[['Document', 'Title']])
test_dataset = Dataset.from_pandas(test_df_downsampled[['Document', 'Title']])

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# %%
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id

# %%
def generate_headlines(text, max_length=50):
   
    prompt_text = text + " Headline:"
    
    encoded_input = tokenizer.encode(prompt_text, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            encoded_input, 
            max_length=len(encoded_input[0]) + max_length, 
            num_return_sequences=1,
            no_repeat_ngram_size=2, 
            early_stopping=True 
        )
        
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    headline = generated_text[len(prompt_text):].strip()
    
    return headline

# %%
sample_text = "जलवायु परिवर्तन समझौते के पेरिस मसौदे ने भारत ..."
generated_headline = generate_headlines(sample_text)
print("Generated Headline:", generated_headline)

# %%
test_df

# %%
test_df['Document'][16068]

# %%
def generate_headlines(text, max_length=50):
    
    prompt_text = text[:500] + " Headline:" 

    encoded_input = tokenizer.encode(prompt_text, return_tensors='pt', max_length=1024, truncation=True)
    
    attention_mask = torch.ones(encoded_input.shape, dtype=torch.long) 

    with torch.no_grad():
        output = model.generate(
            encoded_input, 
            attention_mask=attention_mask,
            max_length=encoded_input.size(1) + max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2, 
            early_stopping=True,    
            pad_token_id=tokenizer.pad_token_id
        )
        
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if len(generated_text) <= len(prompt_text):
        return "No headline generated. Text may be too complex or too long."
    else:
        headline = generated_text[len(prompt_text):].strip()
        return headline


# %%
sample_text = "जलवायु परिवर्तन समझौते के पेरिस मसौदे ने भारत सरकार व उसके वार्ताकारों को खासा निराश कर दिया है। मसौदे में विकासशील देशों के तमाम सुझावों को छोड़ दिया गया है। इसमें भारत के हितों और उसकी चिंता को कम करके आंका गया है। जलवायु परिवर्तन पर विकसित एवं विकासशील देशों की अलग अलग जिम्मेदारी के मसले को छोड़ दिया गया है, जो यूएन फ्रेमवर्क कनवेंशन आन क्लाइमेट चेंज (यूएनएफसीसी) पर आधारित है। मसौदे में उत्सर्जन कम करने पर ज्यादा जोर दिया गया है और वित्तपोषण, बराबरी, तकनीकी साझेदारी, क्षमता निर्माण और हानि एवं क्षति पर बहुत कम ध्यान दिया गया है।   विकसित देशों ने अपनी मौजूदा वित्तीय बाध्यताओं को बगैर किसी खाके के छोड़ दिया है कि कैसे मौजूदा बाध्यताएं लागू की जाएंगी या भविष्य के लिए इन्हें किस तरह से लागू किया जाएगा। इसके बदले भारत जैसे विकासशील देशों से कहा गया है कि वे वैश्विक जलवायु वित्त पूल में अपनी हिस्सेदारी दें। इस मसौदे में विकसित देशों की मौजूदा बाध्यताओं को नरम कर दिया गया है और उसे स्वच्छ तकनीक की ओर मोड़ दिया गया है। यह सब भारतीय दल के लिए चिंता का विषय माना जा रहा है। इस मसौदे पर बिजनेस स्टैंडर्ड ने कुछ भारतीय वार्ताकारों से बात की, लेकिन किसी ने भी सामने आकर कुछ कहने से मना किया। एक वरिष्ठ भारतीय वार्ताकार ने कहा, 'मसौदा पूरी तरह अस्वीकार्य है। यह पिछले दरवाजे से यूएनएफसीसीसी को फिर से लिखने की कवायद है, जिसमें मौजूदा सिद्धांतों और प्रावधानों को कोई जगह नहीं दी गई है। इसमें हमारी या अन्य विकासशील देशों की चिंता को शामिल नहींं किया गया है। इसमेंं कुछ विकसित देशों का पक्ष लिया गया है और हमारे मसलों को छोड़ दिया गया है।' केंद्रीय पर्यावरण एवं वन मंत्री प्रकाश जावडेकर ने पिछले सप्ताह टाइम्स आफ इंडिया को दिए गए साक्षात्कार में इसे लेकर निराशा जताई थी। वहीं केंद्रीय पर्यावरण सचिव अशोक लवासा ने हाल की एक बैठक में मसौदे को लेकर कहा था कि हमारी कुछ 'आपत्तियां' हैं। जलवायु परिवर्तन पर काम करने वाले समूह थर्ड वल्र्ड नेटवर्क की मीना रामन ने कहा, 'इस मसौदे पर सहमति का मतलब यूएनएफसीसीसी समझौते का खत्म होना है। इससे ऐतिहासिक जिम्मेदारियों व हिस्सेदारी का खात्मा हो जाएगा, जैसा कि हम जानते हैं। इससे विकसित देश अपनी जिम्मेदारियों से आराम से निकल भागेंगे।' संबंधित खबरें"
generated_headline = generate_headlines(sample_text)
print("Generated Headline:", generated_headline)

# %%
import pandas as pd

generated_headlines = []

for index, row in test_df.head(10).iterrows():
    headline = generate_headlines(row['Document'])
    generated_headlines.append(headline)

test_df.loc[test_df.head(10).index, 'generated_headline'] = generated_headlines

print(test_df[['Document', 'generated_headline']].head(10))

# %%
test_df[['Document', 'Title','generated_headline']].head(10)

# %%



