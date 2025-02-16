#!/usr/bin/env python
# coding: utf-8

# # Implementation

# ## Step 1: Dataset Prepration

# In[ ]:


import pandas as pd


# ### Load the dataset

# In[ ]:


df = pd.read_csv("MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx")


# ### Data Preprocessing

# In[10]:


import re


# In[11]:


def clean_text(text):
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    text = text.lower()  # Convert to lowercase
    
    return text


# In[146]:


# Apply the function to both columns
df['CHQ'] = df['CHQ'].apply(clean_text)
df['Summary'] = df['Summary'].apply(clean_text)


# In[ ]:


# Remove extremely short or extremely long sentences
# i.e sentences with less than 3 words and more than 80 words
df = df[df['CHQ'].apply(lambda x: len(x.split()) >= 3)]
df = df[df['Summary'].apply(lambda x: len(x.split()) >= 3)]

df = df[df['CHQ'].apply(lambda x: len(x.split()) <= 80)]
df = df[df['Summary'].apply(lambda x: len(x.split()) <= 80)]


# In[148]:


df.info()


# 26 rows are removed after cleaning.

# In[149]:


from nltk.corpus import stopwords


# In[150]:


# Remove stopwords which won't be informative for the model
stop_words = set(stopwords.words("english"))
df['CHQ'] = df['CHQ'].apply(lambda text: ' '.join([word for word in text.split() if word not in stop_words]))
df['Summary'] = df['Summary'].apply(lambda text: ' '.join([word for word in text.split() if word not in stop_words]))


# In[151]:


df.head()


# ## Step 2: Round-Trip Translation

# ### Translation using a machine translation model

# In[17]:


# Using pre-trained MarianMT model for translation
from transformers import MarianMTModel, MarianTokenizer


# In[163]:


# Load pretrained model and tokenizer for the translation
def load_translation_model(source: str, dest: str)-> tuple:
    
    model_name = f"Helsinki-NLP/opus-mt-{source}-{dest}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    model = MarianMTModel.from_pretrained(model_name)
   
    
    return model, tokenizer


# In[164]:


# Function to Translate to Pivot Languages and Back
def translate(text: list, model: MarianMTModel, tokenizer: MarianTokenizer)-> str:
    
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**tokens)
    
    return tokenizer.decode(translated[0], skip_special_tokens=True)


# In[11]:


# Setting up the computer to use the GPU
import torch

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Running on: {device}")


# In[ ]:


rtt_questions = []
languages = ['es', 'de', 'it', 'zh', 'fr']

counter = 1
for question in df['CHQ']:
   
    print(f"Processing question {counter}...")
    counter += 1
    paraphrases = []
    for lang in languages:
        
        # Load models for each pivot language dynamically
        forward_model, forward_tokenizer = load_translation_model("en", lang)
        backward_model, backward_tokenizer = load_translation_model(lang, "en")

        # Forward and backward translation
        translated_text = translate(question, forward_model, forward_tokenizer)
        round_trip_text = translate(translated_text, backward_model, backward_tokenizer)

        paraphrases.append(round_trip_text)
    rtt_questions.append(paraphrases)

df['CHQ_paraphrases'] = rtt_questions


# In[166]:


print("Original Question: ",df.CHQ[0])
print("After translating to five given languages: ",df.CHQ_paraphrases[0])


# ### Translate using Google Translate API

# In[16]:


from deep_translator import GoogleTranslator


# In[ ]:


# Function that uses Google Translate API to generate paraphrased questions
def google(question, pivot_language):
    
    translated = GoogleTranslator(source='en', target=pivot_language).translate(question)
    back_translated = GoogleTranslator(source=pivot_language, target='en').translate(translated)
    return back_translated

google_languages = ['es', 'de', 'it', 'zh-CN', 'fr']

df['CHQ_google_paraphrases'] = df['CHQ'].apply(lambda x: [google(x, lang) for lang in google_languages])


# In[169]:


df.head()


# In[170]:


print("Translation by pretrained model: ",df.CHQ_paraphrases[0])
print("Translation by Google Translate: ",df.CHQ_google_paraphrases[0])


# ## Question Selection

# ### Using FQD to select a subset of the new dataset

# In[26]:


from transformers import BertTokenizer, BertModel


# In[40]:


# Using Bert pretrained model for embedding both original and paraphrased questions.
# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# In[ ]:


import numpy as np
from torch.nn.functional import cosine_similarity


# In[53]:


import ast


# In[56]:


# Function to embed text using the [CLS] token
def embed(text: str) -> torch.Tensor:
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].squeeze()


# In[57]:


# FQD calculation using cosine similarity
def fqd(original_embedding: torch.Tensor, rtt_embedding: torch.Tensor) -> float:
   
    # Normalize the embeddings
    original_embedding = original_embedding / original_embedding.norm(p=2)
    rtt_embedding = rtt_embedding / rtt_embedding.norm(p=2)
    
    similarity = cosine_similarity(original_embedding.unsqueeze(0), rtt_embedding.unsqueeze(0))
    
    return 1 - similarity.item()


# In[ ]:


# Normalization function
def normalize_fqd(fqd_scores: list) -> list:
    
    if len(fqd_scores) == 0:
        return []
    
    fqd_min, fqd_max = np.min(fqd_scores), np.max(fqd_scores)
    
    if fqd_max - fqd_min == 0:
        return [0.0 for _ in fqd_scores]
    
    return [(fqd - fqd_min) / (fqd_max - fqd_min) for fqd in fqd_scores]


# ### Question Selection on Paraphrased questions generated by pretrained model.

# In[61]:


# Process the DataFrame rows
fqd_scores = []

for index, row in df.iterrows():
    
    original_embedding = embed(row['CHQ'])
    
    # Get the paraphrases (convert from string if necessary)
    paraphrases = row['CHQ_paraphrases']
    if isinstance(paraphrases, str):
            paraphrases = ast.literal_eval(paraphrases)

    
    if not isinstance(paraphrases, list) or not paraphrases:
        fqd_scores.append([])
        continue
    
    # Compute embeddings and FQD scores for each paraphrase
    paraphrased_embeddings = [embed(paraphrase) for paraphrase in paraphrases]
    row_scores = [fqd(original_embedding, rtt_embedding) for rtt_embedding in paraphrased_embeddings]
    fqd_scores.append(row_scores)

# Normalize FQD scores per row
normalized_fqd_scores = [normalize_fqd(scores) if scores else [] for scores in fqd_scores]

# Add the results to the DataFrame
df['FQD_scores_MarianMT'] = pd.Series(normalized_fqd_scores)


# In[ ]:


df.head()


# ### Question Selection on Paraphrased questions generated by Google Translate model.

# In[63]:


# Process the DataFrame rows
fqd_scores = []

for index, row in df.iterrows():
    
    original_embedding = embed(row['CHQ'])
    
    # Get the paraphrases (convert from string if necessary)
    paraphrases = row['CHQ_google_paraphrases']
    if isinstance(paraphrases, str):
            paraphrases = ast.literal_eval(paraphrases)

    
    if not isinstance(paraphrases, list) or not paraphrases:
        fqd_scores.append([])
        continue
    
    # Compute embeddings and FQD scores for each paraphrase
    paraphrased_embeddings = [embed(paraphrase) for paraphrase in paraphrases]
    row_scores = [fqd(original_embedding, rtt_embedding) for rtt_embedding in paraphrased_embeddings]
    fqd_scores.append(row_scores)

# Normalize FQD scores per row
normalized_fqd_scores = [normalize_fqd(scores) if scores else [] for scores in fqd_scores]

# Add the results to the DataFrame
df['FQD_scores_MarianMT_Google'] = pd.Series(normalized_fqd_scores)


# In[86]:


df.head()


# In[64]:


# Save the new dataframe containing paraphrases to a CSV file
df.to_csv("MeQSum_ACL2019_BenAbacha_Demner-Fushman.csv", index=False)


# ### Using PRQD to select a subset of the new dataset

# In[71]:


from sentence_transformers import SentenceTransformer
from torch.nn.functional import softmax


# In[66]:


model = SentenceTransformer('all-MiniLM-L6-v2')


# In[ ]:


# Embedding function for PRQD selection approach
def embed_2(text: str) -> torch.Tensor:
    inputs = model.encode([text], convert_to_tensor=True)
    
    return inputs[0]  # Return the embedding tensor


# In[80]:


#  Convert an embedding into a probability distribution over its dimensions using softmax function.
def embedding_to_distribution(embedding: torch.Tensor) -> np.ndarray:

    # Ensure embedding is 1D
    if embedding.dim() != 1:
        embedding = embedding.squeeze()
   
    distribution = softmax(embedding, dim=0)
    # Detach, move to CPU, and convert to NumPy for element-wise operations
    return distribution.cpu().detach().numpy()


# In[ ]:


# Function to calculate PRQD
def prqd_distribution(ref_embedding: torch.Tensor,
            cand_embedding: torch.Tensor, alpha_values: list) -> float:
   
    best_f1 = 0.0

    # Convert embeddings to distributions over their dimensions
    hQ = embedding_to_distribution(ref_embedding)
    hQ_hat = embedding_to_distribution(cand_embedding)

    # Iterate over the provided alpha values
    for alpha in alpha_values:

        precision = np.sum(np.minimum(alpha * hQ, hQ_hat))
        recall = np.sum(np.minimum(hQ, hQ_hat / alpha))
        
        # Compute F1 (harmonic mean of precision and recall) if possible
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # Keep track of the best F1 score over the alpha grid
        best_f1 = max(best_f1, f1)

    return best_f1


# ### Question Selection on Paraphrased questions generated by pretrained model.

# In[ ]:


# Define a grid of values for alpha
alpha_values = np.linspace(0.1, 10, 50)

# List to hold the PRQD scores for all rows
prqd_scores_all = []

for idx, row in df.iterrows():
    original_text = row['CHQ']
    paraphrases = row['CHQ_paraphrases']
    
    # If paraphrases are stored as a string, convert them to a list
    if isinstance(paraphrases, str):
        paraphrases = eval(paraphrases)
    
    # If no valid paraphrase list exists, append an empty list for this row
    if not isinstance(paraphrases, list) or len(paraphrases) == 0:
        prqd_scores_all.append([])
        continue
    
    # Compute the embedding for the original question
    original_embedding = embed_2(original_text)
    
    # List to store the PRQD score for each paraphrase for this row
    row_scores = []
    for paraphrase in paraphrases:
        # Compute the embedding for the candidate paraphrase
        candidate_embedding = embed_2(paraphrase)
        # Compute the PRQD score using our new distribution-based function
        score = prqd_distribution(original_embedding, candidate_embedding, alpha_values)
        row_scores.append(score)
    
    prqd_scores_all.append(row_scores)


df['PRQD_scores_MarianMT'] = pd.Series(prqd_scores_all)


# In[87]:


df.head()


# ### Question Selection on Paraphrased questions generated by Google Translate.

# In[88]:


# Define a grid of values for alpha
alpha_values = np.linspace(0.1, 10, 50)

# List to hold the PRQD scores for all rows
prqd_scores_all = []

for idx, row in df.iterrows():
    original_text = row['CHQ']
    paraphrases = row['CHQ_google_paraphrases']
    
    # If paraphrases are stored as a string, convert them to a list
    if isinstance(paraphrases, str):
        paraphrases = eval(paraphrases)
    
    # If no valid paraphrase list, append an empty list for this row
    if not isinstance(paraphrases, list) or len(paraphrases) == 0:
        prqd_scores_all.append([])
        continue
    
    # Compute the embedding for the original (gold) question
    original_embedding = embed_2(original_text)
    
    # List to store the PRQD score for each paraphrase for this row
    row_scores = []
    for paraphrase in paraphrases:
        # Compute the embedding for the candidate paraphrase
        candidate_embedding = embed_2(paraphrase)
        # Compute the PRQD score using our new distribution-based function
        score = prqd_distribution(original_embedding, candidate_embedding, alpha_values)
        row_scores.append(score)
    
    prqd_scores_all.append(row_scores)


df['PRQD_scores_Google'] = pd.Series(prqd_scores_all)


# In[89]:


df.head()


# In[102]:


# Convert the CHQ_paraphrases column from string to list since it is stored as a string.
def parse_paraphrases(entry):
    if isinstance(entry, str):
       
        return ast.literal_eval(entry)
       
    return entry


# In[154]:


df.head()


# ## Subset Selection

# In[156]:


# Define thresholds for FQD and PRQD scores
FQD_MIN_THRESHOLD = 0.05
FQD_MAX_THRESHOLD = 0.8
PRQD_MIN_THRESHOLD = 0.8
PRQD_MAX_THRESHOLD = 0.99


# ### Selecting subsets of parashrases generated by MarianMT.

# In[ ]:


# Store the parsed paraphrases in a new column
df["CHQ_paraphrases_parsed"] = df["CHQ_paraphrases"].apply(parse_paraphrases)


# In[ ]:


optimal_paraphrase_indices_PRQD = []
optimal_paraphrase_indices_FQD = []

# Iteration over the DataFrame rows
for idx, row in df.iterrows():
    
    # Get the list of scores for the current row
    prqd_list = row["PRQD_scores_MarianMT"] 
    fqd_list  = row["FQD_scores_MarianMT"]
    
    # Initialize index lists for this row
    prqd_indices = []
    fqd_indices = []
    
    # Iterate over each paraphrase candidate (assumed same order in both lists)
    for j, (prqd_score, fqd_score) in enumerate(zip(prqd_list, fqd_list)):
        if PRQD_MIN_THRESHOLD <= prqd_score <= PRQD_MAX_THRESHOLD:
            prqd_indices.append(j)
        if FQD_MIN_THRESHOLD <= fqd_score <= FQD_MAX_THRESHOLD:
            fqd_indices.append(j)
    
    optimal_paraphrase_indices_PRQD.append(prqd_indices)
    optimal_paraphrase_indices_FQD.append(fqd_indices)

# Now we create two new columns for the selected subsets
df["Optimal_Paraphrases_PRQD"] = [
    [row_paraphrases[idx] for idx in indices] 
    for row_paraphrases, indices in zip(df["CHQ_paraphrases_parsed"], optimal_paraphrase_indices_PRQD)
]

df["Optimal_Paraphrases_FQD"] = [
    [row_paraphrases[idx] for idx in indices] 
    for row_paraphrases, indices in zip(df["CHQ_paraphrases_parsed"], optimal_paraphrase_indices_FQD)
]


# In[174]:


df.info()


# ### Selecting subsets of parashrases generated by Google Translate.

# In[175]:


# Store the parsed paraphrases in a new column
df["CHQ_google_paraphrases_parsed"] = df["CHQ_google_paraphrases"].apply(parse_paraphrases)


# In[176]:


optimal_paraphrase_indices_PRQD = []
optimal_paraphrase_indices_FQD = []

# Iteration over the DataFrame rows
for idx, row in df.iterrows():
    
    # Get the list of scores for the current row
    prqd_list = row["PRQD_scores_Google"]  
    fqd_list  = row["FQD_scores_Google"] 
    
    # Initialize index lists for this row
    prqd_indices = []
    fqd_indices = []
    
    # Iterate over each paraphrase candidate (assumed same order in both lists)
    for j, (prqd_score, fqd_score) in enumerate(zip(prqd_list, fqd_list)):
        if PRQD_MIN_THRESHOLD <= prqd_score <= PRQD_MAX_THRESHOLD:
            prqd_indices.append(j)
        if FQD_MIN_THRESHOLD <= fqd_score <= FQD_MAX_THRESHOLD:
            fqd_indices.append(j)
    
    optimal_paraphrase_indices_PRQD.append(prqd_indices)
    optimal_paraphrase_indices_FQD.append(fqd_indices)

# Now create two new columns for the selected subsets
df["Optimal_PRQD_Google"] = [
    [row_paraphrases[idx] for idx in indices] 
    for row_paraphrases, indices in zip(df["CHQ_google_paraphrases_parsed"], optimal_paraphrase_indices_PRQD)
]

df["Optimal_FQD_Google"] = [
    [row_paraphrases[idx] for idx in indices] 
    for row_paraphrases, indices in zip(df["CHQ_google_paraphrases_parsed"], optimal_paraphrase_indices_FQD)
]


# In[177]:


df.head()


# ## Summeriztion
# 
# We will use the T5 Model and Tokenizer to generate summaries.

# In[178]:


from transformers import T5Tokenizer, T5ForConditionalGeneration


# In[179]:


# Load the model
model_ = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_)
model = T5ForConditionalGeneration.from_pretrained(model_)


# In[180]:


# Summary generator function using T5 pretrained model
def generate_single_summary(texts, is_singular: bool = False):
    
    max_length = 10
    min_length = 1
    
    summaries = []
    counter = 1
    for text in texts:
        
        if (is_singular):
            print(f"Generating summary for question {counter}")
            counter += 1
            
        if not isinstance(text, str) or not text.strip():
            summaries.append("")
            continue
       
        input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        output_ids = model.generate(input_ids, max_length=max_length, min_length=min_length, 
                                     length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary)
        
    return summaries


# #### Generating summaries from raw questions.

# In[181]:


df["Raw_CHQ_Summeries"] = generate_single_summary(df["CHQ"], True)


# #### Generating summeries for paraphrased questions generated by MarianMT

# In[182]:


df['summaries_FQD_MarianMT'] = df['Optimal_FQD_MarianMT'].apply(lambda paraphrases: generate_single_summary(paraphrases))


# In[183]:


df['summaries_PRQD_MarianMT'] = df['Optimal_PRQD_MarianMT'].apply(lambda paraphrases: generate_single_summary(paraphrases))


# #### Generating summeries for paraphrased questions generated by Google Translate

# In[184]:


df['summaries_FQD_Google'] = df['Optimal_FQD_Google'].apply(lambda paraphrases: generate_single_summary(paraphrases))


# In[185]:


df['summaries_PRQD_Google'] = df['Optimal_PRQD_Google'].apply(lambda paraphrases: generate_single_summary(paraphrases))


# In[186]:


df.head()


# ## Evaluation

# ### We will use Rouge and BLEU metrics to apply evaluation

# In[132]:


import evaluate


# In[133]:


# Load evaluation metrics
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")


# ### Before starting the evaluation, we need to do some row cleaning,
# 
# ### since some of the rows have null generated summaries.

# In[197]:


df.head(1)


# In[191]:


# Flatten lists in CHQ_paraphrase_summaries_MarianMT
df["summaries_FQD_MarianMT_str"] = df["summaries_FQD_MarianMT"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else x)


# In[193]:


# Flatten lists in CHQ_paraphrase_summaries_MarianMT
df["summaries_PRQD_MarianMT_str"] = df["summaries_PRQD_MarianMT"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else x)


# In[ ]:


df["summaries_FQD_Google_str"] = df["summaries_FQD_Google"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else x)


# In[ ]:


# Flatten lists in the columns below an store them in new columns.
df["summaries_PRQD_Google_str"] = df["summaries_PRQD_Google"].apply(
    lambda x: " ".join(x) if isinstance(x, list) else x)


# In[198]:


# Filter out empty or invalid rows
df = df[(df["Summary"].str.strip() != "") & (df["summaries_FQD_MarianMT_str"].str.strip() != "") &
         (df["summaries_PRQD_MarianMT_str"].str.strip() != "") & (df["summaries_FQD_Google_str"].str.strip() != "") &
         (df["summaries_PRQD_Google_str"].str.strip() != "")]


# In[199]:


df.head(10)


# In[200]:


def eval(reference_texts, generated_texts)-> tuple:
    
    # Compute ROUGE scores using the raw strings
    rouge_scores = rouge_metric.compute(
        predictions=list(generated_texts), 
        references=list(reference_texts)
    )
    
    # For BLEU:
    # predictions is a list of strings
    # references is a list of lists of strings (one reference per prediction)
    bleu_score = bleu_metric.compute(
        predictions=list(generated_texts), 
        references=[[ref] for ref in reference_texts]
    )
    
    return rouge_scores, bleu_score


# ### Evaluating paraphrase summaries selected by FQD measure and generated by MarianMT pretrained model.

# In[202]:


rouge_scores, bleu_score = eval(
    df["Summary"].tolist(), 
    df["summaries_FQD_MarianMT_str"].tolist()
)

print("Results for paraphrases generated by MarianMT model and selected by FQD measure:")
print("ROUGE Scores:", rouge_scores)
print("BLEU Score:", bleu_score)


# ### Evaluating paraphrase summaries retrieved by PRQD measure and generated by MarianMT pretrained model.

# In[203]:


rouge_scores, bleu_score = eval(
    df["Summary"].tolist(), 
    df["summaries_PRQD_MarianMT_str"].tolist()
)

print("Results for paraphrases generated by MarianMT model and selected by PRQD measure:")
print("ROUGE Scores:", rouge_scores)
print("BLEU Score:", bleu_score)


# ### Evaluating paraphrase summaries selected by FQD measure and generated by Google Translate.

# In[204]:


rouge_scores, bleu_score = eval(
    df["Summary"].tolist(), 
    df["summaries_FQD_Google_str"].tolist()
)

print("Results for paraphrases generated by Google Translate and selected by FQD measure:")
print("ROUGE Scores:", rouge_scores)
print("BLEU Score:", bleu_score)


# ### Evaluating paraphrase summaries selected by PRQD measure and generated by Google Translate.

# In[206]:


rouge_scores, bleu_score = eval(
    df["Summary"].tolist(), 
    df["summaries_PRQD_Google_str"].tolist()
)

print("Results for paraphrases generated by Google Translate and selected by PRQD measure:")
print("ROUGE Scores:", rouge_scores)
print("BLEU Score:", bleu_score)


# ### Evaluating summaries from raw questions.

# In[207]:


rouge_scores, bleu_score = eval(
    df["Summary"].tolist(), 
    df["Raw_CHQ_Summeries"].tolist()
)


# In[208]:


print("Result for summaries generated from original questions:")
print("ROUGE Scores:", rouge_scores)
print("BLEU Score:", bleu_score)

