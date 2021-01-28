#%%
import pandas as pd

#%%
data = pd.read_csv('wine.csv')
data

#%%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#%%
inputs = tokenizer("Champagne Ruinart Blanc de Blancs Brut", return_tensors="np")
print(inputs)

#%%
