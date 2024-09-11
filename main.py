import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer
# https://llama3-1.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiY3RteXJqbnY5aXpvbmNyNXZ5MWRxbmhyIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTEubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyNjE2MzUyOX19fV19&Signature=M%7ES5KNVd0CEFxEGSfxtsVQI%7ENfo72uc3T2sUj1wNXzc-ERsAOuTkhw6TFNEehSEuMY86-cJh-tVukKxZwhkV%7EpK6VxCD5DanDGoycWSx2t2xl8JA%7EsnMs1okrqqxTZgOfoCrl1IGPlv7-UdhltV%7EQ%7EigEIa7vqXmS-YRJCMJh1Oare-xMt0c-Wx92imfEERGnT3HyXfXTHeK79n61Xof3cOguLwBndLuH8enhrbjmluJSuTx2TiodTbJq2Igh3xE1MWV4UCXxLhDvp8JfHPZJE%7EFRXmxuajAhNjY1ajfpgoC7AmHhHYKwvVF2SSkmA3yYi2aNDJAPe3t5%7E6NXFWshw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=883343086461026

model_dir = "models/70B"
model = LlamaForCausalLM.from_pretrained(model_dir)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)
pipeline = transformers.pipeline(
"text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
'I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400,
)

for seq in sequences:
    print(f"{seq['generated_text']}")
