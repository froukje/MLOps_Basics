from datasets import load_dataset  # read from hugging face dataset
from transformers import AutoTokenizer

cola_dataset = load_dataset("glue", "cola")
print(cola_dataset)

print("sample datapoint")
train_dataset = cola_dataset["train"]
print(train_dataset[0])

# The tokenizer returns a dictionary with three items:
# * input_ids: the numbers representing the tokens in the text.
# * token_type_ids: indicates which sequence a token belongs to if there is more than one sequence.
# * attention_mask: indicates whether a token should be masked or not.
# These values are actually the model inputs.
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
print(tokenizer(train_dataset[:5]["sentence"]))
