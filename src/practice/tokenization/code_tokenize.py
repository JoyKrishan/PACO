from pygments import lex
from pygments.lexers import PythonLexer
import torch

# Example Python code
code = """
def greet(name):
    print("Hello, " + name + "!")
"""

# Tokenize the code
tokens = list(lex(code, PythonLexer()))

# # Print tokens
# for token in tokens:
#     print(token)

from transformers import AutoTokenizer, AutoModel

# Load the tokenizer and model
model_name = "microsoft/codebert-base"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example source code
source_code = "def hello_world():\n    print('Hello, world!')         print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    print('Hello, world!')    "

# Tokenize the source code
inputs = tokenizer(source_code, return_tensors="pt", max_length=512, truncation=True, padding=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract the `[CLS]` token embedding
cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()


print(cls_embedding.shape)
# Use cls_embedding as the representation for the entire source code snippet
