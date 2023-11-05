# Use a pipeline as a high-level helper
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-AWQ")
model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-AWQ")


# pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1", device="mps")

# while True:
#     cmd = input("> ")
#     out = pipe(cmd)
#     print(out)