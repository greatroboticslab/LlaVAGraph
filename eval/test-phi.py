from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Replace with your Phi model's local path
model_path = "/projects/imo2d/phi-3.5-mini-instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

with open("testOutput.txt", "r") as file:
    conversation = file.read()

# Input and generation
input_text = f"""
Below is a description of a graph's waveform and a multiple-choice question about it. Do not generate additional questions, answers, or options. Only respond directly to the final question. Do not add or change the provided options.

**Answer only the final question provided**

{conversation}

There are three options:

    A) Random noise: This wave will have random points in the entire line. It may or may not appear continuous.
    B) Sine wave: A smooth continuous wave with gradual transitions from one level to another.
    C) Square wave: A wave that is not continuous, not random, and has sharp corners where it jumps from one value to another.

Final Question: Based on this information, which type of graph do I have? Only select from the three options (A, B, C) provided above. Do not invent new answers or modify the options. Explain your reasoning.

"""

inputs = tokenizer(input_text, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=150)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)

