from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse

def main(args):
    # Replace with your Phi model's local path
    model_path = args.model_path

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # Move to GPU if available
    model.to("cuda")

    
    with open(args.conversation_file, "r") as conversations:
        data = json.load(conversations)

    categorizations = [] 

    for conversation in data:
        conversationId = conversation["image"]

        # Input and generation
        prompt = """
Below is a description of a graph's waveform and a multiple-choice question about it. Do not generate additional questions, answers, or options. Only respond directly to the final question. Do not add or change the provided options.

**Answer only the final question provided**
        """
        for interaction in conversation["conversation"]:
            question = interaction["question"]
            answer = interaction["answer"]

            prompt += f"Question: {question}\nAnswer: {answer}\n"


        prompt += """\nThere are three options:

A) Random noise: This wave will have random points in the entire line. It may or may not appear continuous.
B) Sine wave: A smooth continuous wave with gradual transitions from one level to another.
C) Square wave: A wave that is not continuous, not random, and has sharp corners where it jumps from one value to another.

Final Question: Based on this information, which type of graph do I have? Only select from the three options (A, B, C) provided above. Do not invent new answers or modify the options. Explain your reasoning.
        """
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

        outputs = model.generate(inputs, max_new_tokens = 200)

        # this is surprisingly weird to do?
        # https://github.com/huggingface/transformers/issues/17117#issuecomment-1120809167
        modelResponse = tokenizer.batch_decode(outputs[:, inputs.shape[1]:])[0]
    
        print(modelResponse)
        categorizations.append({"conversationId": conversationId, "response": modelResponse})
    
    print(categorizations)
    with open(args.output_file, "w") as outputFile:
        json.dump(categorizations, outputFile, indent=2)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model-path", type=str, required=True)
    parse.add_argument("--conversation-file", type=str, required=True)
    parse.add_argument("--output-file", type=str, required=True)
    args = parse.parse_args()
    main(args)
