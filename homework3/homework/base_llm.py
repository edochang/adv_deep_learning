from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Copilot was used as a teaching assistance and guide
class BaseLLM:
    def __init__(self, checkpoint=checkpoint, use_bfloat16: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if (use_bfloat16 and device == "cuda" and torch.cuda.is_bf16_supported()):
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device
        print(f"BaseLLM __initi__: Loaded model on device: {self.device} using model ({checkpoint}) with {'bfloat16' if use_bfloat16 else 'float32'}")

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        #print(f"BaseLLM parse_answer: Parsing answer:\n{answer}")  # debug
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            #print(f"BaseLLM parse_answer: BAD answer:\n{answer}")  # debug
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]    

    # type checker stubs for overloads. This is a function dectorator, so no implementation needed.
    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    # type checker stubs for overloads. This is a function dectorator, so no implementation needed.
    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
                                         temperature for more or less randomness in generation.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        # hyperparameter for model generation
        """ Temperature notes:
        - A temperature of 0 makes the model deterministic, always picking the most likely next token.
        - Lower temperatures (e.g., 0.2) make the model more focused and deterministic, often resulting in more repetitive or conservative outputs.
        - Higher temperatures (e.g., 0.7, 1.0) introduce randomness, allowing the model to sample from a wider range of possible next tokens.

        0 with do_sample=False: Deterministic output, always the same for the same input.
        0.1 - 0.3: Low temperature, more focused and deterministic outputs.
        0.4 - 0.6: Moderate temperature, balances randomness and focus.
        0.7 - 1.0: High temperature, more diverse and creative outputs. May reduce factual accuracy.
        """
        #temperature = 0.2 # override temperature for more diverse outputs
        repetition_penalty = 1.1 # override repetition penalty to reduce repetitive outputs
        # Set a reasonable limit for the number of tokens to generate. Prevents infinite generation.
        max_new_tokens = 150

        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]
        
        # tokenize the prompts with padding
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

        # n_return_sequences handling per overloads
        #print(f"num_return_sequences: {num_return_sequences}, datatype: {type(num_return_sequences)}")  # debug
        n_return_sequences = num_return_sequences if num_return_sequences is not None else 1

        # generate outputs
        # Enable sampling for more diverse / creative outputs
        do_sample = True if temperature > 0 else False  # Set do_sample based on temperature
        # note: use ** to unpack the inputs dictionary to pass as keyword arguments to the generate function's parameters. input_ids is passed to inputs and attention_mask is passed to attention_mask.
        #with torch.no_grad():
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            temperature=temperature if do_sample else 1.0, # default temperature is 1.0 when do_sample is False
            num_return_sequences=n_return_sequences, 
            repetition_penalty=repetition_penalty if do_sample else None, # apply repetition penalty only when sampling
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Slice outputs to remove input tokens per Pro Tip
        input_length = len(inputs["input_ids"][0]) # Get the length of the input
        # : keeps all rows, input_length: keeps columns from input_length to end
        generated_outputs = outputs[:, input_length:] # Slice outputs to remove input tokens
        
        # use batch_decode for flexibility when num_return_sequences is 1 or more than 1
        decoded_outputs = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

        # handle return type based on num_return_sequences
        if n_return_sequences == 1:
            # return list of strings: list[str]
            return decoded_outputs
        else:
            # return list of list of strings: list[list[str]]
            grouped_outputs = [
                decoded_outputs[i : i + num_return_sequences] for i in range(0, len(decoded_outputs), num_return_sequences)
            ]
            return grouped_outputs

        """
        # use list comprehension to decode each output in outputs
        return [self.tokenizer.decode(generated_output, skip_special_tokens=True) for generated_output in generated_outputs]
        """

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        
        """
        # exports prompts and generations to a debug file and overwrites existing file
        with open("answer_debug_output.txt", "w") as f:
            for i in range(len(generations)):
                f.write(f"Prompt: {prompts[i]}\nGeneration: {generations[i]}\n\n")
        """

        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
