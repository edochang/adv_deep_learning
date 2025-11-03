from .base_llm import BaseLLM
from .data import Dataset, benchmark

# Copilot was used as a teaching assistance and guide
def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    round_answer = round(float(answer), 5)
    tag_answer = f"<answer>{round_answer}</answer>"
    question_answer = {"question": prompt, "answer": tag_answer}
    return question_answer


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer
    
    # initialize dataset and model
    llm = BaseLLM()
    trainset = Dataset("train")

    # tokenize the dataset to prepare for training with LoRA adapters
    tokenized_trainset = TokenizedDataset(llm.tokenizer, trainset, format_example)

    # some r value that keeps the model size below 20MB
    rank = 8
    # about 4-5 times the rank
    lora_alpha = rank * 4

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        #lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # convert BaseLLM model to a PEFT model with LoRA adapters
    llm.model = get_peft_model(llm.model, peft_config)
    
    if llm.device == "cuda":
        # to avoid a bug with gradient_checkpointing and LoRA on CUDA
        llm.model.enable_input_require_grads()

    # setup training arguments
    training_args = TrainingArguments(
        gradient_checkpointing=True, # save GPU memory
        learning_rate=1e-3,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        #gradient_accumulation_steps=4,
        #warmup_steps=100,
        #fp16=True,
        #logging_steps=10,
        #save_total_limit=1,
        #save_steps=200,
    )

    # setup Trainer
    trainer = Trainer(
        model=llm.model, # PEFT model with LoRA adapters
        args=training_args, # TrainingArguments
        train_dataset=tokenized_trainset, # TokenizedDataset
    )

    trainer.train()
    # save PEFT model with LoRA adapters
    trainer.save_model(output_dir)
    #llm.model.save_pretrained(output_dir) # Alternative way to save the LoRA adapters
    # save tokenizer as well
    llm.tokenizer.save_pretrained(output_dir)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
