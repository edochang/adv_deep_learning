from .base_llm import BaseLLM
from .sft import test_model

# Copilot was used as a teaching assistance and guide
def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    raise NotImplementedError()
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer
    from .data import Dataset
    
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
        gradient_checkpointing=True,
        learning_rate=1e-3,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,

    )

    # setup Trainer
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_trainset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    llm.tokenizer.save_pretrained(output_dir)
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
