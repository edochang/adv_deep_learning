# Copilot was used as a teaching assistance and guide
def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .cot import CoTModel
    from .data import is_answer_valid

    # use the 1.7B SmolLM2 model
    #checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
    
    # use the CoTModel to generate data
    model = CoTModel(checkpoint=checkpoint)

    # setup and batch all prompts from train data
    from .data import Dataset
    trainset = Dataset("train") # about a 1000 train samples

    trainset_size = range(len(trainset))
    prompts = [model.format_prompt(trainset[i][0]) for i in trainset_size]
    answers = [trainset[i][1] for i in trainset_size]

    print(f"Generating dataset with {len(prompts)} samples...") # debug

    import json
    best_samples = []

    # for each prompt, generate multiple samples with temperature sampling
    generations = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

    print(f"Generated {len(generations)} samples...") # debug

    for i, prompt in enumerate(prompts):
        answer = answers[i]
        for j in range(oversample):
            sample = generations[i][j]

            sample_answer = model.parse_answer(sample)

            if (is_answer_valid(sample_answer, float(answer))):
                # save the valid sample
                best_sample = []
                best_sample.append(prompt)
                best_sample.append(answer)
                best_sample.append(sample)
                best_samples.append(best_sample)
                break

    with open(f"{output_json}", "w") as f:
        json.dump(best_samples, f, indent=4)

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
