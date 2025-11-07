# Copilot was used as a teaching assistance and guide
def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .cot import CoTModel
    from .data import is_answer_valid

    # use the 1.7B SmolLM2 model
    checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    #checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
    
    # use the CoTModel to generate data
    model = CoTModel(
        checkpoint=checkpoint,
        use_bfloat16=True,
    )

    # setup and batch all prompts from train data
    from .data import Dataset
    trainset = Dataset("train") # about a 1000 train samples
    #trainset = trainset[:34] # debug with smaller set

    trainset_size = range(len(trainset))
    questions = [trainset[i][0] for i in trainset_size]
    prompts = [model.format_prompt(trainset[i][0]) for i in trainset_size]
    answers = [trainset[i][1] for i in trainset_size]

    print(f"Generating dataset with {len(prompts)} prompts...") # debug

    import json
    best_samples = []

    # process in chunks to avoid memory issues with oversampling and large models
    # example: 5 prompts * 10 oversamples = 50 generations at once
    chunk_size = 5  # adjust based on your GPU memory capacity

    # chunk_size = 5 : Accepted 879 / 1000 samples.  Ran about ~30 minutes on 1.7B model with bfloat16


    # observations: apply a different batching logic here because the batching logic in batch_generate does not limit memory usage well enough. 
    # for example, with 32 prompts and 10 oversamples, the model would try to generate 320 sequences at once, which can be too large for GPU memory.
    # instead, we process smaller chunks of prompts to keep memory usage manageable.
    # For the 1.7B model with bfloat16:
    # - 320 sequences: ~4-8GB just for KV cache + intermediates
    # - 50 sequences: ~600MB-1.2GB
    # test and pick the largest chunk size that fits your GPU without OOM. Smaller than 5 usually underutilizes the 3090 and is slower overall; larger chunks improve throughput until you hit VRAM limits.

    from tqdm import tqdm

    for idx in tqdm(range(0, len(prompts), chunk_size), desc="Generating dataset in chunks"):
        chunk_questions = questions[idx : idx + chunk_size]
        chunk_prompts = prompts[idx : idx + chunk_size]
        chunk_answers = answers[idx : idx + chunk_size]

        # generate samples for the current chunk
        generations = model.batched_generate(
            chunk_prompts, num_return_sequences=oversample, temperature=temperature
        )

        for i, prompt in enumerate(chunk_prompts):
            answer = chunk_answers[i]
            for j in range(oversample):
                sample = generations[i][j]
                sample_answer = model.parse_answer(sample)

                if is_answer_valid(sample_answer, float(answer)):
                    # save the valid sample
                    best_sample = [
                        chunk_questions[i],
                        answer,
                        sample
                    ]
                    best_samples.append(best_sample)
                    break

        print(f"Chunk {idx}: Accepted {len(best_samples)} / {len(prompts)} samples.") # debug

    """
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
                best_sample = [
                    prompt,
                    answer,
                    sample
                ]
                best_samples.append(best_sample)
                break
    """

    print(f"Accepted {len(best_samples)} / {len(prompts)} samples.") # debug

    with open(f"{output_json}", "w") as f:
        json.dump(best_samples, f, indent=4)

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
