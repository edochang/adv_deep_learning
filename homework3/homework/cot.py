from .base_llm import BaseLLM


# Copilot was used as a teaching assistance and guide
class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        system_prompt = (
            "You are a helpful mathematician specialized in unit conversions and will be answering a unit conversion question or request. Follow these rules when responding:\n"
            "- Wrap the numerical result in <answer></answer> tags and do not include units of measurement. Example: <answer>5.86</answer>\n"
            "- Example: How many hours are in 1 day? <answer>24.0</answer>\n"
            "- Check that you are not using extra unit conversions beyond what is necessary.\n"
        )

        # build a chat template with system and user messages
        # few-shot examples
        # Use LLM to determine common conversion patterns in train.json: These are:
        
        # - mass conversion ✔️
        # - speed conversion 
        # - data storage conversion 
        # - time conversion ✔️
        # - length conversion 
        # - volume conversion ✔️
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Can you change 9 metric ton to its equivalent in kg?",
            },
            {
                "role": "assistant",
                "content": (
                    "Conversion used: 1000 kg/metric ton\n"
                    "Calculation: 9 * 1000 = 9000\n"
                    "<answer>9000</answer>"
                ),
            },
            {"role": "user", "content": "What is the equivalent of 9 milliliter in mm^3?"},
            {
                "role": "assistant",
                "content": (
                    "Conversion used: 1000 mm^3/milliliter\n"
                    "Calculation: 9 * 1000 = 9000\n"
                    "<answer>9000</answer>"
                ),
            },
            {"role": "user", "content": "how many weeks are there in 2 years?"},
            {
                "role": "assistant",
                "content": (
                    "Conversion used: 52.142857 weeks/year\n"
                    "Calculation: 2 * 52.142857 = 104.285714\n"
                    "<answer>104.285714</answer>"
                ),
            },
            # the actual question
            {"role": "user", "content": question},
        ]

        # apply the model's chat template using the tokenizer utility
        output = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # print(f"CoTModel format_prompt: Generated prompt:\n{output}") # debug
        return output


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
