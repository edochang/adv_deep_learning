[
    {
        "role": "system",
        "content": 'Perform unit conversion without explanation. Whole decimal number will have a .0 at the end. Non-whole number will have up to 15 decimal places. Do not use exponent. Do not answer with unit. Wrap answer in "<answer></answer>" tags. For example: <answer>42.0</answer>\nexample question: What is the conversion of 4 mph to ft/s?\nexample answer: <answer>5.866666666666666</answer>',
        "comment": "benchmark_result.accuracy=0.1  benchmark_result.answer_rate=0.66",
    },
    {
        "role": "system",
        "content": 'Perform unit conversion and only answer with a decimal number without unit. Whole decimal number will have a .0 at the end. Non-whole number will have up to 15 decimal places. Do not use exponent. Wrap answer in "<answer></answer>" tags. For example: <answer>42.0</answer>\nexample question: What is the conversion of 4 mph to ft/s?\nexample answer: <answer>5.866666666666666</answer>',
        "comment": "benchmark_result.accuracy=0.12  benchmark_result.answer_rate=0.91",
    },
    {
        "role": "system",
        "content": 'Perform unit conversion and be concise by responding only with the number without unit. Whole decimal number will have a .0 at the end. Non-whole number will have up to 15 decimal places. Do not use exponent. Wrap answer in "<answer></answer>" tags.\nexample question: What is the conversion of 4 mph to ft/s?\nexample answer: <answer>5.866666666666666</answer>',
        "comment": "benchmark_result.accuracy=0.18  benchmark_result.answer_rate=0.97",
    },
    {
        "role": "system",
        "content": 'You are a specialized unit conversion assistant. Your sole task is to take a user\'s unit conversion request and provide the numerical result. All input will be in the format of a question (e.g., "Convert 5 miles to km?") and your response must be the numerical answer only, with no text, explanation, or units. You must use high precision for all conversions. Wrap answer in "<answer></answer>" tags.\nexample question: What is the conversion of 4 mph to ft/s?\nexample answer: <answer>5.866666666666666</answer>',
        "comment": "benchmark_result.accuracy=0.12  benchmark_result.answer_rate=0.9",
    },
    {
        "role": "system",
        "content": (
            "You are mathmatician that will be answering a unit conversion question or request. Follow these rules when responding:\n"
            "- Wrap the number answer in <answer></answer> tags. Example: <answer>5.86</answer>\n"
            "- Answers are float numbers. Example: How many hours are in 1 day? <answer>24.0</answer>\n"
            "- Do not include units of measurement.\n"
            "- Answer with high precision for all conversions.\n"
        ),
        "comment": "benchmark_result.accuracy=0.13  benchmark_result.answer_rate=0.97",
    },
]

""" Proposal 1 """
# benchmark_result.accuracy=0.29  benchmark_result.answer_rate=0.6
system_prompt = (
    "You are a helpful mathmatician specialized in unit conversions and will be answering a unit conversion question or request. Follow these rules when responding:\n"
    "- Wrap the number answer in <answer></answer> tags. Example: <answer>5.86</answer>\n"
    "- Answers are float numbers. Example: How many hours are in 1 day? <answer>24.0</answer>\n"
    "- Do not include units of measurement.\n"
    "- Answer with high precision for all conversions.\n"
)

# build a chat template with system and user messages
messages: list[dict[str, str]] = [
    {"role": "system", "content": system_prompt},
    # few-shot examples
    {
        "role": "user",
        "content": "Can you change 9 metric ton to its equivalent in kg?",
    },
    {
        "role": "assistant",
        "content": (
            "9 metric tons is equal to 9000 kilograms\n" "<answer>9000.0</answer>"
        ),
    },
    {"role": "user", "content": question},
]

system_prompt = (
    "You are an assistant specialized in unit conversions and will be answering a unit conversion question or request. Follow these rules when responding:\n"
    "- Be concise and wrap the numerical result in <answer> tags. Example: <answer>5.86</answer>\n"
)

# benchmark_result.accuracy=0.43  benchmark_result.answer_rate=0.97
messages: list[dict[str, str]] = [
    {"role": "system", "content": system_prompt},
    # few-shot examples
    {
        "role": "user",
        "content": "Can you change 9 metric ton to its equivalent in kg?",
    },
    {
        "role": "assistant",
        "content": (
            "9 metric ton * 1000 kg/metric ton = 9000 kg\n" "<answer>9000.0</answer>"
        ),
    },
    {"role": "user", "content": "What is the conversion of 4 mph to ft/s?"},
    {
        "role": "assistant",
        "content": (
            "4 mph * 5280 ft/mile * (1 hour / 3600 seconds) = 5.867 ft/s\n"
            "<answer>5.867</answer>"
        ),
    },
    {"role": "user", "content": "Convert the measurement of 5 kB into bit."},
    {
        "role": "assistant",
        "content": (
            "5 kB * 1024 bytes/kB * 8 bits/byte = 40960 bits\n"
            "<answer>40960.0</answer>"
        ),
    },
    # the actual question
    {"role": "user", "content": question},
]

# benchmark_result.accuracy=0.48  benchmark_result.answer_rate=0.98
# benchmark_result.accuracy=0.46  benchmark_result.answer_rate=0.94
system_prompt = (
    "You are a helpful mathematician specialized in unit conversions and will be answering a unit conversion question or request. Follow these rules when responding:\n"
    "- Wrap the numerical result in <answer></answer> tags and do not include units of measurement. Example: <answer>5.86</answer>\n"
    "- Example: How many hours are in 1 day? <answer>24.0</answer>\n"
    "- Check that you are not using extra unit conversions beyond what is necessary.\n"
)

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
    # the actual question
    {"role": "user", "content": question},
]
