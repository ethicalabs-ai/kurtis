from peft import LoraConfig, TaskType

TRANSFORMERS_MODEL_PRETRAINED = "ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct-Zero"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
INFERENCE_MODEL = "ethicalabs/Kurtis-SmolLM2-1.7B-Instruct"
MODEL_NAME = "Kurtis-E1.1-Qwen2.5-3B-Instruct-SFT"
MODEL_DPO_NAME = "Kurtis-E1.1-Qwen2.5-3B-Instruct-DPO"
HF_REPO_ID = "ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct-SFT"
HF_DPO_REPO_ID = "ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct-DPO"
DATASET_NAME = "ethicalabs/Kurtis-E1-SFT"
DPO_DATASET_NAME = "ethicalabs/Kurtis-E1-DPO"

TRAINING_CONFIG = {
    "dataset_name": "ethicalabs/Kurtis-E1-SFT",
    "dataset_split": "train",
    "prompt_column": "question",
    "response_column": "answer",
    "max_length": 1024,
    "num_train_epochs": 2,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 5e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
TRAINING_DPO_CONFIG = {
    "max_length": 1024,
    "num_train_epochs": 1,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 1e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
DATASETS_CONFIG = {
    "kurtis_mental_health": {
        "dataset_name": "ethicalabs/Kurtis-E1-Multilingual-01-SFT",
        "dataset_domain": "mental-health",
        "prompt_column": "question",
        "response_column": "answer",
        "dataset_split": "train",
        "max_length": 1024,
    },
    "amod_mental_health_counseling_conversations": {
        "dataset_name": "Amod/mental_health_counseling_conversations",
        "dataset_domain": "mental-health",
        "prompt_column": "Context",
        "response_column": "Response",
        "dataset_split": "train",
        "dataset_max_samples": 3500,
        "max_length": 1024,
    },
    "google_boolq": {
        "dataset_name": "google/boolq",
        "dataset_domain": "general-qa",
        "prompt_column": "question",
        "response_column": "passage",
        "dataset_split": "train",
        "dataset_max_samples": 7500,
        "max_length": 1024,
    },
    "strix_philosophy_qa": {
        "dataset_name": "sayhan/strix-philosophy-qa",
        "dataset_domain": "philosophy",
        "prompt_column": "question",
        "response_column": "answer",
        "dataset_split": "train",
        "max_length": 1024,
        "dataset_select": [
            {
                "classes": [
                    "category:recursive-functions",
                    "category:epistemic-game",
                    "category:logics-for-games",
                    "category:frege-theorem",
                    "category:game-theory",
                    "category:idealism",
                    "category:mill-moral-political",
                    "category:computational-linguistics",
                    "category:computational-complexity",
                    "category:settheory-alternative",
                    "category:logic-algebraic-propositional",
                    "category:formal-epistemology",
                    "category:principia-mathematica",
                    "category:logic-temporal",
                    "category:social-choice",
                    "category:evil",
                    "category:formal-belief",
                    "category:consciousness-temporal",
                    "category:meaning",
                    "category:legal-probabilism",
                    "category:logic-inductive",
                    "category:descartes-epistemology",
                    "category:epistemology-bayesian",
                    "category:proof-theory" "category:logic-ai",
                    "category:common-knowledge",
                    "category:logic-classical",
                    "category:logicism",
                    "category:continuity",
                    "category:equal-opportunity",
                    "category:bounded-rationality",
                    "category:kant-transcendental-idealism",
                    "category:logic-belief-revision",
                    "category:logic-informal",
                    "category:normativity-metaethics",
                    "category:relativism",
                    "category:pragmatics",
                    "category:logic-dialogical",
                    "category:chaos",
                    "category:collective-responsibility",
                    "category:biodiversity",
                    "category:consciousness",
                    "category:chance-randomness",
                    "category:action-perception",
                    "category:reasoning-automated",
                    "category:egalitarianism",
                    "category:chance-randomness",
                    "category:heidegger",
                    "category:philosophy-mathematics",
                    "category:philosophy-religion",
                    "category:consciousness-neuroscience",
                    "category:vegetarianism",
                    "category:turing-machine",
                    "category:abstract-objects",
                    "category:consciousness-animal",
                    "category:physics-experiment",
                    "category:plato-theaetetus",
                    "category:consciousness-unity",
                    "category:moral-sentimentalism",
                    "category:dreams-dreaming",
                    "category:fallacies",
                    "category:realism-theory-change",
                    "category:folkpsych-simulation",
                    "category:epistemology",
                    "category:exploitation",
                    "category:feminist-science",
                    "category:feminism-trans",
                    "category:logic-if",
                    "category:paradoxes-contemporary-logic",
                    "category:liar-paradox",
                    "category:mysticism",
                    "category:kant-religion",
                    "category:self-consciousness",
                    "category:self-reference",
                    "category:epistemic-paradoxes",
                    "category:games-abstraction",
                    "category:neoliberalism",
                    "category:marxism-analytical",
                    "category:quantum-field-theory",
                    "category:quantum-gravity",
                    "category:logic-ontology",
                    "category:feminist-philosophy-biology",
                    "category:ecology",
                ],
                "max_length": 1024,
            },
        ],
    },
    "tellikoroma_mental_health": {
        "dataset_name": "tellikoroma/mentalhealth",
        "dataset_domain": "mental-health",
        "prompt_column": "pattern",
        "response_column": "response",
        "dataset_split": "train",
        "dataset_select": [
            {
                "classes": [
                    "tag:greeting",
                    "tag:morning",
                    "tag:afternoon",
                    "tag:evening",
                    "tag:night",
                ],
                "max_samples": 7000,
                "max_length": 1024,
            },
            {
                "classes": ["tag:learn-more", "tag:user-disagree", "tag:user-advice"],
                "max_samples": 5000,
                "max_length": 1024,
            },
            {
                "classes": ["tag:meditation", "tag:user-meditation"],
                "max_samples": 4000,
                "max_length": 1024,
            },
            {
                "classes": [
                    "tag:learn-mental-health",
                    "tag:mental-health-fact",
                    "tag:pandora-useful",
                ],
                "max_samples": 5000,
                "max_length": 1024,
            },
            {
                "classes": [
                    "tag:fact-1",
                    "tag:fact-2",
                    "tag:fact-3",
                    "tag:fact-5",
                    "tag:fact-6",
                    "tag:fact-7",
                    "tag:fact-8",
                    "tag:fact-9",
                    "tag:fact-10",
                    "tag:fact-11",
                    "tag:fact-12",
                    "tag:fact-13",
                    "tag:fact-14",
                    "tag:fact-15",
                    "tag:fact-16",
                    "tag:fact-17",
                    "tag:fact-18",
                    "tag:fact-19",
                    "tag:fact-20",
                    "tag:fact-21",
                    "tag:fact-22",
                    "tag:fact-23",
                    "tag:fact-24",
                    "tag:fact-25",
                    "tag:fact-26",
                    "tag:fact-27",
                    "tag:fact-28",
                    "tag:fact-29",
                    "tag:fact-30",
                    "tag:fact-31",
                    "tag:fact-32",
                ],
                "max_samples": 4000,
                "max_length": 1024,
            },
        ],
        "max_length": 1024,
    },
}


EVALUATION_DATASET = {
    "dataset_name": "ethicalabs/Kurtis-E1-SFT",
    "prompt_column": "question",
    "response_column": "answer",
    "dataset_max_samples": 500,
    "split": "validation",
    "max_length": 1024,
}


FINAL_DATASETS = {
    # "datasets/kurtis_mental_health": "ethicalabs/kurtis_mental_health",
    "datasets/kurtis_e1_sft/": DATASET_NAME,
}
DPO_DATASETS = {"datasets/kurtis_mental_health_dpo_clean": DPO_DATASET_NAME}

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=64,
    lora_dropout=0.075,
    target_modules=[
        "down_proj",
        "gate_proj",
        "up_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "v_proj",
    ],
    bias="none",
    use_dora=True,
)

QA_INSTRUCTION = "You are a compassionate and empathetic mental-health assistant named Kurtis, trained by ethicalabs.ai.\nYou provide thoughtful and supportive responses to user queries."
CHAT_TEMPLATE = (
    "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- '"
    + QA_INSTRUCTION
    + "' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\n"
    + QA_INSTRUCTION
    + "<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
)
