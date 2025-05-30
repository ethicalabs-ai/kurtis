# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------

import os
from peft import LoraConfig, TaskType

TRANSFORMERS_MODEL_PRETRAINED = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
INFERENCE_MODEL = "ethicalabs/Kurtis-E1-SmolLM2-1.7B-Instruct"
MODEL_NAME = "Kurtis-E1.1-SmolLM2-1.7B-Instruct"
MODEL_DPO_NAME = "Kurtis-E1.1-SmolLM2-1.7B-Instruct-DPO"
HF_REPO_ID = "ethicalabs/Kurtis-E1.1-SmolLM2-1.7B-Instruct"
HF_DPO_REPO_ID = "ethicalabs/Kurtis-E1.1-SmolLM2-1.7B-Instruct-DPO"
DATASET_NAME = "ethicalabs/Kurtis-E1-SFT"
DPO_DATASET_NAME = "mrs83/kurtis_mental_health_dpo"
OPENAI_API_URL = "http://localhost:11434/v1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
TRANSLATION_GGUF_MODEL = "hf.co/mradermacher/TowerInstruct-7B-v0.2-GGUF:IQ4_XS"
TRANSLATION_LANGUAGES = [
    "Italian",
    "Spanish",
    "German",
    "French",
    "Portuguese",
    "Dutch",
]

TRAINING_CONFIG = {
    "dataset_name": DATASET_NAME,
    "dataset_split": "train",
    "prompt_column": "question",
    "response_column": "answer",
    "max_length": 1024,
    "num_train_epochs": 3,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 5e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
TRAINING_DPO_CONFIG = {
    "max_length": 1024,
    "num_train_epochs": 2,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 1e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
DATASETS_CONFIG = {
    "kurtis_mental_health": {
        "dataset_name": "mrs83/kurtis_mental_health_initial",
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
        "dataset_max_samples": 9000,
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
                    "category:abilities",
                    "category:altruism",
                    "category:altruism-biological",
                    "category:altruism-empirical",
                    "category:analysis",
                    "category:authenticity",
                    "category:basing-epistemic",
                    "category:boundary",
                    "category:biodiversity",
                    "category:causation-counterfactual",
                    "category:causation-metaphysics",
                    "category:causation-probabilistic",
                    "category:change",
                    "category:chaos",
                    "category:cognitive-science",
                    "category:communitarianism",
                    "category:common-good",
                    "category:computational-philosophy",
                    "category:concept-evil",
                    "category:concept-religion",
                    "category:computational-mind",
                    "category:computational-philosophy",
                    "category:computational-linguistics",
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
    "dataset_name": DATASET_NAME,
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
    r=12,
    lora_alpha=24,
    lora_dropout=0.0075,
    target_modules=[
        "down_proj",
        "gate_proj",
        "up_proj",
        # "k_proj",
        "o_proj",
        # "q_proj",
        # "v_proj",
    ],
    bias="none",
    use_dora=True,
)

QA_INSTRUCTION = "You are a compassionate and empathetic mental-health assistant named Kurtis, trained by ethicalabs.ai.\nYou provide thoughtful and supportive responses to user queries."
CHAT_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n"
    + QA_INSTRUCTION
    + "<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
