from dataclasses import dataclass
from typing import Any

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


@dataclass
class AssistantMaskingCollator(DataCollatorForLanguageModeling):
    """
    Data collator that masks everything in the input sequence except the assistant response.
    It expects the tokenizer to be set and a response_template to segment the input.
    """

    response_template: str = "<|start_of_role|>assistant<|end_of_role|>"
    tokenizer: PreTrainedTokenizerBase = None
    mlm: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to AssistantMaskingCollator")

        self.response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        # Handle dict input (transformers dataset style) or list input
        batch = super().torch_call(examples)

        input_ids = batch["input_ids"]
        labels = batch["labels"].clone()

        # Mask non-assistant tokens
        for i in range(len(input_ids)):
            # Mask prompt (everything before assistant response)
            # Simple linear search
            sequence = input_ids[i].tolist()
            p_len = len(self.response_token_ids)

            found_response = False
            for j in range(len(sequence) - p_len + 1):
                if sequence[j : j + p_len] == self.response_token_ids:
                    # Found response template starting at j
                    # Mask everything up to the end of the response template

                    # If this is the first one found:
                    if not found_response:
                        # Mask everything up to this point
                        labels[i, : j + p_len] = -100
                        found_response = True
                    else:
                        pass

            if not found_response:
                # If no assistant response found, mask entire sequence (don't train on prompt only)
                labels[i, :] = -100

        batch["labels"] = labels
        return batch
