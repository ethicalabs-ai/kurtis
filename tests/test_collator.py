from transformers import AutoTokenizer

from kurtis.collator import AssistantMaskingCollator


def test_masking_collator():
    model_name = "ibm-granite/granite-4.0-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    response_template = "<|start_of_role|>assistant<|end_of_role|>"
    collator = AssistantMaskingCollator(tokenizer, response_template=response_template)

    # Test case 1: Single turn with generic text
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(text)
    input_ids = encoded["input_ids"]
    batch = collator([{"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}])

    labels = batch["labels"][0]

    # For robust testing, we can check decoding
    response_start_idx = -1
    for i in range(len(labels)):
        if labels[i] != -100:
            response_start_idx = i
            break

    assert response_start_idx > 0, "Everything was masked or response started at 0"

    # Verify the unmasked part matches "Hi" + EOS
    unmasked_ids = batch["input_ids"][0][response_start_idx:]
    decoded_response = tokenizer.decode(unmasked_ids, skip_special_tokens=True)

    # Granite adds newline sometimes
    assert "Hi" in decoded_response

    # Confirm everything before is masked
    assert all(label == -100 for label in labels[:response_start_idx])


def test_no_response_template():
    # Test behavior when response template is missing (should mask all)
    model_name = "ibm-granite/granite-4.0-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = AssistantMaskingCollator(tokenizer, response_template="<|NON_EXISTENT|>")

    input_ids = [1, 2, 3, 4]
    batch = collator([{"input_ids": input_ids, "attention_mask": [1] * 4}])

    assert all(batch["labels"][0] == -100), "Should mask all if template not found"
