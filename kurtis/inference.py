import torch

from .utils import get_device


def inference_model(
    model,
    tokenizer,
    config,
    input_text,
):
    """
    Generates a response from Kurtis using the trained T5 model.

    Args:
        model: The trained language model.
        tokenizer: The tokenizer used to encode and decode the text.
        input_text (str): The input text/question from the user.

    Returns:
        str: The generated response from the model or an error message.
    """

    response = None
    try:
        device = get_device()
        # Ensure the model is on the correct device
        model.eval()
        messages = [
            {
                "role": "system",
                "content": config.QA_INSTRUCTION,
            },
            {"role": "user", "content": input_text},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(f"{input_text}assistant\n", return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.4,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.0,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = outputs[0][inputs.shape[-1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        response = f"An error occurred during inference: {str(e)}"

    fallback_response = "I'm sorry, I don't have an answer for that."
    return response.strip() if response else fallback_response
