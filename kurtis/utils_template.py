def ensure_template_compatibility(tokenizer):
    """
    Ensures the tokenizer's chat template supports TRL's assistant_only_loss=True.
    It checks for the presence of {% generation %} tags and patches the template
    if they are missing, specifically targeting standard Granite/Jinja templates.
    """
    original_template = tokenizer.chat_template

    if not original_template:
        print("Warning: No chat template found on tokenizer.")
        return

    if "{% generation %}" in original_template:
        print("✅ Chat template already supports assistant masking ({% generation %} found).")
        return

    print("⚠️ Chat template missing assistant masking tags. Attempting to patch...")

    # Target string to replace (standard Granite pattern)
    target = "{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + content.val }}"
    replacement = "{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' }}{% generation %}{{ content.val }}{% endgeneration %}"

    if target in original_template:
        patch = original_template.replace(target, replacement)
        tokenizer.chat_template = patch
        print("✅ Chat template successfully patched for assistant_only_loss.")
    else:
        print(
            "❌ Could not verify template structure for patching. assistant_only_loss may fail explicitly or implicitly."
        )
        print("Please verify the chat template manually.")
