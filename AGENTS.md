# Kurtis Agents Documentation

This document describes the common AI agents available in the Kurtis toolkit and their specialized roles.

## Specialized Agents

### 1. Kurtis (Empathetic AI Assistant)
- **Role**: Compassionate and empathetic AI assistant.
- **Key Traits**: Thoughtful, supportive, non-judgmental.
- **Guidelines**:
    - Listen actively and validate emotions.
    - Ask clarifying questions.
    - Offer evidence-based coping strategies.
    - Maintain appropriate boundaries.
- **Training Pattern**: Trained on mental health counseling datasets with specific alignment for empathy.

### 2. Echo (The Base Reasoner)
- **Role**: High-performance Small Language Model (SLM) focused on logical reasoning and concise instruction following.
- **Architecture**: Echo-DSRN (Dual State Recurrent Network).
- **Key Traits**: Efficient, precise, low latency.
- **Usage**: Typically used as the base model for specific domain adaptations.

## Common Architecture: DSRN-486M
Most agents in this toolkit utilize the **Dual State Recurrent Network** architecture, specifically the 486M parameter version, which balances performance and efficiency for local deployment.

## Training Paradigms
- **SFT (Supervised Fine-Tuning)**: Aligns the base model with specific instructions and character traits.
- **DPO (Direct Preference Optimization)**: Fines-tunes the model based on human preference data to improve safety and helpfulness.
- **TTT (Test-Time Training)**: Specialized fine-tuning performed on small target datasets to adapt the model's weights dynamically.
