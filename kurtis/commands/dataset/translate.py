# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------

import os
import click
from openai import OpenAI
from datasets import load_dataset


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)


# Define the target languages for translation.
TARGET_LANGUAGES = ["Italian", "Spanish", "German", "French", "Portuguese"]


def translate_text(sentence, languages=TARGET_LANGUAGES):
    """
    Translate a sentence into multiple languages using the OpenAI client via Ollama.
    Returns a dictionary mapping language names to their translations.

    The prompt used follows this template:

    Translate the text inside <text></text> from English

    <text>
    .........
    </text>

    to ......:
    """
    translations = {}
    for lang in languages:
        prompt = (
            f"Translate the following text from English into {lang}:\n"
            f"English: {sentence}\n\n{lang}: "
        )
        response = client.completions.create(
            model="hf.co/mradermacher/TowerInstruct-13B-v0.1-GGUF:IQ4_XS",
            prompt=prompt,
            max_tokens=2048,
            temperature=0.0,
        )
        translations[lang] = response.choices[0].text.strip()
        print(f"{lang} translation: {translations[lang]}\n---")
    return translations


@click.command(name="translate")
@click.option(
    "--dataset-name",
    required=True,
    help="Name of the Hugging Face dataset (e.g., 'squad').",
)
@click.option(
    "--split", default="train", help="Dataset split to process (default is 'train')."
)
@click.option(
    "--batch-size", default=100, help="Number of records to process per batch."
)
@click.option(
    "--start-from", default=0, help="Batch index to start from (useful for resuming)."
)
@click.option(
    "--output-dir",
    default="./translated_batches",
    help="Directory where batch files will be saved.",
)
@click.option(
    "--columns",
    default="question,answer",
    help="Comma-separated list of columns to translate.",
)
def command(dataset_name, split, batch_size, start_from, output_dir, columns):
    # Create the output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)
    state_file = os.path.join(output_dir, "translation-state.txt")

    # Load the dataset.
    dataset = load_dataset(dataset_name, split=split)
    total_records = len(dataset)
    total_batches = (total_records + batch_size - 1) // batch_size
    print(f"Dataset has {total_records} records in {total_batches} batches.")

    # Parse the columns to translate.
    cols_to_translate = [col.strip() for col in columns.split(",")]

    # Process each batch starting from the provided start batch.
    for batch_idx in range(start_from, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_records)
        print(f"\nProcessing batch {batch_idx} (records {start_idx} to {end_idx})")

        # Use select to get a Dataset object for the current batch, then convert to pandas.
        batch_dataset = dataset.select(range(start_idx, end_idx))
        batch_df = batch_dataset.to_pandas()

        # For each column we want to translate, add new columns for each target language.
        for col in cols_to_translate:
            if col not in batch_df.columns:
                print(f"Column '{col}' not found in the dataset. Skipping.")
                continue
            # Create new empty columns for each language translation.
            for lang in TARGET_LANGUAGES:
                new_col = f"{col}_{lang.lower()}"
                batch_df[new_col] = ""

            # Process each row for the current column.
            for idx, text in batch_df[col].items():
                if not isinstance(text, str):
                    continue  # Skip non-string entries.
                translations = translate_text(text, languages=TARGET_LANGUAGES)
                for lang in TARGET_LANGUAGES:
                    new_col = f"{col}_{lang.lower()}"
                    batch_df.at[idx, new_col] = translations[lang]

        # Save the processed batch to a Parquet file.
        batch_file = os.path.join(output_dir, f"batch_{batch_idx}.parquet")
        batch_df.to_parquet(batch_file, index=False)
        print(f"Saved batch {batch_idx} to {batch_file}")

        # Write/update a state file with the current batch index.
        with open(state_file, "w") as f:
            f.write(str(batch_idx))
        print(f"Updated state file with batch index {batch_idx}")

    print("\nTranslation completed.")
