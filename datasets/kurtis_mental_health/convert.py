import click
import pandas as pd


def convert_txt_to_parquet(csv_path, parquet_path):
    """
    Convert a TXT file with Context|Response structure to Parquet format.

    Parameters:
    - csv_path: str, path to the input CSV file.
    - parquet_path: str, path where the output Parquet file will be saved.
    """
    try:
        # Read the CSV file with '|' as the delimiter and header handling
        df = pd.read_csv(
            csv_path, delimiter="|", quoting=3, header=0
        )  # header=0 ensures the first row is treated as the header
        # Save the DataFrame to Parquet format
        df.to_parquet(parquet_path, index=False)
        click.echo(f"Successfully converted {csv_path} to {parquet_path}")
    except Exception as e:
        click.echo(f"Error during conversion: {e}")


if __name__ == "__main__":
    convert_txt_to_parquet("src/kurtis.txt", "kurtis.parquet")
