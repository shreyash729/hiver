import re
import html
import pandas as pd


def clean_text(text):
    """
    Clean a customer message or AmazonHelp response.

    Steps:
    1. Decode HTML entities
    2. Remove URLs
    3. Remove Twitter @mentions
    4. Remove AmazonHelp agent signatures such as ^SK
    5. Convert hashtags to normal words
    6. Remove remaining non-alphanumeric characters
    7. Normalize whitespace
    8. Convert to lowercase
    """

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove Twitter @mentions
    text = re.sub(r'@\w+', ' ', text)

    # Remove AmazonHelp agent signatures such as ^SK, ^HN
    text = re.sub(r'\^[A-Z]{2}\b', ' ', text)

    # Convert hashtags to normal words
    # Example: #MotoG5SPlus -> MotoG5SPlus
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()


def clean_pairs(df):
    """
    Clean customer messages and AmazonHelp responses
    in a customer-response pair dataframe.
    """

    df = df.copy()

    df["cleaned_customer_message"] = (
        df["customer_message"].apply(clean_text)
    )

    df["cleaned_amazon_response"] = (
        df["amazon_response"].apply(clean_text)
    )

    # Remove rows where either side became empty
    df = df[
        (df["cleaned_customer_message"].str.strip() != "") &
        (df["cleaned_amazon_response"].str.strip() != "")
    ].copy()

    # Remove exact duplicate customer-response pairs
    df = df.drop_duplicates(
        subset=[
            "cleaned_customer_message",
            "cleaned_amazon_response"
        ]
    )

    return df.reset_index(drop=True)


def load_cleaned_dataset(path):
    """
    Load the final cleaned dataset used by the agent.
    """

    df = pd.read_csv(path)

    required_columns = [
        "cleaned_customer_message",
        "cleaned_amazon_response",
        "Intent"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df