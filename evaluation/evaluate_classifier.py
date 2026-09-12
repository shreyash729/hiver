import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sentence_transformers import SentenceTransformer
import src.config as config

DATA_PATH = "data/amazonhelp_cleaned_with_intents.csv"
MODEL_NAME = config.EMBEDDING_MODEL

RANDOM_STATE = config.RANDOM_STATE
TEST_SIZE = config.TEST_SIZE


def load_data():
    data = pd.read_csv(DATA_PATH)

    return data[
        data["Intent"].notna()
    ].copy()


def create_split(data):
    return train_test_split(
        data["cleaned_customer_message"],
        data["Intent"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["Intent"]
    )


def evaluate_majority_baseline(X_train, X_test, y_train, y_test):

    model = DummyClassifier(
        strategy="most_frequent"
    )

    model.fit(
        X_train.to_frame(),
        y_train
    )

    predictions = model.predict(
        X_test.to_frame()
    )

    print("\n===== MAJORITY BASELINE =====")
    print(
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


def evaluate_tfidf_baseline(
    X_train,
    X_test,
    y_train,
    y_test
):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    predictions = model.predict(
        X_test_tfidf
    )

    print("\n===== TF-IDF + LOGISTIC REGRESSION =====")
    print(
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


def evaluate_bge_model(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\nLoading BGE model...")

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    print("Generating training embeddings...")

    train_embeddings = embedding_model.encode(
        X_train.tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Generating test embeddings...")

    test_embeddings = embedding_model.encode(
        X_test.tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(
        train_embeddings,
        y_train
    )

    predictions = model.predict(
        test_embeddings
    )

    print("\n===== BGE + LOGISTIC REGRESSION =====")
    print(
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


def main():

    data = load_data()

    X_train, X_test, y_train, y_test = create_split(
        data
    )

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")

    evaluate_majority_baseline(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluate_tfidf_baseline(
        X_train,
        X_test,
        y_train,
        y_test
    )

    evaluate_bge_model(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()