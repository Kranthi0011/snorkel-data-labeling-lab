import pandas as pd
from sklearn.model_selection import train_test_split

def load_sms_spam_dataset(test_size=0.2, random_state=42):
    df = pd.read_csv(
        "data/SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label_text", "text"]
    )
    df["label"] = (df["label_text"] == "spam").astype(int)
    df = df.drop(columns=["label_text"])

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train_labels = df_train["label"].values.copy()

    return df_train, df_test, df_train_labels
