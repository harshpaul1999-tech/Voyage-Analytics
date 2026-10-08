import pandas as pd


def load_users_data():

    df = pd.read_csv("data/raw/users.csv")

    # Keep relevant columns
    df = df[["name", "gender", "age"]]

    # Remove missing
    df = df.dropna()

    # Remove invalid gender rows
    df = df[df["gender"] != "none"]

    # Convert to lowercase
    df["name"] = df["name"].str.split().str[0].str.lower()
    df["gender"] = df["gender"].str.lower()

    return df