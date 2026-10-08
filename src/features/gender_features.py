def prepare_features(df):

    X = df[["name", "age"]]
    y = df["gender"]

    return X, y