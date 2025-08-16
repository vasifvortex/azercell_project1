import os
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_data(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        df = pickle.load(f)
    return df


def test_model(model, df):
    X = df.drop("data_compl_usg_local_m1", axis=1)
    y = df["data_compl_usg_local_m1"]

    # Select features same as training
    numeric_features = [
        "refill_total_m2",
        "refill_total_m3",
        "frequency",
        "recency",
        "tenure",
        "lastrefillamount_m2",
        "tot_inact_status_days_l1m_m2",
        "refill_total_m4",
        "data_compl_usg_local_m2",
        "data_compl_usg_local_m3",
        "data_compl_usg_local_m4",
    ]
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    X = X[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"R²: {r2:.3f}")


def ramen_ratings_test_model(model, df):
    y = df["stars"]
    X = df.drop("stars", axis=1)

    numeric_features = X.columns[X.dtypes == "float32"]
    categorical_features = ["brand", "style", "country"]

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    # Select only columns you want
    X = X[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


def multisim_dataset_test_model(model, df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X, y = shuffle(X, y, random_state=42)
    numeric_features = X.columns[X.dtypes == "float64"].append(X.columns[X.dtypes == "int64"])
    categorical_features = X.columns[X.dtypes == "object"]

    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
    # Select only columns you want
    X = X[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


def data_usage_production_test():
    base_path = r"C:\Users\user\azercell_project1\src\data"
    model_path = r"C:\Users\user\azercell_project1\src\models"
    model_path = os.path.join(model_path, "data_usage_production_train_model.pkl")
    data_path = os.path.join(base_path, "data_usage_production.pkl")

    model = load_model(model_path)
    df = load_data(data_path)
    test_model(model, df)


def ramen_ratings_test():
    base_path = r"C:\Users\user\azercell_project1\src\data"
    model_path = r"C:\Users\user\azercell_project1\src\models"
    model_path = os.path.join(model_path, "ramen-ratings_train_model.pkl")
    data_path = os.path.join(base_path, "ramen-ratings.pkl")

    model = load_model(model_path)
    df = load_data(data_path)
    ramen_ratings_test_model(model, df)


def multisim_dataset_test():
    base_path = r"C:\Users\user\azercell_project1\src\data"
    model_path = r"C:\Users\user\azercell_project1\src\models"
    model_path = os.path.join(model_path, "multisim_dataset_train_model.pkl")
    data_path = os.path.join(base_path, "multisim_dataset.pkl")

    model = load_model(model_path)
    df = load_data(data_path)
    multisim_dataset_test_model(model, df)


def main():
    data_usage_production_test()
    ramen_ratings_test()
    multisim_dataset_test()


if __name__ == "__main__":
    main()
