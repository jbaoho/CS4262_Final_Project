"""
Fit Kernel PCA with the best-found configurations (sigmoid and linear)
on the cleaned Spaceship Titanic data and export transformed train/test CSVs.

Outputs (written to the same folder as the cleaned data):
    - train_sigmoid.csv
    - test_sigmoid.csv
"""
from pathlib import Path
import pandas as pd
from sklearn.decomposition import KernelPCA


DATA_DIR = Path("Cleaned Spaceship Titanic Dataset")
TRAIN_PATH = DATA_DIR / "train_clean.csv"
TEST_PATH = DATA_DIR / "test_clean.csv"

# Best configs from the grid search
SIGMOID_CFG = dict(kernel="sigmoid", n_components=10, gamma=0.1, coef0=0.0)


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def fit_transform_kpca(train_df, test_df, cfg):
    kpca = KernelPCA(
        **cfg,
        fit_inverse_transform=False,
        remove_zero_eig=True,
        random_state=42,
    )
    # Features exclude identifiers/target
    X_train = train_df.drop(columns=["PassengerId", "Transported"])
    X_test = test_df.drop(columns=["PassengerId"])

    Z_train = kpca.fit_transform(X_train)
    Z_test = kpca.transform(X_test)

    comp_cols = [f"comp_{i+1}" for i in range(Z_train.shape[1])]
    train_out = pd.DataFrame(Z_train, columns=comp_cols)
    train_out.insert(0, "PassengerId", train_df["PassengerId"].values)
    train_out["Transported"] = train_df["Transported"].values

    test_out = pd.DataFrame(Z_test, columns=comp_cols)
    test_out.insert(0, "PassengerId", test_df["PassengerId"].values)
    return train_out, test_out


def main():
    train_df, test_df = load_data()

    train_sigmoid, test_sigmoid = fit_transform_kpca(train_df, test_df, SIGMOID_CFG)

    train_sigmoid.to_csv(DATA_DIR / "train_sigmoid.csv", index=False)
    test_sigmoid.to_csv(DATA_DIR / "test_sigmoid.csv", index=False)

    print("Wrote:")
    print(" -", DATA_DIR / "train_sigmoid.csv")
    print(" -", DATA_DIR / "test_sigmoid.csv")


if __name__ == "__main__":
    main()
