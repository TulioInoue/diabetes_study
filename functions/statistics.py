import polars as pl
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import joblib

cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)

def find_IV(df: pl.DataFrame, target_col: str, feature_cols: list[str]) -> pl.DataFrame:

    """
        Find the Importance Value (IV) of a target value.

        Args:
            df (pl.DataFrame): dataset.
            target_col (str): target column used to find importance value.
            feature_cols (list[str]): columns where we suspect to have high importance value related to target_column.
    """
    
    df_prep = df.select(feature_cols + [target_col]).with_columns([
        pl.col(pl.Enum).to_physical(),
        pl.col(pl.Boolean).cast(pl.Int8),
        pl.col(target_col).cast(pl.Int8).alias("TARGET_NUM")
    ])

    X = df_prep.select(feature_cols).to_pandas()
    y = df_prep.select("TARGET_NUM").to_pandas().values.ravel()

    importancias = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

    result = pl.DataFrame({
        "FEATURE": feature_cols,
        "IMPORTANCE": importancias
    }).sort("IMPORTANCE", descending=True)

    return result

def getting_best_model(
    model,
    X_train: list,
    y_train: list,
    path: str,
    param_grid: dict,
    scoring,
    n_iter: int = 50,
) -> None:

    randomized_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid,
        n_iter = n_iter,
        cv = cv,
        n_jobs = -1,
        verbose = 2,
        scoring = scoring,
        refit = True,
        error_score = "raise",
        
    )

    X_train_fixed = X_train.astype('float32') if hasattr(X_train, 'astype') else X_train

    randomized_search.fit(
        X = X_train_fixed,
        y = y_train
    )

    print(f"Best ROC AUC Score: {randomized_search.best_score_:.4f}")

    joblib.dump(
        value = randomized_search.best_estimator_,
        filename = path
    )