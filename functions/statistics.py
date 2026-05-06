import polars as pl
from sklearn.feature_selection import mutual_info_classif

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