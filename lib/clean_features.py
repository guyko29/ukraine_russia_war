from typing import Any, List
import pandas as pd


class FeatureCleaner:
    """
    Cleans special characters from specified DataFrame columns.
    Keeps only alphanumeric characters and the following punctuation: . , space / : ?
    """

    def __init__(self, df: pd.DataFrame, features: List[str]) -> None:
        self._df: pd.DataFrame = df
        self._features: List[str] = list(features)
        # Allowed punctuation as specified: point, comma, spaces, slash, colon, question mark
        self._allowed_chars = {".", ",", " ", "/", ":", "?", '!', '-'}

    def _clean_text(self, value: Any) -> Any:
        """
        Return the value with disallowed special characters removed.
        - Preserve None/NaN as-is.
        - Convert non-strings to string before cleaning (to ensure consistent behavior).
        """
        if pd.isna(value):
            return value
        if not isinstance(value, str):
            value = str(value)
        return "".join(ch for ch in value if ch.isalnum() or ch in self._allowed_chars)

    def clean(self) -> pd.DataFrame:
        """
        Apply cleaning to all specified features and return a new cleaned DataFrame.
        Columns not present in the DataFrame are skipped.
        """
        df_clean = self._df.copy(deep=True)
        for column_name in self._features:
            if column_name in df_clean.columns:
                df_clean[column_name] = df_clean[column_name].apply(self._clean_text)
        return df_clean


