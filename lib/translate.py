from deep_translator import GoogleTranslator
from langdetect import detect
import pandas as pd


class Translator:
    def __init__(self, df: pd.DataFrame, need_translation: list[str]):
        self.translator = GoogleTranslator()
        self.df = df
        self.need_translation = need_translation
    
    def _translate_to_english(self, text):
        translated = self.translator.translate(text)
        try:
            source_lang = detect(text)
        except:
            source_lang = "unknown"
        return translated, source_lang
    
    def _translate_cell(self, cell):
        if pd.isna(cell):
            return pd.Series([cell, None])
        text_str = str(cell)
        if not text_str.strip():
            return pd.Series([cell, None])
        try:
            translated_text, lang = self._translate_to_english(text_str)
        except Exception:
            # Fallback: keep original text and unknown language
            translated_text, lang = text_str, None
        return pd.Series([translated_text, lang])

    def translate_required_columns(self) -> pd.DataFrame:
        """
        Translate columns listed in NEED_TRANSLATION (if present in df).
        For each translated column <col>, also add <col>_language with the
        detected/origin language when available.
        """
        print(f"Translating columns: {self.need_translation}")
        translated_df = self.df.copy()

        for col in self.need_translation:
            if col in translated_df.columns:
                result = translated_df[col].apply(self._translate_cell)
                translated_df[col] = result.iloc[:, 0]
                translated_df[f"{col}_language"] = result.iloc[:, 1]

        return translated_df
