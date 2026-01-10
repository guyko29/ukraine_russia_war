from deep_translator import GoogleTranslator
from langdetect import detect
import pycountry
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor


class Translator:
    def __init__(self, df: pd.DataFrame, need_translation: list[str], max_workers: int = 8):
        # Thread-local storage to keep a separate translator per thread
        self._thread_local = threading.local()
        self.df = df
        self.need_translation = need_translation
        self.max_workers = max_workers

    def _get_thread_local_translator(self) -> GoogleTranslator:
        translator = getattr(self._thread_local, "translator", None)
        if translator is None:
            translator = GoogleTranslator()
            setattr(self._thread_local, "translator", translator)
        return translator
    
    def _get_full_language_name(self, lang_code):
        try:
            language = pycountry.languages.get(alpha_2=lang_code)
            if language:
                return language.name.lower()
            return lang_code
        except Exception:
            return lang_code
    
    def _translate_to_english(self, text):
        # Use a per-thread translator to avoid shared-state/thread-safety issues
        translator = self._get_thread_local_translator()
        translated = translator.translate(text)
        try:
            source_lang_code = detect(text)
            source_lang = self._get_full_language_name(source_lang_code)
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

    def _translate_cell_tuple(self, cell):
        """
        Tuple-returning version for efficient use with ThreadPoolExecutor.map.
        """
        if pd.isna(cell):
            return (cell, None)
        text_str = str(cell)
        if not text_str.strip():
            return (cell, None)
        try:
            translated_text, lang = self._translate_to_english(text_str)
        except Exception:
            translated_text, lang = text_str, None
        return (translated_text, lang)

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
                series_values = translated_df[col].values
                # Parallelize translation per column to preserve order of rows
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(self._translate_cell_tuple, series_values))
                translated_df[col] = [r[0] for r in results]
                translated_df[f"{col}_language"] = [r[1] for r in results]

        return translated_df
