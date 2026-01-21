from pathlib import Path
import pandas as pd

from lib.translate import Translator
from lib.classify_location import ClassifyLocation
from lib.clean_features import FeatureCleaner


COUNTRY_MAPPING = {
    "Ukraine": "../Ukraine/manual label-1 - ukraine.xlsx",
    "Russia": "../Russia/manual label-1- russia.xlsx",
}

CRITERIA_MAPPING = {
    "local": ["name", "bio", "location"],
    "nationality": ["name", "location", "bio"],
    "private": ["name", "bio", "following", "followers"]
} 

NEED_TRANSLATION = ["name", "bio", "location"]

CLEAN_FEATURES = ["bio"]


class GenerateFeatures:
    def __init__(self, country: str, token: str, helper_dict: dict = {}):
        self.country = country
        project_dir = Path(__file__).resolve().parent
        excel_path = project_dir / COUNTRY_MAPPING[self.country]
        self.df = pd.read_excel(excel_path)

        self.translator = Translator(self.df, NEED_TRANSLATION)
        self.location_classifier = ClassifyLocation(df=self.df, token=token, helper_dict=helper_dict)

    def _filter_out_known_usernames(self) -> pd.DataFrame:
        """
        Return a copy of df excluding rows whose `user_name` exists in all_user_name.xlsx.
        If the file or `user_name` column is missing, returns df unchanged.
        """
        project_dir = Path(__file__).resolve().parent
        allowed_file = project_dir / "all_user_name.xlsx"
        try:
            known_df = pd.read_excel(allowed_file)
        except Exception:
            return self.df
        if "user_name" not in self.df.columns or "user_name" not in known_df.columns:
            return self.df
        known_usernames = set(known_df["user_name"].dropna().astype(str))
        filtered = self.df.copy()
        filtered = filtered[~filtered["user_name"].astype(str).isin(known_usernames)]
        return filtered

    def add_usernames_to_all_user_name(self) -> int:
        """
        Append all user_name values from self.df into all_user_name.xlsx (deduplicated).
        Returns the number of newly added usernames.
        If self.df lacks `user_name`, no changes are made and 0 is returned.
        """
        if "user_name" not in self.df.columns:
            return 0
        project_dir = Path(__file__).resolve().parent
        allowed_file = project_dir / "all_user_name.xlsx"
        try:
            existing_df = pd.read_excel(allowed_file)
        except Exception:
            existing_df = pd.DataFrame({"user_name": []})
        if "user_name" not in existing_df.columns:
            existing_df["user_name"] = pd.Series(dtype=str)

        existing_set = set(existing_df["user_name"].dropna().astype(str))
        current_set = set(self.df["user_name"].dropna().astype(str))
        to_add = sorted(current_set - existing_set)

        if not to_add:
            return 0

        append_df = pd.DataFrame({"user_name": to_add})
        final_df = pd.concat([existing_df[["user_name"]], append_df], ignore_index=True)
        final_df.drop_duplicates(subset=["user_name"], inplace=True)
        final_df.to_excel(allowed_file, index=False)
        return len(to_add)


    def _filter_by_label_criteria(self, criteria: str) -> pd.DataFrame:
        """
        Create and return a new DataFrame containing only the columns
        required for the given labeling criteria. Missing columns are added
        and filled with NA values.
        """
        # copy to avoid mutating the global mapping lists
        required_features = ['user_name'] + list(CRITERIA_MAPPING[criteria])
        list_translation_features = [f"{feature}_language" for feature in NEED_TRANSLATION if feature in required_features]
        
        if 'location' in required_features:
            required_features.append('classified_country')
            required_features.remove('location')
        
        required_features = required_features + list_translation_features

        judge_prefix = criteria
        if criteria == "nationality":
            judge_prefix = "ukraine" if self.country == "Ukraine" else "russian"
        
        judge_suffixes = ["1st judge", "2nd judge", "3rd judge"]
        judge_columns = [f"{judge_prefix}:{suffix}" for suffix in judge_suffixes]
        final_judge_column = f"{judge_prefix}:judge"

        filtered_df = self.df.copy()

        # Build single consolidated judge column:
        # - If judge1 and judge2 agree → that value
        # - Else → value of judge3
        j1, j2, j3 = judge_columns
        def _normalize_value(v):
            if v is None:
                return None
            # Handle pandas NA
            if isinstance(v, float) and pd.isna(v):
                return None
            # Booleans
            if isinstance(v, bool):
                return "true" if v else "false"
            # Numerics
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if pd.isna(v):
                    return None
                if v == 1:
                    return "true"
                if v == 0:
                    return "false"
                return None
            # Strings
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("", "none", "null", "na", "n/a"):
                    return None
                if s in ("true", "t", "yes", "y", "1"):
                    return "true"
                if s in ("false", "f", "no", "n", "0"):
                    return "false"
                if s == "natural":
                    return "natural"
                return None
            return None

        def _consolidate_judges(row):
            v1 = _normalize_value(row.get(j1))
            v2 = _normalize_value(row.get(j2))
            v3 = _normalize_value(row.get(j3))
            if v1 is not None and v1 == v2:
                return v1
            return v3
        filtered_df[final_judge_column] = filtered_df.apply(_consolidate_judges, axis=1)

        # Reindex on columns ensures exact order, adds missing as NA, and drops extras
        filtered_df = filtered_df.reindex(columns=required_features + [final_judge_column])
        return filtered_df
    

    def _create_xlsx_file(self, df: pd.DataFrame, criteria: str):
        project_dir = Path(__file__).resolve().parent
        output_dir = project_dir / "Outputs" / self.country
        output_dir.mkdir(parents=True, exist_ok=True)
        excel_path = output_dir / f"{self.country}_{criteria}.xlsx"
        if excel_path.exists():
            try:
                existing_df = pd.read_excel(excel_path)
            except Exception:
                existing_df = pd.DataFrame()
            final_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            final_df = df
        final_df.to_excel(excel_path, index=False)


    def generate_features(self) -> dict:
        print("Filtering out known users...")
        self.df = self._filter_out_known_usernames()
        if self.df.empty:
            print("No users to process after filtering known usernames. Skipping pipeline.")
            return {}

        print("Cleaning features...")
        # Ensure cleaner uses the current (filtered) DataFrame
        self.feature_cleaner = FeatureCleaner(self.df, CLEAN_FEATURES)
        self.df = self.feature_cleaner.clean()

        print(f"Generating features for {self.country}...")
        # Ensure translator uses the cleaned DataFrame
        self.translator.df = self.df
        self.df = self.translator.translate_required_columns()

        print("Classifying locations...")
        # Ensure classification runs on the translated DataFrame so we keep translations
        self.location_classifier.df = self.df
        self.df, helper_dict = self.location_classifier.process_dataframe()

        print("Filtering by label criteria...")
        df_filtered_local = self._filter_by_label_criteria("local")
        df_filtered_nationality = self._filter_by_label_criteria("nationality")
        df_filtered_private = self._filter_by_label_criteria("private")

        print("Creating xlsx files...")
        self._create_xlsx_file(df_filtered_local, "local")
        self._create_xlsx_file(df_filtered_nationality, "nationality")
        self._create_xlsx_file(df_filtered_private, "private")

        print("Adding usernames to all_user_name.xlsx...")
        self.add_usernames_to_all_user_name()

        return helper_dict
