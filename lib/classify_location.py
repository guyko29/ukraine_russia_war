import pandas as pd
import torch
from transformers import pipeline
from huggingface_hub import login
import os
from torch.utils.data import Dataset


class ClassifyLocation:
    def __init__(self, df: pd.DataFrame, token: str, helper_dict: dict = {}):
        login(token=token)
        self.df = df
        self.helper_dict = helper_dict
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        
        print("Loading model... This might take a few minutes.")
        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "dtype": torch.bfloat16,
                "low_cpu_mem_usage": True
            },
            device_map="auto",
        )
        
        self.system_prompt = (
            "Your task is to identify the country associated with a given location. "
            "Output only the name of the country in English. "
            "If not recognized, return: none. No explanations."
        )

        self.counter_in_helper_dict = 0
        self.counter_not_in_helper_dict = 0


    def _classify_single_location(self, location_text):
        if pd.isna(location_text) or str(location_text).strip() == "":
            return "none"
        
        if location_text in self.helper_dict:
            self.counter_in_helper_dict += 1
            return self.helper_dict[location_text]
        
        self.counter_not_in_helper_dict += 1

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(location_text)},
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=self.pipe.tokenizer.eos_token_id
        )
        
        result = outputs[0]["generated_text"][-1]["content"].strip()
        self.helper_dict[location_text] = result
        return result

    def process_dataframe(self, column_name='location'):
        new_df = self.df.copy()
        new_df['classified_country'] = new_df[column_name].apply(self._classify_single_location)
        print(f"Counter in helper dictionary: {self.counter_in_helper_dict}")
        print(f"Counter not in helper dictionary: {self.counter_not_in_helper_dict}")
        return new_df, self.helper_dict

# --- example usage ---
# data = {'location': ['Tel Aviv', 'Київ', '123 Fake Street', 'Paris']}
# df = pd.DataFrame(data)
# huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
# huggingface_token = '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'
# classifier = ClassifyLocation(df, huggingface_token, helper_dict={'Tel Aviv': 'Israel'})
# processed_df, helper_dict = classifier.process_dataframe()
# print(processed_df)
# print(helper_dict)
