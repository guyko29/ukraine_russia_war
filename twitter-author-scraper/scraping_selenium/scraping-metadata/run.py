import pandas as pd
import re

# Read the Excel file
file_path = "/home/ukraine-russia-war/Desktop/poi_russia/russia_poi.xlsx"
df = pd.read_excel(file_path)

# Function to extract username from URL
def extract_username(value):
    if isinstance(value, str) and re.match(r'https?://', value):
        # Use a more general regex to capture the last part after the last slash
        match = re.search(r'https?://(?:www\.)?[^/]+/([^/?#]+)', value)
        if match:
            return match.group(1)
    # If it's not a URL, return the value as is
    return value

# Apply the function to the 'x' column and create 'user_name'
df['user_name'] = df['link_x'].apply(extract_username)

# Create a new column 'filtered_user_name' with only non-null, non-zero values
df['filtered_user_name'] = df['user_name'].apply(lambda x: x if x not in [None, 0, ""] else None)

# Save the modified DataFrame back to an Excel file
output_path = "/home/ukraine-russia-war/Desktop/poi_russia/russia_poi_cleaned.xlsx"
df.to_excel(output_path, index=False)

print(f"File saved to {output_path}")

