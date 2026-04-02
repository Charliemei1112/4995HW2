from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def write_variable_descriptions(variables: pd.DataFrame, output_path: str) -> None:
    lines = []

    if "name" in variables.columns and "description" in variables.columns:
        for _, row in variables.iterrows():
            name = row["name"]
            description = row["description"]

            if pd.isna(description):
                description = "No description available."

            lines.append(f"{name}:")
            lines.append(str(description).strip())
            lines.append("")
    else:
        # Fallback: store the full table if the expected columns are unavailable.
        lines.append(variables.to_string(index=False))

    Path(output_path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# variable information 
print(bank_marketing.variables) 

# variable information
write_variable_descriptions(bank_marketing.variables, "variable_information.txt")
print("Saved variable descriptions to variable_information.txt")

# store data in csv file, put X,y together
df = pd.concat([X, y], axis=1)
df.to_csv('bank_marketing.csv', index=False)