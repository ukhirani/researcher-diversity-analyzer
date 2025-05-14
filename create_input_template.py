import pandas as pd

# Create a sample DataFrame with researcher names
df = pd.DataFrame({
    'Researcher Name': [
    # "Yaojie Lu",
    # "Hongming Zhang",
    # "Micheal Abaho",
    # "Piotr Mardziel",
    # "Seung-Hoon Na",
    # "Fangyu Liu",
    # "Yige Xu",
    # "Yue Yu",
    # "Liang Yao",
    # "Divyansh Kaushik",
    # "Kang Liu",
    # "Feng Nie",
    # "Yawei Sun",
    # "Prashanth Vijayaraghavan",
    "Nan Jiang",
    "Florian Kunneman",
    "Manasi Patwardhan",
    
]
})

# Save to Excel file
df.to_excel('researchers.xlsx', index=False)
print("Created researchers.xlsx template with the provided researcher names.") 