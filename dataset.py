import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure output directories exist
os.makedirs("processed", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Load dataset
data_path = os.path.join("data", "online_retail.xlsx")
df = pd.read_excel(data_path)

# Initial cleanup
df.dropna(subset=['Description'], inplace=True)

# Save cleaned data
df.to_csv(os.path.join("processed", "cleaned_data.csv"), index=False)

# WordCloud generation
desc_text = " ".join(str(desc) for desc in df['Description'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(desc_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Product Descriptions")
plt.tight_layout()
plt.savefig(os.path.join("metrics", "wordcloud_description.png"))
plt.close()

# New feature generation (WITHOUT NLTK)
colours = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'purple', 'pink',
           'silver', 'gold', 'beige', 'brown', 'grey', 'gray', 'black', 'white', 'cream']

Product_type = []
Colour_type = []

dataset = df.head(50000)

for desc in dataset['Description']:
    s = ""
    description = re.sub('[^a-zA-Z]', " ", str(desc).lower())
    words = description.split()

    # Colour extraction
    color_found = next((word for word in words if word in colours), "no_color")
    Colour_type.append(color_found)

    # Simulated Product Type (first 3 significant words)
    product_keywords = [word for word in words if len(word) > 2]
    s = " ".join(product_keywords[:3])
    Product_type.append(s)

dataset['ProductType'] = Product_type
dataset['Colour'] = Colour_type

# Save processed dataset with new features
dataset.to_csv(os.path.join("data", "preprocessed_data.csv"), index=False)

# Optional: print sample
print(dataset[['Description', 'ProductType', 'Colour']].head())
