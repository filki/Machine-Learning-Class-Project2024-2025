import pandas as pd
import time
import os
import litellm
from dotenv import load_dotenv

CSV_DATA_PATH = "../../output/results/lsa_lda/lsa_100_topics_gensim.csv" 
OUTPUT_CSV_PATH = "../../output/results/lsa_lda/named_topics_lsa.csv"
GEMINI_MODEL = "gemini/gemini-2.0-flash-lite"
DELAY_SECONDS = 2.5

load_dotenv(override=True)
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

print(f"Loading data from {CSV_DATA_PATH}...")
df = pd.read_csv(CSV_DATA_PATH, usecols=['topic_id', 'words'])

def get_gemini_name_from_words(top_words_string):
    prompt = f"""
    The top keywords for a topic are: "{top_words_string}"
    Based on these keywords, suggest a concise and descriptive name (3-6 words) for this topic cluster.
    Output ONLY the topic name itself. <topic name>
    """

    response = litellm.completion(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20, temperature=0.95
    )
    name = response.choices[0].message.content.strip().replace('"', '').replace('**', '')
    print(name)
    return name if name else "Name Gen Failed"

# --- Generate Names ---
print("Generating topic names via Gemini")
cluster_name_map = {}

for index, row in df.iterrows():
    topic_id = row['topic_id']
    top_words = row['words']

    if not top_words or pd.isna(top_words):
        print(f"  Skipping topic {topic_id} due to empty top words.")
        cluster_name_map[topic_id] = "Topic with No Words"
    else:
        print(f"  Naming topic {topic_id}")
        cluster_name_map[topic_id] = get_gemini_name_from_words(top_words)
        time.sleep(DELAY_SECONDS)


df['topic_name'] = df['topic_id'].map(cluster_name_map)

print("Topic naming complete.")

print(f"Saving results to {OUTPUT_CSV_PATH}...")
df[['topic_id', 'words', 'topic_name']].to_csv(OUTPUT_CSV_PATH, index=False)
print("Done.")