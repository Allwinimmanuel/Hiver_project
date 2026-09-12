import pandas as pd
df = pd.read_csv('data/gold/amazonhelp_golden_set.csv')
other = df[df['intent'] == 'other_unknown']
sample = other.sample(n=min(30, len(other)), random_state=42)
out = []
for idx, row in sample.iterrows():
    text = str(row['customer_text']).encode('ascii', 'replace').decode('ascii')
    out.append(f"[{row['example_id']}] {text}")
with open('scratch/temp_sample.txt', 'w') as f:
    f.write("\n".join(out))
