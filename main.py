import pandas as pd

dframe = pd.read_csv("ner_dataset.csv", encoding = "ISO-8859-1", error_bad_lines=False)

df =  dframe[:5000]
x = df['Sentence #'].dropna().index
print(df[24:53][['Word', 'Tag']])
print('walla')