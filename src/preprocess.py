import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv", index_col=0)
df_train, df_devtest = train_test_split(df, random_state=0)
df_dev, df_test = train_test_split(df_devtest, test_size=0.5, random_state=0)

df_train.to_csv("./train.tsv", sep="\t")
df_dev.to_csv("./dev.tsv", sep="\t")
df_test.to_csv("./test.tsv", sep="\t")
