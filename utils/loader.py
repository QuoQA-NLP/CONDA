
import pandas as pd
from datasets import Dataset

def load_dataset(path) :
    df = pd.read_csv(path)
    df = df.dropna()

    dataset = Dataset.from_pandas(df)
    return dataset