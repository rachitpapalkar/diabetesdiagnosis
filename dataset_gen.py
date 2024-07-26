import pandas as pd
import numpy as np
from sdv.lite import SingleTablePreset
from sdv.metadata import SingleTableMetadata

datab = pd.read_excel('diabetes2.xlsx')
datab.head()

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=datab)
metadata.visualize()
synthesizer = SingleTablePreset(metadata, name='FAST_ML')
synthesizer.fit(data=datab)
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.head()
synthetic_data.to_csv('file1.csv')
