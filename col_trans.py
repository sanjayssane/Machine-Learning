import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer 
from sklearn.compose import make_column_selector
df = pd.read_csv("Housing.csv")

ohc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

df_ohc = ohc.fit_transform(df)

ct = make_column_transformer((ohc,
   make_column_selector(dtype_include=object)),
   ("passthrough",
    make_column_selector(dtype_include=['int64',
                                        'float64'])), 
   verbose_feature_names_out=False).set_output(transform='pandas')
df_transf = ct.fit_transform(df)

