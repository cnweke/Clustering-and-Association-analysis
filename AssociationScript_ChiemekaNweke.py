import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_excel("Online Retail.xlsx")

df.head()

# start preprocessing data by removing null values if any

df.isnull().sum()

#Drop rows with no invoice numebr
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')

#Drop transactions that are cancelled, they are denoted by Invoice number starting with C
df = df[-df['InvoiceNo'].str.contains('C')]

df.Country.unique()
#drop rows with "Unspecified" country
df = df[df.Country != 'Unspecified']
df.Country.unique()

df.shape

# Splitting the data according to the region of transaction
# Transactions done in any country of dataset
country = "France" #change country to see association of other countries if any
basket_country = (df[df['Country'] ==country]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1
# Applying one hot encoding to find market trends in data, 1 means customer bought item, 0 means did not buy

basket_encoded = basket_country.applymap(hot_encode)
basket_country = basket_encoded

basket_country.head()

# Building the model
frq_items = apriori(basket_country, min_support=0.1, use_colnames=True)

# Collecting the inferred rules using association module and sorting them
ass_rule = association_rules(frq_items, metric="lift", min_threshold=1)
ass_rule = ass_rule.sort_values(['confidence', 'lift'], ascending=[False, False])

pd.set_option("display.max_rows",None,"display.max_columns", None)

print(ass_rule.head())