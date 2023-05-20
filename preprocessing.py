import pandas as pd
df = pd.read_csv('HAM10000_metadata.csv')
df['target'] = df['dx']
df['target'] = df['target'].replace(['akiec', 'bkl', 'df', 'nv', 'vasc'], 0)
df['target'] = df['target'].replace(['bcc', 'mel'], 1)
df.target.nunique()
df['target'].value_counts()
df = df.drop('dx_type', axis=1)
df.to_csv("data_with_target", index=False)

from sklearn.model_selection import train_test_split

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
# Use 20% as test data since we have a large dataset
# Set Random State to 42 to always have similar results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_copy = df.copy()
df_copy["kfold"] = -1
df_copy = df_copy.sample(frac=1).reset_index(drop=True)
kf = model_selection.StratifiedKFold(n_splits=5)
for fold_, (_, _) in enumerate(kf.split(X=X, y=y)):
  X.loc[:, "kfold"] = fold_
df_copy.to_csv('new_df.csv', index=False)