
if __name__ == "__main__":
  # import pandas
  import pandas as pd
  # read the metadata csv file into a pandas dataframe
  df = pd.read_csv('HAM10000_metadata.csv')
  # create a column for target variable from dx column
  # replace 'akiec', 'bkl', 'df', 'nv', and 'vasc' with numerical zero meaning benign
  # replace 'bcc', and 'mel' with numerical one meaning malignant
  df['target'] = df['dx']
  df['target'] = df['target'].replace(['akiec', 'bkl', 'df', 'nv', 'vasc'], 0)
  df['target'] = df['target'].replace(['bcc', 'mel'], 1)
  df.target.nunique()
  df['target'].value_counts()
  # drop the dx_type column. It is not needed
  df = df.drop('dx_type', axis=1)
  # save the new dataframe to a csv file
  df.to_csv("train.csv", index=False)
  
  # the data is skewed; more than 80% cases are benign
  # create training folds to increase randomness of the training set
  from sklearn.model_selection import StratifiedKFold
  
  # Create a new column 'kfold' and initialize it with -1
  df["kfold"] = -1
  
  # Randomize the training data
  df = df.sample(frac=1).reset_index(drop=True)
  
  # Get the target variable from the DataFrame
  y = df.target.values
  
  # Create an instance of StratifiedKFold
  kf = StratifiedKFold(n_splits=5)
  
  # Assign the fold index to the 'kfold' column for each row
  for fold_, (_, _) in enumerate(kf.split(X=df, y=y)):
    df.loc[:, "kfold"] = fold_
    
  # Save the training data with fold indices to a CSV file
  df.to_csv('train_folds.csv', index=False)
