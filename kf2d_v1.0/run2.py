# importing the package
import pandas as pd

print("Pandas import successful!")

# creating new pandas DataFrame
dataframe = pd.DataFrame(list())

# writing empty DataFrame to the new csv file
dataframe.to_csv('file.csv')
print("File Created Successfully")
