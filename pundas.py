import pandas as pd

# exercise 1

euro = pd.read_csv("Euro12.csv")
#
# print(euro.head(1))
# print(euro.tail(1))
# print(euro["Goals"])
# print("number of teams: ", euro["Team"].value_counts().sum())
# print("number of columns: ", euro.columns.value_counts().sum())
# discipline = euro[["Team", "Yellow Cards", "Red Cards"]]
# print(discipline)
# print(discipline.sort_values(by=["Red Cards", "Yellow Cards"]))
# print(euro.describe())
# print("mean of yellow cards: ", euro["Yellow Cards"].mean())
# print(euro[euro["Goals"]>6])
# euro["Total Cards"] = euro["Yellow Cards"] + euro["Red Cards"]
# print(euro)
# print(euro[["Team", "Shooting Accuracy"]].sort_values(by=["Shooting Accuracy"]))
# print(euro[euro["Team"] == 'Portugal'])

# exercise 2

auto = pd.read_csv("Automobile_data.csv")
#
# print(auto.head(), auto.tail())
# expensive = auto[auto["price"] == auto["price"].max()]
# print(expensive["company"])
# print(auto[auto["company"] == "toyota"])
# print(auto["company"].value_counts())
# print(auto[["company", "price"]].sort_values(by=["price"]))
# print(auto.describe())
# print(auto[auto["price"] > 10000])
# print(auto[auto["body-style"] == 'sedan'])

print(euro.head(), "\n", euro.dtypes)
print(auto.head(), "\n", auto.dtypes)
print(auto[auto.isnull().any(axis=1)],euro[euro.isnull().any(axis=1)])
print(auto.fillna(10000), euro.fillna(0))
print("Mean of horsepower: ", auto["horsepower"].mean(),
      "\nMinimum of horsepower: ", auto["horsepower"].min(),
      "\nMaximum of horsepower: ", auto["horsepower"].max(),
      "\nMean of price: ", auto["price"].mean(),
      "\nMinimum of price: ", auto["price"].min(),
      "\nMaximum of price: ", auto["price"].max())
print(auto[["company", "body-style"]].value_counts())
print(auto[auto["horsepower"] > 200])
print("Median of car prices: ", auto["price"].median(),
      "\nStandart deviation of car prices: ", auto["price"].std())
print(auto["engine-type", "horsepower"].mean())



