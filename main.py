from JYan import Model

import pandas as pd

data = pd.read_csv("ds_test.csv")


model = Model()

model.train(data)
print("Is model trained? " + str(model.is_fitted))

x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")
#print(x_test)

print(model.predict(x_test))

model.evaluate(x_test, y_test)

