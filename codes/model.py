import pandas as pd
import os
from numpy import asarray

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

X = pd.read_excel("emotions_df.xlsx")
y = pd.read_excel("ocean_df.xlsx",sheet_name='ocean')

from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense


def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer="he_uniform", activation="relu"))
    model.add(Dense(n_outputs))
    model.compile(loss="mae", optimizer="adam")
    return model


def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    model = get_model(n_inputs, n_outputs)
    model.fit(X, y, verbose=0, epochs=500)
    mae = model.evaluate(X, y, verbose=1)
    print(">%.3f" % mae)
    results.append(mae)
    return results, model


results, model = evaluate_model(X, y)
print("MAE: %.3f (%.3f)" % (mean(results), std(results)))

outputs = []
for i, row in X.iterrows():
    output = model.predict(asarray([list(row)]))
    outputs.append(output[0])
pd.DataFrame(outputs, columns=y.columns).to_excel("output.xlsx", index=False)
