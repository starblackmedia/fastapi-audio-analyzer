# check_model.py
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
