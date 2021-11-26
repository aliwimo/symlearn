import os

files = os.listdir()

for item in files:
    if item.endswith(".png"):
        os.remove(item)