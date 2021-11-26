import os

files = os.listdir()

for item in files:
    if item.endswith(".png") or item.endswith(".gv"):
        os.remove(item)