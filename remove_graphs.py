import os

files = os.listdir()

for item in files:
    if item.endswith(".png") or item.endswith(".gv") or item.endswith(".pdf"):
        os.remove(item)
        # print(item)