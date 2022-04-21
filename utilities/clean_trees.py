import os

method = 'DFP'
exp = '3'

folder_path = 'temp/BOXN/' + method + '/' + exp + '/'
trees_path = folder_path + 'trees/'

files = os.listdir(trees_path)

for item in files:
    if item.endswith(".gv"):
        os.remove(trees_path + item)
        # print(item)

suffex = '.gv.pdf'

files = os.listdir(trees_path)
for item in files:
    num = int(item.replace(suffex, ''))
    formated_num = str("{0:0=2d}".format(num))
    os.rename(trees_path + item, trees_path + formated_num + suffex)

