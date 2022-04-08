import numpy as np

files = ['1', '2', '3', '4']
print('Case:   - Trn Mean - Tst Mean - Trn Std  - Tst Std  - Trn Best - Trn Wrst - Tst Best - Tst Wrst - Bst Idx')
for i in files:
    data_file = open('tests\\data\\DFP\\' + i + '.dat', 'r')
    lines = data_file.readlines()

    data = np.zeros((int(len(lines)/5), 2))
    
    for x in range(len(lines)):
        if x % 5 == 1:
            data[int(x // 5), 0] = lines[x]
        elif x % 5 == 3:
            data[int(x // 5), 1] = lines[x]

    training_min = np.min(data[:, 0])
    testing_min = np.min(data[:, 1])
    training_max = np.max(data[:, 0])
    testing_max = np.max(data[:, 1])

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    best_ind = np.where(data[:, 0] == training_min)[0]

    print('Case: {} - {:.6f} - {:.6f} - {:.6f} - {:.6f} - {:.6f} - {:.6f} - {:.6f} - {:.6f} - {}'.format(
    # print('Case: {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
        i,
        mean[0],
        mean[1],
        std[0],
        std[1],
        training_min,
        training_max,
        testing_min,
        testing_max,
        best_ind
    ))