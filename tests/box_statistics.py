import numpy as np

methods = ['FP', 'DFP']
# methods = ['FP']
exp = '4'

for method in methods:
    print('-' * 100)
    
    data = np.loadtxt('temp/BOXN/' + method + '/' + exp + '/records.dat')
    print(f'Method: {method}  -  Exp: {exp}  -  Records: {data.shape[0]}')

    train_set = data[:, 0]
    test_set = data[:, 1]
    train_best = train_set.min()
    train_best_idx = np.where(train_set == train_best)[0][0] + 1
    train_worst = train_set.max()
    train_worst_idx = np.where(train_set == train_worst)[0][0] + 1
    test_best = test_set.min()
    test_best_idx = np.where(test_set == test_best)[0][0] + 1
    test_worst = test_set.max()
    test_worst_idx = np.where(test_set == test_worst)[0][0] + 1

    print('Train MSE  : {:.8f} \t - \t Std: {:.8f} \t - \t Best: {:.8f} [{}] \t - \t Worst: {:.8f} [{}]'.format(
        train_set.mean(),
        train_set.std(), 
        train_best,
        train_best_idx,
        train_worst,
        train_worst_idx
        )
    )
    print('Test MSE   : {:.8f} \t - \t Std: {:.8f} \t - \t Best: {:.8f} [{}] \t - \t Worst: {:.8f} [{}]'.format(
        test_set.mean(),
        test_set.std(), 
        test_best,
        test_best_idx,
        test_worst,
        test_worst_idx
        )
    )
