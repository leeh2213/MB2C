import numpy as np
import matplotlib.pyplot as plt
'''
Cosine similarity of feature pairs of 200 concepts in the test set. The results calculated by the trained models of 10 subjects were averaged,
and all the concepts were rearranged into five categories: animal, food, vehicle, tool, and others.
'''

def confusion_matrix_testdataset():
    y_true_all = []
    y_pred_all = []

    # 打开文件并读取每一行
    with open('pic/output.txt', 'r') as file:
        for line in file:
            sublist = eval(line)
            y_pred_all.extend(sublist)
    y_true_all = np.tile(np.arange(200), 10)

    # draw confusion matrix with y_true and y_pred, normalized
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, cmap='Reds')
    plt.title('Confusion matrix', fontsize=20)
    # scale the colorbar
    plt.colorbar(shrink=0.8)
    x_ticks = [35, 82, 102, 137, 147, 200]
    x_ticklabels = ['animal', 'food', 'vehicle', 'tool', 'sport', 'others']
    plt.xticks(x_ticks, x_ticklabels, fontsize=16)
    plt.xlabel('Predicted', fontsize=20)
    plt.yticks(x_ticks, x_ticklabels, fontsize=16, rotation=90)
    plt.ylabel('True', fontsize=20)
    # plt.tight_layout()
    plt.savefig('pic/confusion_matrix.png', dpi=800)
    print('Down!')
    
    
confusion_matrix_testdataset()
