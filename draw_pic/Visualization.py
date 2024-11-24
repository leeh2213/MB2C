from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

colors_per_class = {
    '0' : [254, 202, 87],
    '1' : [255, 107, 107],
    '2' : [10,  189, 227],
    '3' : [255, 159, 243],
    '4' : [16,  172, 132],
    '5' : [0 ,  0,   0],
    '6' : [0,   139, 139],
    '7' : [139, 0,   0],
    '8':  [20,  126,  14]
}

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zesro
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_tsne(tsne, labels):
    '''
    tsne:(590, 2)
    labels:torch.Size([590])
    '''
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0, 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)


def visualize_tsne_points(tx, ty, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = labels.numpy()

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        
        # find the samples of the current class in the data
        indices = np.where(labels == int(label)) # error

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float64) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()

embedding_list = []   
label_list = []    
tsne = TSNE(n_components=2).fit_transform(embedding_list)
visualize_tsne(tsne, label_list)