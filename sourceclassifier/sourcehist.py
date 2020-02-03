import matplotlib.pyplot as plt

def sourcehist(x, title='Histogram', xlabel='x', ylabel='y',
               **kwargs):
    """
    This function provides a histogram plot given a single array in
    input. Most of the features are inherited from the matplotlib hist
    function.

    :x: an array of values
    :title: the title of the histogram shown in the plot
    :xlabel: x label shown in the plot
    :ylabel: y label shown in the plot
    :kwargs: the same parameters of the plt.hist function
    :return: a histogram plot of values.
    """

    plt.hist(x, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return
