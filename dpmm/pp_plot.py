from numpy import arange, array, empty_like, searchsorted, sort
from pylab import plot, show, savefig, title


def cdf(data):
    """
    Returns the empirical CDF (a function) for the specified data.

    Arguments:

    data -- data from which to compute the CDF
    """

    tmp = empty_like(data)
    tmp[:] = data
    tmp.sort()

    def f(x):
        return searchsorted(tmp, x, 'right') / float(len(tmp))

    return f


def pp_plot(a, b, savefile=None, plot_title=None):
    """
    Generates a P-P plot.
    """

    x = sort(a)

    if len(x) > 10000:
        step = len(x) / 5000
        x = x[::step]

    plot(cdf(a)(x), cdf(b)(x), alpha=0.5)
    plot([0, 1], [0, 1], ':', c='k', lw=2, alpha=0.5)

    if plot_title is not None:
        title(plot_title)
    if savefile is None:
        show()
    else:
        if plot_title is not None:
            title(plot_title)
        savefig(savefile, dpi=None, facecolor='w', edgecolor='w',
                                  orientation='portrait', papertype=None, format=None,
                                  transparent=False, bbox_inches="tight", pad_inches=0.1,
                                  frameon=None)        
        

def test(num_samples=100000):

    from numpy.random import normal

    a = normal(20.0, 5.0, num_samples)
    b = normal(20.0, 5.0, num_samples)

    pp_plot(a, b)


if __name__ == '__main__':
    test()
