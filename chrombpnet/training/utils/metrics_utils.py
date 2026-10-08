
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy.spatial.distance import jensenshannon


#: Logarithm base for every Jensen-Shannon distance this package reports.
#: Base 2 bounds the distance to [0, 1]: 0 for identical profiles, 1 for
#: profiles with disjoint support. (The original package used scipy's default,
#: the natural log, which bounds it by sqrt(ln 2) ~ 0.833; a base-2 value is the
#: base-e one divided by sqrt(ln 2).)
JSD_BASE = 2


def configure_plot_style():
    """Apply the shared figure size and font used by every metrics plot.

    metrics.py set the same two rcParams again at import, so which one won
    depended on import order -- and metrics_utils additionally passed
    ``'family': 'normal'``, which is not a matplotlib font family. That logged
    "findfont: Font family 'normal' not found" eight times per figure and then
    fell back to the default, so dropping it changes nothing but the noise:
    verified by rendering a figure both ways and comparing the PNG bytes.
    """
    plt.rcParams["figure.figsize"] = 10, 5
    matplotlib.rc('font', weight='bold', size=10)


configure_plot_style()


def normalize_profile(profile):
    """Return a finite probability vector, or None for an empty profile."""
    profile = np.asarray(profile, dtype=np.float64)
    total = np.nansum(profile)
    if not np.isfinite(total) or total <= 0 or not np.all(np.isfinite(profile)):
        return None
    return profile / total
    

def density_scatter(x, y, xlab, ylab, ax = None, sort = True, bins = 20):
    """
    Scatter plot colored by 2d histogram
    """
    # One boolean mask for both. The previous mask concatenated the two
    # np.where tuples, which only worked when x and y had the same number of
    # NaNs; a NaN prediction (labels have none) raised "inhomogeneous shape".
    bad = np.isnan(x) | np.isnan(y)
    x = x[~bad]
    y = y[~bad]

    if ax is None:
        _, ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # ax.figure rather than a local `fig`: that name was only bound in the
    # `ax is None` branch, so passing an explicit ax raised UnboundLocalError.
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return ax
 
#https://github.com/kundajelab/basepairmodels/blob/cf8e346e9df1bad9e55bd459041976b41207e6e5/basepairmodels/cli/metrics.py#L129
def get_min_max_normalized_value(val, minimum, maximum):
    if not np.all(np.isfinite([val, minimum, maximum])) or np.isclose(minimum, maximum):
        return np.nan
    ret_val = (val - maximum) / (minimum - maximum)

    if ret_val < 0:
        return 0
    
    if ret_val > 1:
        return 1
    return ret_val

#https://github.com/kundajelab/basepairmodels/blob/cf8e346e9df1bad9e55bd459041976b41207e6e5/basepairmodels/cli/fastpredict.py#L131
def jsd_min_max_bounds(profile):
    """
        Min Max bounds for the jsd metric
        
        Args:
            profile (numpy.ndarray): true profile 
            
        Returns:
            tuple: (min, max) bounds values
    """
    
    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile_prob = normalize_profile(profile)
    if profile_prob is None:
        return (np.nan, np.nan)

    # jsd of profile with uniform profile
    max_jsd = jensenshannon(profile_prob, uniform_profile, base=JSD_BASE)

    # jsd of profile with itself (upper bound)
    min_jsd = 0.0

    return (min_jsd, max_jsd)
