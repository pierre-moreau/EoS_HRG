import time
from matplotlib.pyplot import rc

########################################################################
def timef(fun):
  """
  decorator to record execution time of a function
  """
  def wrapper(*args,**kwargs):
    start = time.time()
    res = fun(*args,**kwargs)
    end = time.time()
    print(f'Function {fun.__name__} took {end-start}s\n')
    return res
  return wrapper

########################################################################
def str2bool(val):
  """
  convert string to bool (used to pass arguments with argparse)
  """
  dict = {'True': True, 'False': False, 'true': True, 'false': False, '1': True, '0': False}
  return dict[val]

########################################################################
# settings for plots
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
rc('axes', linewidth=3) # width of axes
font = {'family' : 'Arial',
        'size' : MEDIUM_SIZE,
        'weight' : 'bold'}
rc('font', **font)  # controls default text sizes
rc('axes', titlesize=MEDIUM_SIZE, titleweight='bold')     # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE, labelweight='bold')    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE, direction='in', top='True')    # fontsize of the tick labels
rc('xtick.major', size=7, width=3, pad=10)
rc('ytick', labelsize=SMALL_SIZE, direction='in', right='True')    # fontsize of the tick labels
rc('ytick.major', size=7, width=3, pad=10)
rc('legend', fontsize=SMALL_SIZE, title_fontsize=SMALL_SIZE, handletextpad=0.25)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE, titleweight='bold')  # fontsize of the figure title
rc('savefig', dpi=300, bbox='tight')