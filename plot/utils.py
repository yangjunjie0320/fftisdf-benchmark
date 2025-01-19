import matplotlib as mpl

params = {}
params["font.size"] = 18
params["axes.titlesize"] = 20
params["axes.labelsize"] = 20
params["legend.fontsize"] = 18
params["xtick.labelsize"] = 14
params["ytick.labelsize"] = 14
params["figure.subplot.wspace"] = 0.0
params["figure.subplot.hspace"] = 0.0
params["axes.spines.right"] = True
params["axes.spines.top"] = True
params["xtick.direction"] = 'in'
params["ytick.direction"] = 'in'
params["text.usetex"] = True
params["font.family"] = "serif"
params["text.latex.preamble"] = r"\usepackage{amsmath}"

# add grid
params["axes.grid"] = True
# params["axes.grid.which"] = "both"
params["grid.linestyle"] = "--"
params["grid.linewidth"] = 0.5
params["grid.alpha"] = 0.5
params["grid.color"] = "gray"

# only show grid for pow

mpl.rcParams.update(params)
