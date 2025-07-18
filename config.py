SEED = 123
X_MIN_WA = -125
X_MAX_WA = -117
Y_MIN_WA = 45.5
Y_MAX_WA = 49

# pressure vessel design
THICKNESS_SCALAR = 0.0625
THICKNESS_MIN_INT = 1
THICKNESS_MAX_INT = 99
THICKNESS_MIN = THICKNESS_MIN_INT * THICKNESS_SCALAR
THICKNESS_MAX = THICKNESS_MAX_INT * THICKNESS_SCALAR
RADIUS_MIN = 10
RADIUS_MAX = 100  # arbitrary, but not too high
LENGTH_MIN = 10   # arbitrary, but not too low
LENGTH_MAX = 200
NUM_DESIGN_VARIABLES = 4
PVD_PLOT_Y_MAX = 10000
PVD_PLOT_Y_MIN = 5500

# --------------
# for plotting

Y_AXIS_SA_TSP_GRID = "Tour Distance (units)"
Y_AXIS_SA_TSP_RANDOM = "Tour Distance (Euclidean)"
Y_AXIS_SA_BPP = "Number of Bins Used"
Y_AXIS_SA_PVD = "Design Cost ($)"

Y_AXIS_POP_TSP_GRID = "Best Tour Distance (units)"
Y_AXIS_POP_TSP_RANDOM = "Best Tour Distance (Euclidean)"
Y_AXIS_POP_BPP = "Best Number of Bins Used"
Y_AXIS_POP_PVD = "Best Design Cost ($)"

X_AXIS_SA = "Iteration"
X_AXIS_GA = "Generation"
X_AXIS_PSO = "Iteration"

LEGEND_TSP = "Tour Distance"
LEGEND_BPP = "Bins Used"
LEGEND_PVD = "Design Cost"

COLOR_SA = "indianred"
COLOR_GA = "seagreen"
COLOR_PSO = "slateblue"


# titles

TITLE_SA_TSP = "Tour Distance over SA Iterations"
TITLE_GA_TSP = "Tour Distance over GA Generations"
TITLE_PSO_TSP = "Tour Distance over PSO Iterations"

TITLE_SA_BPP = "Bins Used over SA Iterations"
TITLE_GA_BPP = "Bins Used over GA Generations"
TITLE_PSO_BPP = "Bins Used over PSO Iterations"

TITLE_SA_PVD = "Pressure Vessel Cost over SA Iterations"
TITLE_GA_PVD = "Pressure Vessel Cost over GA Generations"
TITLE_PSO_PVD = "Pressure Vessel Cost over PSO Iterations"


# ==========================================================================
# |  Different combinations of hyperparameters below:
# ==========================================================================

# Distance 20.839
# max_iterations = 8000
# initial_temperature = 40
# cooling_rate = 0.999
# num_cities = 20
# shift_max = 2

# Distance 20.548
# max_iterations = 8000
# initial_temperature = 33
# cooling_rate = 0.999
# num_cities = 20
# shift_max = 2

# Distance 20.839
# max_iterations = 10000
# initial_temperature = 40
# cooling_rate = 0.999
# num_cities = 20
# shift_max = 2

# Distance 21.315
# max_iterations = 10000
# initial_temperature = 46
# cooling_rate = 0.999
# num_cities = 20
# shift_max = 2

# Distance 19.752
# max_iterations = 10000
# initial_temperature = 17
# cooling_rate = 0.999
# num_cities = 20
# shift_max = 2

# Distance 19.780
# max_iterations = 10000
# initial_temperature = 5
# cooling_rate = 0.999
# num_cities = 20
# shift_max = 2

# Distance 19.782
# max_iterations = 10000
# initial_temperature = 2.5
# cooling_rate = 0.999
# num_cities = 20
# shift_max = 2

# ==================


# 25 cities, distance 26.243
# max_iterations = 10000
# initial_temperature = 9
# cooling_rate = 0.999
# num_cities = 20
# shift_max = 10

# 49 cities, distance 56.055
# max_iterations = 20000
# initial_temperature = 12
# cooling_rate = 0.9995
# num_cities = 20
# shift_max = 20

# 64 cities, distance 73.265
# max_iterations = 30000
# initial_temperature = 20
# cooling_rate = 0.9997
# num_cities = 20
# shift_max = 30

# # 64 cities, distance 72.028
# max_iterations = 40000
# initial_temperature = 19
# cooling_rate = 0.9997
# num_cities = 20
# shift_max = 32

# 64 cities, distance 69.136
# max_iterations = 50000
# initial_temperature = 29
# cooling_rate = 0.9998
# num_cities = 20
# shift_max = 32
