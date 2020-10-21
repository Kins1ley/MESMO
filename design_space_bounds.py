#small design
parameter_bounds_small_design={
    'place_freq': (1150, 1310),
    'place_rcfactor': (1.10, 1.25),
    'place_global_max_density': (0.65, 0.90),
    'max_transition': (0.10, 0.35),
}

#medium design
parameter_bounds_medium_design={
    'place_rcfactor': (1.0, 1.3),
    'flow_effort': 2, # standard, extreme two modes
    'place_global_timing_effort': 2, # medium, high two modes
    'place_global_clock_power_driven': 2, # FALSE, TRUE two modes
    'max_length': (250, 350),
    'max_density': (0.4, 1), # NULL value may be shown in the .csv file. It refers to a default value. You can set 0.7 to represent the NULL.
    'max_capacitance': (0.05, 0.15),
    'max_fanout': (25, 40),
    'max_allowed_delay': (0.05, 0.12),
}

#print(parameter_bounds_small_design['place_freq'][1])