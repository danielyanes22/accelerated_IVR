from scipy.stats import qmc

num_param_sets = 100

zero_order_bounds = {'k0': [1e-8, 100]}

first_order_bounds = {'k1': [1e-8, 100]}

higuchi_bounds = {'kh': [1e-8, 100]}

kp_bounds = {'kKP': [1e-8, 100], 'n': [1e-8, 10]}

weibull_bounds = {'a': [1e-8, 300], 'b': [1e-8, 5]}

reciprocal_bounds = {'k': [1e-8, 100], 'a': [1e-8, 100]}

models_bounds = {
    'Zero Order': zero_order_bounds,
    'First Order': first_order_bounds,
    'Higuchi': higuchi_bounds,
    'KP': kp_bounds,
    'Weibull': weibull_bounds,
    'Reciprocal': reciprocal_bounds
}

lhs_samples = []

# Generate LHS samples for each model
for model, bounds in models_bounds.items():
    parameter_space = qmc.LatinHypercube(len(bounds), seed=1884)
    sample = parameter_space.random(n=num_param_sets)
    
    # Scale the sample according to the model's bounds
    l_bounds = [bound[0] for bound in bounds.values()]
    u_bounds = [bound[1] for bound in bounds.values()]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    
    lhs_samples.append(sample_scaled)

#parameter sets ordered zero, first, higuchi, kp, weibull, reciprocal


