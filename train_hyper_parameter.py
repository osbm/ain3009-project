import hyperopt

from train_model import train_model

# Define the search space
space = {
    'batch_size': hyperopt.hp.choice('batch_size', [16, 32, 64]),
    'num_epochs': hyperopt.hp.choice('num_epochs', [2, 3, 4]),
    'color_jitter': hyperopt.hp.choice('color_jitter', [0.1, 0.2, 0.3]),
}

# Define the objective function
def objective(params):
    return train_model(**params)

# Run the hyperparameter search
best = hyperopt.fmin(
    objective,
    space=space,
    algo=hyperopt.tpe.suggest,
    max_evals=10
)

print(best)