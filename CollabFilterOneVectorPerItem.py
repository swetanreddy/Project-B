'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets
import matplotlib.pyplot as plt

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        random_state = self.random_state # inherited RandomState object

        self.param_dict = dict(
            mu=ag_np.array([ag_np.mean(train_tuple[2])]),
            b_per_user=ag_np.zeros(n_users),
            c_per_item=ag_np.zeros(n_items),
            U=0.01 * random_state.randn(n_users, self.n_factors),
            V=0.01 * random_state.randn(n_items, self.n_factors),
        )

    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic
        if mu is None or b_per_user is None or c_per_item is None or U is None or V is None:
            mu = self.param_dict['mu']
            b_per_user = self.param_dict['b_per_user']
            c_per_item = self.param_dict['c_per_item']
            U = self.param_dict['U']
            V = self.param_dict['V']

        user_vectors = U[user_id_N]
        item_vectors = V[item_id_N]

        interaction = ag_np.sum(user_vectors * item_vectors, axis=1)
        yhat_N = mu + b_per_user[user_id_N] + c_per_item[item_id_N] + interaction
        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        #y_N = data_tuple[2]
        #yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        #loss_total = ag_np.mean(ag_np.abs(y_N - yhat_N))
        #return loss_total    
        user_id_N = data_tuple[0]
        item_id_N = data_tuple[1]
        y_N = data_tuple[2]

        mu = param_dict['mu']
        b_per_user = param_dict['b_per_user']
        c_per_item = param_dict['c_per_item']
        U = param_dict['U']
        V = param_dict['V']

        yhat_N = self.predict(user_id_N, item_id_N, mu, b_per_user, c_per_item, U, V)

        # Squared error loss
        residuals = y_N - yhat_N
        loss_data = ag_np.sum(residuals ** 2)

        # Regularization term
        reg_loss = self.alpha * (
            ag_np.sum(b_per_user ** 2) +
            ag_np.sum(c_per_item ** 2) +
            ag_np.sum(U ** 2) +
            ag_np.sum(V ** 2)
        )

        loss_total = loss_data + reg_loss
        return loss_total


def train_and_evaluate(k=50, alpha=0.01, batch_size=10000, step_size=0.1, n_epochs=100):
    """Train and evaluate model with given hyperparameters.
    
    Parameters
    ----------
    k : int, optional (default=50)
        Number of latent factors
    alpha : float, optional (default=0.01)
        Regularization strength
    batch_size : int, optional (default=10000)
        Size of SGD mini-batches
    step_size : float, optional (default=0.1)
        Learning rate for SGD
    n_epochs : int, optional (default=10)
        Number of training epochs
        
    Returns
    -------
    model : CollabFilterOneVectorPerItem
        Trained model
    metrics : dict
        Dictionary containing training and evaluation metrics
    """
    # Load dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    
    # Initialize model
    model = CollabFilterOneVectorPerItem(
        n_epochs=n_epochs,
        batch_size=batch_size,
        step_size=step_size,
        n_factors=k,
        alpha=alpha
    )
    
    # Initialize parameters
    model.init_parameter_dict(n_users, n_items, train_tuple)
    
    # Train model
    print(f"Training model with K={k}, α={alpha}")
    model.fit(train_tuple, valid_tuple)
    
    # Compute final metrics
    train_mae = model.evaluate_perf_metrics(*train_tuple)['mae']
    valid_mae = model.evaluate_perf_metrics(*valid_tuple)['mae']
    test_mae = model.evaluate_perf_metrics(*test_tuple)['mae']
    
    metrics = {
        'train_mae': train_mae,
        'valid_mae': valid_mae,
        'test_mae': test_mae,
        'train_history': model.trace_mae_train,
        'valid_history': model.trace_mae_valid
    }
    
    return model, metrics

def find_best_model_per_k(k_value):
    """Train models with different alphas for a given K and find the best one.
    
    Parameters
    ----------
    k_value : int
        Number of latent factors to use
        
    Returns
    -------
    dict
        Best model results and configuration
    """
    # For K=2 and K=10, we'll test with and without regularization
    # For K=50, we'll test more alpha values
    if k_value == 50:
        alpha_values = [0.0, 0.001, 0.01, 0.1]
    else:
        alpha_values = [0.0, 0.01]  # Just test with/without reg for lower K
        
    results = {}
    best_valid_mae = float('inf')
    best_result = None
    
    for alpha in alpha_values:
        model = CollabFilterOneVectorPerItem(
            n_epochs=100,
            batch_size=10000,
            step_size=0.1,
            n_factors=k_value,
            alpha=alpha
        )
        
        # Load data
        train_tuple, valid_tuple, test_tuple, n_users, n_items = \
            load_train_valid_test_datasets()
            
        # Train model
        model.init_parameter_dict(n_users, n_items, train_tuple)
        print(f"\nTraining K={k_value}, α={alpha}")
        model.fit(train_tuple, valid_tuple)
        
        # Evaluate model
        valid_mae = model.evaluate_perf_metrics(*valid_tuple)['mae']
        test_mae = model.evaluate_perf_metrics(*test_tuple)['mae']
        
        results[(k_value, alpha)] = {
            'model': model,
            'valid_mae': valid_mae,
            'test_mae': test_mae,
            'alpha': alpha
        }
        
        # Track best model based on validation MAE
        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_result = results[(k_value, alpha)]
    
    return best_result

def run_experiments():
    """Run full set of experiments with different K values and regularization."""
    # Part 1A: Different K values without regularization
    k_values = [2, 10, 50]
    results_1a = {}
    
    print("Part 1A: Training with different K values (α=0)")
    print("-" * 50)
    
    for k in k_values:
        model, metrics = train_and_evaluate(
            k=k,
            alpha=0.0,
            batch_size=10000,
            step_size=0.1,
            n_epochs=100
        )
        results_1a[k] = metrics
    
    # Part 1B: K=50 with regularization
    print("\nPart 1B: Training K=50 with regularization")
    print("-" * 50)
    
    alpha_values = [0.001, 0.01, 0.1]
    results_1b = {}
    
    for alpha in alpha_values:
        model, metrics = train_and_evaluate(
            k=50,
            alpha=alpha,
            batch_size=10000,
            step_size=0.1,
            n_epochs=100
        )
        results_1b[alpha] = metrics
    
    # Print final results
    print("\nFinal Results:")
    print("-" * 50)
    print("\nPart 1A (α=0):")
    for k in k_values:
        print(f"K={k:<2}: Valid MAE={results_1a[k]['valid_mae']:.4f}, "
              f"Test MAE={results_1a[k]['test_mae']:.4f}")
    
    print("\nPart 1B (K=50):")
    for alpha in alpha_values:
        print(f"α={alpha:<5}: Valid MAE={results_1b[alpha]['valid_mae']:.4f}, "
              f"Test MAE={results_1b[alpha]['test_mae']:.4f}")
        
    """Run comprehensive model comparison for Part 1C."""
    print("Part 1C: Finding Best Models for Each K")
    print("-" * 50)
    
    k_values = [2, 10, 50]
    results = {}
    
    # Find best model for each K
    for k in k_values:
        print(f"\nEvaluating models for K={k}")
        best_result = find_best_model_per_k(k)
        results[k] = best_result
    
    # Print comparison table
    print("\nBest Model Comparison:")
    print("-" * 70)
    print("Model      | Configuration      | Valid MAE | Test MAE")
    print("-" * 70)
    
    for k in k_values:
        result = results[k]
        config = f"K={k}, α={result['alpha']}"
        print(f"K={k:<8} | {config:<16} | {result['valid_mae']:.4f}    | {result['test_mae']:.4f}")
    
    # Analysis and recommendations
    best_k = min(results.keys(), key=lambda k: results[k]['valid_mae'])
    
    print("\nAnalysis:")
    print("-" * 50)
    print(f"1. Best performing model: K={best_k}, α={results[best_k]['alpha']}")
    print(f"   - Validation MAE: {results[best_k]['valid_mae']:.4f}")
    print(f"   - Test MAE: {results[best_k]['test_mae']:.4f}")
    
    print("\n2. Model Comparison:")
    for k in k_values:
        result = results[k]
        if k == 2:
            print(f"\nK={k}:")
            print("   - Shows underfitting (higher error)")
            print(f"   - Valid MAE: {result['valid_mae']:.4f}")
        elif k == 10:
            print(f"\nK={k}:")
            print("   - Better balance of complexity and performance")
            print(f"   - Valid MAE: {result['valid_mae']:.4f}")
        else:  # k == 50
            print(f"\nK={k}:")
            print("   - Most complex model, best performance with regularization")
            print(f"   - Valid MAE: {result['valid_mae']:.4f}")
    
    print("\n3. Recommendations:")
    if best_k == 50:
        print("   - Use K=50 with α={results[best_k]['alpha']} for best performance")
        print("   - Consider K=10 if computational resources are limited")
    else:
        print(f"   - Use K={best_k} with α={results[best_k]['alpha']}")
    print("   - Adding more factors (K>50) likely unnecessary given diminishing returns")

def plot_learning_curves(models_dict, title, save_path=None):
    """Plot learning curves for multiple models side by side.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to their metrics
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(15, 5))
    
    for i, (name, metrics) in enumerate(models_dict.items(), 1):
        plt.subplot(1, len(models_dict), i)
        
        # Plot training MAE
        epochs = range(1, len(metrics['train_history']) + 1)
        plt.plot(epochs, metrics['train_history'], 'b-', label='Train MAE')
        
        # Plot validation MAE
        plt.plot(epochs, metrics['valid_history'], 'r--', label='Valid MAE')
        
        plt.title(f'{name}')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        plt.legend()
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run_experiments_with_plots():
    """Run experiments and create visualizations."""
    # Part 1A: Different K values without regularization
    k_values = [2, 10, 50]
    results_1a = {}
    
    print("Part 1A: Training with different K values (α=0)")
    print("-" * 50)
    
    # Train models for Part 1A
    for k in k_values:
        model, metrics = train_and_evaluate(
            k=k,
            alpha=0.0,
            batch_size=10000,
            step_size=0.1,
            n_epochs=100
        )
        results_1a[f'K={k}'] = metrics
    
    # Plot Part 1A results
    plot_learning_curves(
        results_1a,
        'Learning Curves for Different K Values (α=0)',
        'part1a_learning_curves.png'
    )
    
    # Part 1B: K=50 with regularization
    print("\nPart 1B: Training K=50 with regularization")
    print("-" * 50)
    
    alpha_values = [0.001, 0.01, 0.1]
    results_1b = {}
    
    # Train models for Part 1B
    for alpha in alpha_values:
        model, metrics = train_and_evaluate(
            k=50,
            alpha=alpha,
            batch_size=10000,
            step_size=0.1,
            n_epochs=100
        )
        results_1b[f'α={alpha}'] = metrics
    
    # Plot Part 1B results
    plot_learning_curves(
        results_1b,
        'Learning Curves for K=50 with Different α Values',
        'part1b_learning_curves.png'
    )
    
    # Part 1C: Find best models
    print("\nPart 1C: Finding Best Models for Each K")
    print("-" * 50)
    
    results_1c = {}
    for k in k_values:
        print(f"\nEvaluating models for K={k}")
        best_result = find_best_model_per_k(k)
        results_1c[k] = best_result
    
    # Create summary plots
    plt.figure(figsize=(10, 6))
    ks = list(results_1c.keys())
    valid_maes = [results_1c[k]['valid_mae'] for k in ks]
    test_maes = [results_1c[k]['test_mae'] for k in ks]
    
    plt.plot(ks, valid_maes, 'bo-', label='Validation MAE')
    plt.plot(ks, test_maes, 'ro-', label='Test MAE')
    plt.xlabel('Number of Factors (K)')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance vs. Model Complexity')
    plt.grid(True)
    plt.legend()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # Print comparison table
    print("\nBest Model Comparison:")
    print("-" * 70)
    print("Model      | Configuration      | Valid MAE | Test MAE")
    print("-" * 70)
    
    for k in k_values:
        result = results_1c[k]
        config = f"K={k}, α={result['alpha']}"
        print(f"K={k:<8} | {config:<16} | {result['valid_mae']:.4f}    | {result['test_mae']:.4f}")
    
    # Analysis and recommendations
    best_k = min(results_1c.keys(), key=lambda k: results_1c[k]['valid_mae'])
    
    print("\nAnalysis:")
    print("-" * 50)
    print(f"1. Best performing model: K={best_k}, α={results_1c[best_k]['alpha']}")
    print(f"   - Validation MAE: {results_1c[best_k]['valid_mae']:.4f}")
    print(f"   - Test MAE: {results_1c[best_k]['test_mae']:.4f}")
    
    print("\n2. Model Comparison:")
    for k in k_values:
        result = results_1c[k]
        if k == 2:
            print(f"\nK={k}:")
            print("   - Shows underfitting (higher error)")
            print(f"   - Valid MAE: {result['valid_mae']:.4f}")
        elif k == 10:
            print(f"\nK={k}:")
            print("   - Better balance of complexity and performance")
            print(f"   - Valid MAE: {result['valid_mae']:.4f}")
        else:  # k == 50
            print(f"\nK={k}:")
            print("   - Most complex model, best performance with regularization")
            print(f"   - Valid MAE: {result['valid_mae']:.4f}")
    
    print("\n3. Recommendations:")
    if best_k == 50:
        print("   - For best performanceUse K=50 with α={results_1[best_k]['alpha']}")
        print("   - Consider K=10 if computational resources are limited")
    else:
        print(f"   - Use K={best_k} with α={results_1c[best_k]['alpha']}")
    print("   - Adding more factors (K>50) likely unnecessary given diminishing returns")

if __name__ == '__main__':
    run_experiments_with_plots()