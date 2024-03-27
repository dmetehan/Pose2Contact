import numpy as np
from sklearn.dummy import DummyClassifier


def get_baseline_predictors(all_labels):
    all_strategies = ["most_frequent", "prior", "stratified", "uniform", "constant"]
    all_clfs = {key: {strategy: DummyClassifier(strategy=strategy,
                                                constant=[1 for _ in range(eval(key.replace('x', '*')))]) for strategy in all_strategies} for key in all_labels}
    for key in all_clfs:
        for strategy in all_clfs[key]:
            if strategy == "constant":
                all_clfs[key][strategy].fit(all_labels[key] + [[1 for _ in range(eval(key.replace('x', '*')))]],
                                            all_labels[key] + [[1 for _ in range(eval(key.replace('x', '*')))]])  # (X, y) but X doesn't matter for the dummy classifiers
            else:
                all_clfs[key][strategy].fit(all_labels[key], all_labels[key])  # (X, y) but X doesn't matter for the dummy classifiers
    return all_clfs


def predict_with_predictors(all_labels, all_clfs, eval_func, **kwargs):
    all_results, best_results = {}, {}
    for key in all_clfs:
        all_results[key], best_results[key] = {}, (0, 0)
        for strategy in all_clfs[key]:
            all_results[key][strategy] = eval_func(all_labels[key], all_clfs[key][strategy].predict(all_labels[key]), **kwargs)
            if all_results[key][strategy] > best_results[key][0]:
                best_results[key] = (all_results[key][strategy], strategy)
    return all_results, best_results
