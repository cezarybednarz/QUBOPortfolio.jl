include("../../src/QUBOPortfolio.jl")

# This script compares the performance of an XGBoost-based algorithm portfolio
# trained on MQLib heuristics against the individual MQLib heuristics on a test
# dataset. (also trains a Random Forest-based portfolio for comparison)


include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")
import JLD2

# Prepare heuristics and instances
heuristics = QUBOPortfolio.load_8_best_mqlib_heuristics()
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(false)
training_instances = training_instances[1:400]
test_instances_mqlib = test_instances_mqlib[1:100]
test_instances_qubolib = test_instances_qubolib[1:100]

# Number of repetitions for each heuristic-instance pair
repeat_rate = 3

# How to combine the execution energies into one target value
target_function = QUBOPortfolio.HIGHEST_MEAN

# Use cached execution data from previous runs
result_cache = QUBOPortfolio.ResultCache()

# Execute heuristics on training instances
training_results = QUBOPortfolio.run_heuristics_on_dataset(heuristics, training_instances; repeat=repeat_rate, verbosity=true, cache=result_cache)

# Classify the instances with cached metrics
metrics_cache = QUBOPortfolio.load_metrics_from_cache()
classified_instances = QUBOPortfolio.classify_dataset_with_cache(training_instances, metrics_cache)

# Create training data and train the XGBoost portfolio
selector_training_data = QUBOPortfolio.create_selector_training_data(
    heuristics,
    classified_instances,
    training_results;
    target_function=target_function,
    repeat=repeat_rate
)
xgboost_portfolio = QUBOPortfolio.create_portfolio(heuristics)
QUBOPortfolio.train!(xgboost_portfolio, selector_training_data, target_function, :xgboost)

random_forest_portfolio = QUBOPortfolio.create_portfolio(heuristics)
QUBOPortfolio.train!(random_forest_portfolio, selector_training_data, target_function, :random_forest)

# Save the trained portfolio to cache
# portfolio_cache_name = "cache/portfolios/vanilla_MQLib_xgboost_portfolio.jld2"
# JLD2.save(portfolio_cache_name, "portfolio", portfolio)

# Create a Random Forest portfolio heuristic
random_forest_portfolio_heuristic = QUBOPortfolio.Heuristic(
    type=QUBOPortfolio.QUBOPORTFOLIO,
    name="MQLib Portfolio Random Forest top_k=1",
    portfolio=random_forest_portfolio,
    hyperparameters=Dict("top_k" => 1)
)


# Create a XGBoost portfolio heuristic
xgboost_portfolio_heuristic = QUBOPortfolio.Heuristic(
    type=QUBOPortfolio.QUBOPORTFOLIO,
    name="MQLib Portfolio XGBoost top_k=1",
    portfolio=xgboost_portfolio,
    hyperparameters=Dict("top_k" => 1)
)

all_heuristics = vcat(heuristics, [xgboost_portfolio_heuristic, random_forest_portfolio_heuristic])

#! MQLIb
# Run the portfolio (as well as heuristics it was created with) on the test instances
test_results_mqlib = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    test_instances_mqlib;
    repeat=1,
    verbosity=true,
    cache=result_cache,
    readonly_cache=true,
    max_concurrent_tasks=2
)

# Plot the results
print_performance_table(test_results_mqlib)
plot_histogram_number_of_heuristics(test_results_mqlib, random_forest_portfolio_heuristic.name; skip_portfolio=false, filename="plots/MQLib_RandomForest_Portfolio_heuristic_usage_histogram.png")
plot_histogram_number_of_heuristics(test_results_mqlib, xgboost_portfolio_heuristic.name; skip_portfolio=false, filename="plots/MQLib_XGBoost_Portfolio_heuristic_usage_histogram.png")


#! QUBOLib
# Run the portfolio (as well as heuristics it was created with) on the test instances
test_results_qubolib = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    test_instances_qubolib;
    repeat=1,
    verbosity=true,
    cache=result_cache,
    readonly_cache=true,
    max_concurrent_tasks=2
)

# Plot the results
print_performance_table(test_results_qubolib)
plot_histogram_number_of_heuristics(test_results_qubolib, random_forest_portfolio_heuristic.name; skip_portfolio=false, filename="plots/QUBOLib_RandomForest_Portfolio_heuristic_usage_histogram.png")
plot_histogram_number_of_heuristics(test_results_qubolib, xgboost_portfolio_heuristic.name; skip_portfolio=false, filename="plots/QUBOLib_XGBoost_Portfolio_heuristic_usage_histogram.png")
