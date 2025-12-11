# if !isdefined(Main, :QUBOPortfolio)
include("../../src/QUBOPortfolio.jl")
# end
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")

import JLD2

# Prepare heuristics and instances
heuristics = QUBOPortfolio.load_8_best_mqlib_heuristics()
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(false)
training_instances = training_instances[1:20]  # TODO Use only first 20 instances for training
test_instances_mqlib = test_instances_mqlib[1:20]  # TODO Use only first 20 instances for testing

# Number of repetitions for each heuristic-instance pair
repeat_rate = 3

# How to combine the execution energies into one target value
target_function = QUBOPortfolio.HIGHEST_MEAN

# Use cached execution data from previous runs
result_cache = QUBOPortfolio.ResultCache()

# Execute heuristics on training instances
training_results = QUBOPortfolio.run_heuristics_on_dataset(heuristics, training_instances; repeat=repeat_rate, verbosity=true, cache=result_cache)

# Classify the instances with cached metrics
cached_metrics = QUBOPortfolio.load_metrics_from_cache("cache/metrics.jld2")
classified_instances = QUBOPortfolio.classify_dataset_with_cache(training_instances, cached_metrics)

# Create training data and train the portfolio
selector_training_data = QUBOPortfolio.create_selector_training_data(
    heuristics,
    classified_instances,
    training_results;
    target_function=target_function,
    repeat=repeat_rate
)
portfolio = QUBOPortfolio.create_portfolio(heuristics)
QUBOPortfolio.train!(portfolio, selector_training_data, target_function, :random_forest)

# Save the trained portfolio to cache
# portfolio_cache_name = "cache/portfolios/vanilla_MQLib_random_forest_portfolio.jld2"
# JLD2.save(portfolio_cache_name, "portfolio", portfolio)

# Create a portfolio heuristic
portfolio_heuristic = QUBOPortfolio.Heuristic(
    type=QUBOPortfolio.QUBOPORTFOLIO,
    name="Portfolio, Random Forest, 8*MQLib",
    portfolio=portfolio,
    hyperparameters=Dict("top_k" => 1)
)

#! MQLIb
# Run the portfolio (as well as heuristics it was created with) on the test instances
test_results_mqlib = QUBOPortfolio.run_heuristics_on_dataset(
    vcat(portfolio_heuristic, heuristics),
    test_instances_mqlib;
    repeat=1,
    verbosity=true,
    cache=result_cache
)

print_performance_table(test_results_mqlib)

# Plot the results
plot_execution_times(test_results_mqlib, test_instances_mqlib)
plot_performance(test_results_mqlib, test_instances_mqlib, "BURER2002", [portfolio_heuristic.name])
print_performance_table(test_results_mqlib)


# #! QUBOLib
# # Run the portfolio (as well as heuristics it was created with) on the test instances
# test_results_qubolib = QUBOPortfolio.run_heuristics_on_dataset(
#     vcat(portfolio_heuristic, heuristics),
#     test_instances_qubolib;
#     repeat=1,
#     verbosity=true,
#     cache=result_cache
# )

# # Plot the results
# plot_execution_times(test_results_qubolib, test_instances_qubolib)
# plot_performance(test_results_qubolib, test_instances_qubolib, "BURER2002", [portfolio_heuristic.name])
# print_performance_table(test_results_qubolib)
