
include("../../src/QUBOPortfolio.jl")
include("../Portfolios/create_portfolios.jl")
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")
include("../Plots/plot_random_forest.jl")
import JLD2

# Heuristics
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(false)

all_mqlib_heuristics = QUBOPortfolio.load_8_best_mqlib_heuristics()

cache = QUBOPortfolio.ResultCache()

all_heuristics_results = QUBOPortfolio.run_heuristics_on_dataset(
    all_mqlib_heuristics,
    test_instances_mqlib;
    repeat=1,
    verbosity=true,
    cache=cache,
    readonly_cache=true
)

# Load MQLib based portfolio heuristic (Random Forest)
mqlib_portfolio_heuristic_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/MQLib_RandomForest_Portfolio_test_results.jld2")

# Load MQLib based portfolio heuristic (XGBoost)
mqlib_portfolio_heuristic_xgb_results = QUBOPortfolio.load_results_from_file("cache/ExecutionResults/MQLib_XGBoost_Portfolio_test_results.jld2")

all_results = QUBOPortfolio.join_execution_results(vcat([mqlib_portfolio_heuristic_results, mqlib_portfolio_heuristic_xgb_results, all_heuristics_results]))

include("../Plots/plot_execution_data.jl")
plot_histogram_number_of_heuristics(all_results, "MQLib_RandomForest_Portfolio"; skip_portfolio=false, filename="plots/MQLib_RandomForest_Portfolio_heuristic_usage_histogram.png")
plot_histogram_number_of_heuristics(all_results, "MQLib_XGBoost_Portfolio"; skip_portfolio=false, filename="plots/MQLib_XGBoost_Portfolio_heuristic_usage_histogram.png")

print_performance_table(all_results)
