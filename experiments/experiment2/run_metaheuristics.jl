#if !isdefined(Main, :QUBOPortfolio)
include("../../src/QUBOPortfolio.jl")
#end
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")

using Random

# Set a seed for reproducibility
Random.seed!(1234)

# Load instances
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(false)

# Get the heuristics with default parameters
default_brkga_heuristic = QUBOPortfolio.Heuristic(
    type=QUBOPortfolio.METAHEURISTICSJL_BRKGA,
    name="MetaheuristicsJL - BRKGA - default",
    hyperparameters=Dict{String, Any}() # empty dict means default hyperparameters
)
default_pso_heuristic = QUBOPortfolio.Heuristic(
    type=QUBOPortfolio.METAHEURISTICSJL_PSO,
    name="MetaheuristicsJL - PSO - default",
    hyperparameters=Dict{String, Any}() # empty dict means default hyperparameters
)

mqlib_heuristics = QUBOPortfolio.load_8_best_mqlib_heuristics()


cache = QUBOPortfolio.ResultCache()
#! Test tuned heuristics againt default ones on test set
all_heuristics = vcat(default_brkga_heuristic, default_pso_heuristic, mqlib_heuristics)
results_mqlib = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    test_instances_mqlib[1:100]; repeat=1, verbosity=true, cache=cache
)

# plot_execution_times(results_mqlib, test_instances_mqlib)
# plot_performance(results_mqlib, test_instances_mqlib; baseline_heuristic_name="", filter_heuristics=[])
print_performance_table(results_mqlib)
