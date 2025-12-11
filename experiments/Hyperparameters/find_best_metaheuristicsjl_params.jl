#if !isdefined(Main, :QUBOPortfolio)
include("../../src/QUBOPortfolio.jl")
#end
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")

using Random

# The main idea for this experiment:
#  For a heuristic generate 4 different subsets of training data. Tune the hyperparamters to fit the best on each subset.
#  Then we have like 4 heuristics of this type.

# Set a seed for reproducibility
Random.seed!(1234)

# Load instances
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances()

#! get tuned METAHEURISTICSJL_BRKGA heuristics
tuned_BRKGA_heuristics = QUBOPortfolio.create_set_of_tuned_heuristics_for_type(
    QUBOPortfolio.METAHEURISTICSJL_BRKGA,
    training_instances,
    number_of_tuned_heuristics=4,
    size_of_instance_subsets=4,
    number_of_iterations=4
)

#! get tuned METAHEURISTICSJL_PSO heuristics
tuned_PSO_heuristics = QUBOPortfolio.create_set_of_tuned_heuristics_for_type(
    QUBOPortfolio.METAHEURISTICSJL_PSO,
    training_instances,
    number_of_tuned_heuristics=4,
    size_of_instance_subsets=4,
    number_of_iterations=4
)

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

#! Test tuned heuristics againt default ones on test set
all_heuristics = vcat(tuned_BRKGA_heuristics, tuned_PSO_heuristics, [default_brkga_heuristic, default_pso_heuristic])
results_mqlib = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    test_instances_mqlib[1:10]; repeat=1, verbosity=false, cache=nothing
)

plot_execution_times(results_mqlib, test_instances_mqlib)
plot_performance(results_mqlib, test_instances_mqlib; baseline_heuristic_name="", filter_heuristics=[])
print_performance_table(results_mqlib)
