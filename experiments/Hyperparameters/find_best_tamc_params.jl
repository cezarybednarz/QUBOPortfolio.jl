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

#! get tuned TAMC heuristics
tuned_TAMC_heuristics = QUBOPortfolio.create_set_of_tuned_heuristics_for_type(
    QUBOPortfolio.TAMC,
    training_instances,
    number_of_tuned_heuristics=1,
    size_of_instance_subsets=1,
    number_of_iterations=1,
    tamc_icm=true,
    time_limit=100.0
)

@info "Tuned TAMC heuristics: $tuned_TAMC_heuristics"
JLD2.save("cache/best_tamc_params_results_100s.jld2", "tuned_tamc_heuristics", tuned_TAMC_heuristics)
# tuned_TAMC_heuristics = JLD2.load("cache/best_tamc_params_results.jld2", "tuned_tamc_heuristics")

default_tamc_heuristic = QUBOPortfolio.load_default_tamc_heuristic()

#! Test tuned heuristics againt default ones on test set
@info "Running TAMC heuristics on test set..."
all_heuristics = vcat(tuned_TAMC_heuristics, [default_tamc_heuristic])
all_test_instances = test_instances_mqlib[1:10] # TODO decrease the limit for testing
results_mqlib = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    all_test_instances; repeat=1, verbosity=true, cache=nothing
)

plot_execution_times(results_mqlib, all_test_instances)
plot_performance(results_mqlib, all_test_instances; baseline_heuristic_name=default_tamc_heuristic.name, filter_heuristics=[])
# print_performance_table(results_mqlib)
plot_histogram_execution_times(results_mqlib, tuned_TAMC_heuristics[1].name)
plot_histogram_execution_times(results_mqlib, tuned_TAMC_heuristics[2].name)
