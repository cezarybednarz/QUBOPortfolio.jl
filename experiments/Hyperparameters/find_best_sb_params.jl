#if !isdefined(Main, :QUBOPortfolio)
include("../../src/QUBOPortfolio.jl")
#end
include("../Plots/plot_execution_data.jl")
include("../Dataset/load_dataset.jl")

using Random

# Set a seed for reproducibility
Random.seed!(1234)

# Load instances
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances()

#! get tuned SIMULATED_BIFURCATION heuristics
tuned_SB_heuristics = QUBOPortfolio.create_set_of_tuned_heuristics_for_type(
    QUBOPortfolio.SIMULATED_BIFURCATION,
    training_instances,
    number_of_tuned_heuristics=2,
    size_of_instance_subsets=20,
    number_of_iterations=75,
    sb_heated=false
)


JLD2.save("cache/best_sb_params_results.jld2", "tuned_sb_heuristics", tuned_SB_heuristics)
# tuned_SB_heuristics = JLD2.load("cache/best_sb_params_results.jld2", "tuned_sb_heuristics")


default_sb_heuristic = QUBOPortfolio.load_all_simulated_bifurcation_heuristics()[1]

#! Test tuned heuristics againt default ones on test set
@info "Running SB heuristics on test set..."
all_heuristics = vcat(tuned_SB_heuristics, [default_sb_heuristic])
all_test_instances = test_instances_mqlib[1:5]
results_mqlib = QUBOPortfolio.run_heuristics_on_dataset(
    all_heuristics,
    all_test_instances; repeat=1, verbosity=true, cache=nothing
)

plot_execution_times(results_mqlib, all_test_instances)
plot_performance(results_mqlib, all_test_instances; baseline_heuristic_name=QUBOPortfolio.load_all_simulated_bifurcation_heuristics()[1].name, filter_heuristics=[])
print_performance_table(results_mqlib)
