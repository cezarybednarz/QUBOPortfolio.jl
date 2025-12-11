include("../../src/QUBOPortfolio.jl")
include("../Dataset/load_dataset.jl")
include("../Plots/plot_execution_data.jl")
import JLD2

training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(true)
tuned_heuristics = JLD2.load("cache/best_sb_params_results.jld2", "tuned_sb_heuristics")
default_heuristic = QUBOPortfolio.load_default_sb_heuristic()

all_instances = vcat(training_instances, test_instances_mqlib, test_instances_qubolib)

cache = QUBOPortfolio.ResultCache()
training_results = QUBOPortfolio.run_heuristics_on_dataset(
    vcat(tuned_heuristics, [default_heuristic]),
    all_instances;
    repeat=3, verbosity=true, cache=cache
)

# plot_execution_times(training_results, all_instances)
# plot_performance(training_results, all_instances; baseline_heuristic_name=default_heuristic.name, filter_heuristics=[])
print_performance_table(training_results)
