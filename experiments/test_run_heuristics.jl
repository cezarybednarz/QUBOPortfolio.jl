
include("../src/QUBOPortfolio.jl")
include("Plots/plot_execution_data.jl")
include("Dataset/load_dataset.jl")

using Random

training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances(false)

results = QUBOPortfolio.run_heuristics_on_dataset(
    QUBOPortfolio.load_8_best_mqlib_heuristics(),
    training_instances[1:4];
    repeat=1,
    verbosity=true,
    max_concurrent_tasks=4
)
