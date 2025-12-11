if !isdefined(Main, :QUBOPortfolio)
    include("../../src/QUBOPortfolio.jl")
end
include("load_dataset.jl")
include("../Plots/plot_dataset.jl")

train_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances()

cache = QUBOPortfolio.load_metrics_from_cache("cache/metrics.jld2")
classified_train_instances = QUBOPortfolio.classify_dataset_with_cache(train_instances, cache)
classified_test_instances_qubolib = QUBOPortfolio.classify_dataset_with_cache(test_instances_qubolib, cache)
classified_test_instances_mqlib = QUBOPortfolio.classify_dataset_with_cache(test_instances_mqlib, cache)

# num of variables vs num of terms
plot_density_vs_num_variables(classified_train_instances, "MQLib Training Instances")
plot_density_vs_num_variables(classified_test_instances_qubolib, "QUBOlib Test Instances")
plot_density_vs_num_variables(classified_test_instances_mqlib, "MQLib Test Instances")
