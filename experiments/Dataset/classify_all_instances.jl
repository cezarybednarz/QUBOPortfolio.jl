include("../../src/QUBOPortfolio.jl")
include("load_dataset.jl")
# This script is used to classify all instances and save their metrics to cache.


metrics_cache = QUBOPortfolio.load_metrics_from_cache("cache/metrics.jld2")
training_instances, test_instances_qubolib, test_instances_mqlib = load_experimental_instances()
all_instances = vcat(training_instances, test_instances_qubolib, test_instances_mqlib)
classified_instances = QUBOPortfolio.classify_dataset_with_cache(all_instances, metrics_cache)
QUBOPortfolio.save_metrics_to_cache(classified_instances, "cache/metrics.jld2")
