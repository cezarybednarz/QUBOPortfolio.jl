include("../../src/QUBOPortfolio.jl")
include("load_dataset.jl")
import JLD2
import Random

# This script is used to save datasets for experiments into data/ folder

# Bigger instances that 1_000_000 bytes are not yet classified in cache/metrics.jld2
mqlib_instances = QUBOPortfolio.load_mqlib_instances("/home/cezary/Studia/Magisterka/MQLib-dataset/", 10000, 0, 1_000_000)
qubolib_instances = QUBOPortfolio.get_dataset_from_qubolib()

mqlib_instances_sizes = get_instances_sizes(mqlib_instances)
qubolib_instances_sizes = get_instances_sizes(qubolib_instances)

@info "Loaded $(length(mqlib_instances)) MQLib instances and $(length(qubolib_instances)) QUBOLib instances."

# Leave only instances smaller than 20,000 variables
num_variables_threshold = 20_000
filtered_instances_qubolib = filter(instance -> qubolib_instances_sizes[instance.name] <= num_variables_threshold, qubolib_instances)
filtered_instances_mqlib = filter(instance -> mqlib_instances_sizes[QUBOPortfolio.get_instance_name(instance)] <= num_variables_threshold, mqlib_instances)

Random.seed!(42)
shuffled_qubolib_instances = Random.shuffle(filtered_instances_qubolib)
shuffled_mqlib_instances = Random.shuffle(filtered_instances_mqlib)

# 400 MQLib instances for training
JLD2.save("data/train_instances.jld2", "mqlib_instances", shuffled_mqlib_instances[1:400])
# 100 QUBOLib instances for testing
JLD2.save("data/test_instances_qubolib.jld2", "test_instances_qubolib", shuffled_qubolib_instances[1:100])
# 100 MQLib instances for testing
JLD2.save("data/test_instances_mqlib.jld2", "test_instances_mqlib", shuffled_mqlib_instances[401:500])
