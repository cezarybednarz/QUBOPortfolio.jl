if !isdefined(Main, :QUBOPortfolio)
    include("../../src/QUBOPortfolio.jl")
end
import JLD2

function get_instances_sizes(instances::Vector{<:QUBOPortfolio.AbstractInstance})
    cached_metrics = QUBOPortfolio.load_metrics_from_cache("cache/metrics.jld2")
    classified_instances = QUBOPortfolio.classify_dataset_with_cache(instances, cached_metrics)
    instance_sizes = Dict{String, Int}()
    for (instance_name, metrics) in classified_instances
        for metric in metrics
            if metric.name == "log_n"
                instance_sizes[instance_name] = round(Int, exp(metric.value))
            end
        end
    end
    return instance_sizes
end

function load_experimental_instances(replace_path::Bool=true)::Tuple{Vector{QUBOPortfolio.AbstractInstance}, Vector{QUBOPortfolio.AbstractInstance}, Vector{QUBOPortfolio.AbstractInstance}}
    train_instances = JLD2.load("data/train_instances.jld2", "mqlib_instances")
    test_instances_qubolib = JLD2.load("data/test_instances_qubolib.jld2", "test_instances_qubolib")
    test_instances_mqlib = JLD2.load("data/test_instances_mqlib.jld2", "test_instances_mqlib")

    # TODO remove hardcoded path conversions between local and VM paths
    if replace_path
        for instance in vcat(train_instances, test_instances_mqlib)
            instance.path_to_load_from = replace(instance.path_to_load_from, "Studia/Magisterka/" => "")
        end
    end

    return train_instances, test_instances_qubolib, test_instances_mqlib
end
