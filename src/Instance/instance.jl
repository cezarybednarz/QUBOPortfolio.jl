abstract type AbstractInstance end

# The basic type to hold the instance data in memory. The QUBO and Max-Cut representations
# are equivalent
Base.@kwdef struct Instance <: AbstractInstance
    name::String
    qubo_instance::QUBOTools.Model
    max_cut_instance::SimpleWeightedGraph
end

# This is for MQLib instances which are stored locally as zipped files.
# This holds the path to load the instance from if needed.
Base.@kwdef mutable struct InstanceWithPath <: AbstractInstance
    instance::Union{Instance, Nothing} = nothing
    path_to_load_from::String = ""
end

# A generic function to retrieve the concrete 'Instance' data, regardless of
# whether it was stored directly or needs to be loaded from a path.
function get_instance_data(instance::AbstractInstance)::Instance
    return get_instance_data(instance)
end

function get_instance_data(instance::Instance)::Instance
    return instance
end

function get_instance_data(instance::InstanceWithPath)::Instance
    if instance.instance === nothing
        @info "Instance data not loaded. Loading from path: $(instance.path_to_load_from)..."
        loaded_data = load_zipped_max_cut_instance(instance.path_to_load_from)
        instance.instance = loaded_data
    end
    return instance.instance::Instance
end

function get_instance_name(instance::AbstractInstance)::String
    return get_instance_name(instance)
end

function get_instance_name(instance::Instance)::String
    return instance.name
end

function get_instance_name(instance::InstanceWithPath)::String
    return basename(instance.path_to_load_from)
end

function num_qubo_variables(instance::AbstractInstance)::Int
    loaded_instance = get_instance_data(instance)
    linear_terms = Dict(QUBOTools.linear_terms(loaded_instance.qubo_instance))
    quadratic_terms = Dict(QUBOTools.quadratic_terms(loaded_instance.qubo_instance))

    all_vars = Set{Int}()
    for v in keys(linear_terms)
        push!(all_vars, v)
    end
    for (v1, v2) in keys(quadratic_terms)
        push!(all_vars, v1)
        push!(all_vars, v2)
    end

    variables = sort(collect(all_vars))
    num_vars = length(variables)
    return num_vars
end
