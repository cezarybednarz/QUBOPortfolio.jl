include("../Instance/instance.jl")

struct Metric
    name::String
    value::Float64
end

function load_metrics_from_cache(cache_location::String="cache/metrics.jld2")::Dict{String, Vector{Metric}}
    if !isfile(cache_location)
        @info "No cached metrics found at $cache_location"
        return Dict{String, Vector{Metric}}()
    end
    return JLD2.load(cache_location, "metrics")
end

function save_metrics_to_cache(metrics::Dict{String, Vector{Metric}}, cache_location::String)
    JLD2.save(cache_location, "metrics", metrics)
    @info "Saved metrics to $cache_location"
end

function classify(instance::QUBOPortfolio.Instance)::Vector{Metric}
    # convert instance to MAX-CUT graph
    max_cut_instance = instance.max_cut_instance
    @info "classifying instance $(instance.name) with $(length(vertices(max_cut_instance))) vertices and $(length(edges(max_cut_instance))) edges"

    instance_edges = collect(edges(max_cut_instance))

    max_degree = maximum(degree(max_cut_instance))

    deg = [degree(max_cut_instance, v) / max_degree for v in vertices(max_cut_instance)]
    mis = get_mis(max_cut_instance)

    log_norm_ev1, log_norm_ev2, log_ev_ratio = get_spectral_metrics(max_cut_instance)
    weights = [weight(e) for e in instance_edges]

    weights_summary = get_summary(weights)
    deg_summary = get_summary(deg)
    core_summary = get_summary(get_core(max_cut_instance))
    avg_neighbour_summary = get_summary(get_avg_neighbour(max_cut_instance))
    clust_summary = get_summary(get_clust(max_cut_instance))
    avg_deg_conn = get_summary(get_avg_deg_conn(max_cut_instance))

    # calculate all the MQLib metrics
    calculated_metrics = [
        Metric("log_n", log(length(vertices(max_cut_instance)))),
        Metric("log_m", log(length(instance_edges))),
        Metric("percent_pos", get_percent_pos(max_cut_instance)),
        Metric("chromatic", get_chromatic(max_cut_instance)),
        Metric("disconnected", get_disconnected(max_cut_instance)),
        Metric("assortativity", Graphs.assortativity(max_cut_instance)),
        Metric("mis", mis),
        Metric("log_norm_ev1", log_norm_ev1),
        Metric("log_norm_ev2", log_norm_ev2),
        Metric("log_ev_ratio", log_ev_ratio),
        Metric("weight_min", weights_summary.min),
        Metric("weight_max", weights_summary.max),
        Metric("weight_mean", weights_summary.mean),
        Metric("weight_stdev", weights_summary.stdev),
        Metric("weight_const", weights_summary.constant),
        Metric("weight_log_abs_skew", weights_summary.log_abs_skew),
        Metric("weight_skew_positive", weights_summary.skew_positive),
        Metric("weight_log_kurtosis", weights_summary.log_kurtosis),
        Metric("deg_min", deg_summary.min),
        Metric("deg_max", deg_summary.max),
        Metric("deg_mean", deg_summary.mean),
        Metric("deg_stdev", deg_summary.stdev),
        Metric("deg_const", deg_summary.constant),
        Metric("deg_log_abs_skew", deg_summary.log_abs_skew),
        Metric("deg_skew_positive", deg_summary.skew_positive),
        Metric("deg_log_kurtosis", deg_summary.log_kurtosis),
        Metric("core_min", core_summary.min),
        Metric("core_max", core_summary.max),
        Metric("core_mean", core_summary.mean),
        Metric("core_stdev", core_summary.stdev),
        Metric("core_const", core_summary.constant),
        Metric("core_log_abs_skew", core_summary.log_abs_skew),
        Metric("core_skew_positive", core_summary.skew_positive),
        Metric("core_log_kurtosis", core_summary.log_kurtosis),
        Metric("avg_neighbour_min", avg_neighbour_summary.min),
        Metric("avg_neighbour_max", avg_neighbour_summary.max),
        Metric("avg_neighbour_mean", avg_neighbour_summary.mean),
        Metric("avg_neighbour_stdev", avg_neighbour_summary.stdev),
        Metric("avg_neighbour_const", avg_neighbour_summary.constant),
        Metric("avg_neighbour_log_abs_skew", avg_neighbour_summary.log_abs_skew),
        Metric("avg_neighbour_skew_positive", avg_neighbour_summary.skew_positive),
        Metric("avg_neighbour_log_kurtosis", avg_neighbour_summary.log_kurtosis),
        Metric("clust_min", clust_summary.min),
        Metric("clust_max", clust_summary.max),
        Metric("clust_mean", clust_summary.mean),
        Metric("clust_stdev", clust_summary.stdev),
        Metric("clust_const", clust_summary.constant),
        Metric("clust_log_abs_skew", clust_summary.log_abs_skew),
        Metric("clust_skew_positive", clust_summary.skew_positive),
        Metric("clust_log_kurtosis", clust_summary.log_kurtosis),
        Metric("avg_deg_conn_min", avg_deg_conn.min),
        Metric("avg_deg_conn_max", avg_deg_conn.max),
        Metric("avg_deg_conn_mean", avg_deg_conn.mean),
        Metric("avg_deg_conn_stdev", avg_deg_conn.stdev),
        Metric("avg_deg_conn_const", avg_deg_conn.constant),
        Metric("avg_deg_conn_log_abs_skew", avg_deg_conn.log_abs_skew),
        Metric("avg_deg_conn_skew_positive", avg_deg_conn.skew_positive),
        Metric("avg_deg_conn_log_kurtosis", avg_deg_conn.log_kurtosis)
    ]

    return calculated_metrics
end

function classify_dataset_with_cache(instances::Vector, cached_metrics::Union{Dict{String, Vector{Metric}}, Nothing}=nothing)::Dict{String, Vector{Metric}}
    metrics = Dict{String, Vector{Metric}}()
    num_instances = length(instances)
    loaded_metrics = Dict{String, Vector{Metric}}()
    if cached_metrics !== nothing
        loaded_metrics = cached_metrics
    end
    for instance in instances
        @info "Classifying instance: ($(length(metrics)+1)/$num_instances)"
        instance_name = get_instance_name(instance)
        if haskey(loaded_metrics, instance_name)
            @info "Loaded metrics for instance $instance_name from cache"
            metrics[instance_name] = loaded_metrics[instance_name]
        else
            # not in cache, need to calculate
            loaded_instance = get_instance_data(instance)
            metrics[loaded_instance.name] = classify(loaded_instance)
            # TODO: store the metrics in cache
        end
    end
    return metrics
end


function classify_to_df(instances::Vector, cached_metrics::Union{Dict{String, Vector{Metric}}, Nothing}=nothing)::DataFrame
    if (length(instances) == 0)
        error("Instances cannot be empty")
    end

    instance_metrics = classify_dataset_with_cache(instances, cached_metrics)
    first_instance_name = get_instance_name(instances[1])
    metric_names = map(metric -> metric.name, instance_metrics[first_instance_name])

    metrics_df = DataFrame()

    for metric_name in metric_names
        metrics_df[!, metric_name] = Float64[]
    end

    for (instance, metrics) in instance_metrics
        row_data = Dict{Symbol, Float64}()
        for (i, metric_name) in enumerate(metric_names)
            row_data[Symbol(metric_name)] = metrics[i].value
        end
        push!(metrics_df, row_data)
    end

    return metrics_df
end

### Metrics implementations

struct Summary
    min::Float64
    max::Float64
    mean::Float64
    stdev::Float64
    log_kurtosis::Float64
    log_abs_skew::Float64
    skew_positive::Float64
    constant::Float64
end

function get_summary(data::Vector{Float64})::Summary
    n = length(data)
    if n == 0
        error("Data cannot be empty")
    end

    data_min = minimum(data)
    data_max = maximum(data)

    if (data_min == data_max)
        return Summary(data_min, data_min, data_min, 0.0, 0.0, 0.0, 1.0, 1.0)
    end

    data_mean = mean(data)
    data_stdev = std(data)

    skew = StatsBase.skewness(data)
    log_abs_skew = log(1.0 + abs(skew))

    kurtosis = StatsBase.kurtosis(data)
    log_kurt = log(4.0 + kurtosis)

    skew_pos = skew >= 0.0 ? 1.0 : 0.0

    return Summary(data_min, data_max, data_mean, data_stdev, log_kurt, log_abs_skew,
                   skew_pos, 0.0)
end


function get_percent_pos(g::SimpleWeightedGraph)
    g_edges = collect(edges(g))
    weights = [weight(e) for e in g_edges]
    length(filter(e -> e > 0.0, weights)) / length(weights)
end

function get_chromatic(g::SimpleWeightedGraph)
    # todo: implement Welsh-Powell algorithm
    greedy_color(g).num_colors / length(vertices(g))
end

function get_disconnected(g::SimpleWeightedGraph)
    is_connected(g) ? 0.0 : 1.0
end

function get_core(g::SimpleWeightedGraph)
    n = length(vertices(g))
    [core_number / n for core_number in core_number(g)]
end

function get_avg_neighbour(g::SimpleWeightedGraph)
    max_degree = maximum(degree(g))
    [d / max_degree for d in degree(g)]
end

function get_clust(g::SimpleWeightedGraph)
    n = length(vertices(g))
    subset_size = min(ceil(Int, 3 * log(n)), n)
    random_subset = shuffle(collect(1:n))[1:subset_size]
    coefficients = local_clustering_coefficient(g, random_subset)
    filter(!isnan, coefficients)
end

function get_avg_deg_conn(g::SimpleWeightedGraph)
    degrees = degree(g)
    degree_connectivity = Dict{Int, Float64}()

    for degree_value in unique(degrees)
        nodes_with_degree = findall(x -> x == degree_value, degrees)

        if !isempty(nodes_with_degree)
            neighbor_degrees = [degree(g, neighbor) for node in nodes_with_degree for neighbor in neighbors(g, node)]
            degree_connectivity[degree_value] = mean(neighbor_degrees)
        end
    end

    return  collect(values(degree_connectivity))
end

function get_mis(g::SimpleWeightedGraph)
    length(independent_set(g, DegreeIndependentSet())) / length(vertices(g))
end

function get_spectral_metrics(g::SimpleWeightedGraph)
    L = laplacian_matrix(g, Float64)

    # Use eigs to compute the two largest eigenvalues (faster than graphs.jl implementation)
    ev, _ = eigs(L, nev=2, ncv=20, maxiter=5000, which=:LR)

    eigenvals = sort(real.(ev), rev=true)

    avg_degree = mean(degree(g))

    # Add a small epsilon to avoid division by zero or log of zero
    ev1 = eigenvals[1]
    ev2 = eigenvals[2]

    norm_ev1 = ev1 / avg_degree
    norm_ev2 = ev2 / avg_degree
    ev_ratio = ev1 / ev2

    log_norm_ev1 = log(max(norm_ev1, 1e-10))
    log_norm_ev2 = log(max(norm_ev2, 1e-10))
    log_ev_ratio = log(max(ev_ratio, 1e-10))

    return log_norm_ev1, log_norm_ev2, log_ev_ratio
end
