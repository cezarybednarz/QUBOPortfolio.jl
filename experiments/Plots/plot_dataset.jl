if !isdefined(Main, :QUBOPortfolio)
    include("../../src/QUBOPortfolio.jl")
end
include("../Dataset/load_dataset.jl")
using Plots

# This script is used to plot the dataset, the QUBOPortfolio.Instances

function get_density(instance_metrics::Vector{QUBOPortfolio.Metric})
    n = 0
    m = 0
    for metric in instance_metrics
        if metric.name == "log_n"
            n = exp(metric.value)
        elseif metric.name == "log_m"
            m = exp(metric.value)
        end
    end
    if n == 0
        return NaN
    end
    return m / (n * (n - 1) / 2)
end
function plot_density_vs_num_variables(classified_instances::Dict, name::String="Density vs Number of variables")
    x_values = Float64[]
    y_values = Float64[]
    labels = String[]

    for (name, metrics) in classified_instances
        n = NaN
        for metric in metrics
            if metric.name == "log_n"
                n = exp(metric.value)
                break
            end
        end
        density = get_density(metrics)
        if !isnan(n) && !isnan(density)
            push!(x_values, n)
            push!(y_values, density * 100)
            push!(labels, name)
        end
    end

    p = scatter(x_values, y_values;
        xlabel="Number of variables",
        ylabel="Density (%)",
        title=name,
        legend=false,
        markerstrokewidth=0.5,
        markersize=6,
        fontfamily="Computer Modern",
        xscale=:log10,
        xticks=[10^1, 10^2, 10^3, 10^4],
        size=(300, 300))
    display(p)
    savefig(p, "plots/$(replace(name, ' ' => '_')).png")
end


function plot_metrics_comparison(classified_instances::Dict, metric_x::String, metric_y::String, x_func, y_func)
    x_values = Float64[]
    y_values = Float64[]
    labels = String[]

    for (name, metrics) in classified_instances
        metric_x_value = NaN
        metric_y_value = NaN

        for metric in metrics
            if metric.name == metric_x
                metric_x_value = x_func(metric.value)
            elseif metric.name == metric_y
                metric_y_value = y_func(metric.value)
            end
        end

        if !isnan(metric_x_value) && !isnan(metric_y_value)
            push!(x_values, metric_x_value)
            push!(y_values, metric_y_value)
            push!(labels, name)
        end
    end

    p = scatter(x_values, y_values,
            xlabel=metric_x,
            ylabel=metric_y,
            title="$metric_x vs $metric_y",
            legend=false,
            markerstrokewidth=0.5,
            markersize=6,
            fontfamily="Computer Modern")
    display(p)
end
