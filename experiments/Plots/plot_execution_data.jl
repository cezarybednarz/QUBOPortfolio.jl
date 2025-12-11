if !isdefined(Main, :QUBOPortfolio)
    include("../../src/QUBOPortfolio.jl")
end
# This script is used to plot the results from `run_heuristics_on_dataset` function.
# This is a LLM generated mess but gets the job done.

import MQLib
using Plots
using Dates
using PrettyTables
using Statistics
using StatsBase
using Distributions
using Printf

function best_result_from_results(results::Vector{QUBOPortfolio.Result}; filename::Union{String, Nothing}=nothing)
    return minimum(result.energy for result in results)
end

# Analyse benchmark results
function plot_execution_times(results::QUBOPortfolio.ExecutionResults, instances::Vector{<:QUBOPortfolio.AbstractInstance}; filename="plots/execution_time_vs_instance_size.png")
    # Create a dictionary to store data for each heuristic
    heuristic_data = Dict{String, Dict{Int, Vector{Float64}}}()
    metrics_cache = QUBOPortfolio.load_metrics_from_cache()
    classified_instances = QUBOPortfolio.classify_dataset_with_cache(instances, metrics_cache)

    instance_sizes = get_instances_sizes(instances)

    for (key, result_vector) in results.results
        heuristic_name = key.heuristic_name
        instance_name = key.instance_name

        if !haskey(instance_sizes, instance_name)
            continue # Skip if size could not be determined
        end
        instance_size = instance_sizes[instance_name]

        if !haskey(heuristic_data, heuristic_name)
            heuristic_data[heuristic_name] = Dict{Int, Vector{Float64}}()
        end
        if !haskey(heuristic_data[heuristic_name], instance_size)
            heuristic_data[heuristic_name][instance_size] = Float64[]
        end

        # Each result in the vector has a runtime. We collect all of them.
        for result in result_vector
            push!(heuristic_data[heuristic_name][instance_size], result.time_taken)
        end
    end

    # Compute averages for each heuristic and size
    heuristic_averages = Dict{String, Tuple{Vector{Int}, Vector{Float64}}}()
    for (heuristic_name, size_times) in heuristic_data
        sizes = Int[]
        avg_times = Float64[]
        for (size, times) in sort(collect(size_times), by=x->x[1])
            push!(sizes, size)
            push!(avg_times, sum(times) / length(times))
        end
        heuristic_averages[heuristic_name] = (sizes, avg_times)
    end

    # Create the plot
    p = plot(xlabel="Instance Size (# of QUBO variables)", ylabel="Runtime (s)",
             title="Runtime vs Instance Size", legend=:topleft,
             xaxis=:log10, yaxis=:log10,
             xticks=[10^1, 10^2, 10^3, 10^4],
             fontfamily="Computer Modern")

    # Plot each heuristic with a different color, using both lines and markers
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5, :cross, :xcross]
    sorted_heuristics = sort(collect(keys(heuristic_averages)))
    for (i, heuristic_name) in enumerate(sorted_heuristics)
        sizes, avg_times = heuristic_averages[heuristic_name]
        marker_symbol = markers[mod1(i, length(markers))]
        plot!(p, sizes, avg_times, label=heuristic_name,
              marker=marker_symbol, markersize=4, alpha=0.7, linewidth=1.5)
    end
    display(p)
    savefig(p, filename)
end

# Plot Reference Gap vs Instance Size
function plot_performance(results::QUBOPortfolio.ExecutionResults,
                          instances::Vector{<:QUBOPortfolio.AbstractInstance};
                          baseline_heuristic_name="BASELINE",
                          filter_heuristics=String[],
                          filename="plots/performance_gap.png"
)
    baseline_results = Dict{String, Float64}()
    for (key, result_vector) in results.results
        if key.heuristic_name == baseline_heuristic_name
            value = best_result_from_results(result_vector)
            baseline_results[key.instance_name] = value
        end
    end

    if isempty(baseline_results)
        @warn "No baseline results found for '$baseline_heuristic_name'. Using lowest result as baseline."
    end

    instance_sizes = get_instances_sizes(instances)

    # Store gaps by heuristic and size
    heuristic_size_gaps = Dict{String, Dict{Int, Vector{Float64}}}()
    for (key, result_vector) in results.results
        heur_name = key.heuristic_name
        inst_name = key.instance_name

        if heur_name == baseline_heuristic_name
            continue
        end

        if !isempty(filter_heuristics) && !(heur_name in filter_heuristics)
            continue
        end

        if !haskey(baseline_results, inst_name)
            @warn "No baseline for instance $inst_name, skipping"
            continue
        end
        inst_size = instance_sizes[inst_name]

        # calculate best result (best of multiple runs)
        value = best_result_from_results(result_vector)
        baseline_val = baseline_results[inst_name]

        # Avoid division by zero or near-zero for gap calculation
        gap = if abs(baseline_val) > 1e-9
            100.0 * (value - baseline_val) / abs(baseline_val)
        else
            100.0 * (value - baseline_val)
        end

        if !haskey(heuristic_size_gaps, heur_name)
            heuristic_size_gaps[heur_name] = Dict{Int, Vector{Float64}}()
        end

        if !haskey(heuristic_size_gaps[heur_name], inst_size)
            heuristic_size_gaps[heur_name][inst_size] = Float64[]
        end
        push!(heuristic_size_gaps[heur_name][inst_size], gap)
    end

    # Compute averages for each heuristic and size
    heuristic_gaps = Dict{String, Tuple{Vector{Int}, Vector{Float64}}}()
    for (heur_name, size_gaps_dict) in heuristic_size_gaps
        sizes = Int[]
        avg_gaps = Float64[]
        for (size, gaps) in sort(collect(size_gaps_dict), by=x->x[1])
            push!(sizes, size)
            push!(avg_gaps, sum(gaps) / length(gaps))
        end
        heuristic_gaps[heur_name] = (sizes, avg_gaps)
    end

    p = plot(xlabel="QUBO Variables", ylabel="Reference Gap (%)",
             title="Reference Gap vs. Instance Size (ref: $baseline_heuristic_name)", legend=:best,
             xaxis=:log10, xticks=[10^1, 10^2, 10^3, 10^4], fontfamily="Computer Modern")

    hline!(p, [0], linestyle=:dash, color=:black, label="", linewidth=1.5) # Baseline reference line

    markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5, :cross, :xcross]
    sorted_heuristics = sort(collect(keys(heuristic_gaps)))
    for (i, heur_name) in enumerate(sorted_heuristics)
        sizes, gaps = heuristic_gaps[heur_name]
        marker_symbol = markers[mod1(i, length(markers))]
        plot!(p, sizes, gaps, label=heur_name,
              marker=marker_symbol, markersize=4, alpha=0.7, linewidth=0, seriestype=:scatter)
    end
    display(p)
    savefig(p, filename)
end

# Count best, only, and worst per heuristic (lower objective is better).
# Best: heuristic achieved the best value (within tolerance).
# Only: heuristic was the unique one to achieve that best value.
# Worst: heuristic achieved the worst value (within tolerance).
function performance_by_heuristic(results::QUBOPortfolio.ExecutionResults; rtol=1e-9, atol=1e-6)
    inst_to_vals = Dict{String, Vector{Tuple{String,Float64}}}()
    for (key, res) in results.results
        heur = key.heuristic_name
        inst = key.instance_name
        value = best_result_from_results(res)
        push!(get!(inst_to_vals, inst, Tuple{String,Float64}[]), (heur, value))
    end

    heuristics = unique([key.heuristic_name for (key, _) in results.results])
    best_count = Dict(h => 0 for h in heuristics)
    only_count = Dict(h => 0 for h in heuristics)
    worst_count = Dict(h => 0 for h in heuristics)

    for (_, hv) in inst_to_vals
        best_val = minimum(last.(hv))
        worst_val = maximum(last.(hv))

        δ_best = max(abs(best_val) * rtol, atol)
        δ_worst = max(abs(worst_val) * rtol, atol)

        near_best = [(h, v) for (h, v) in hv if v <= best_val + δ_best]
        near_worst = [(h, v) for (h, v) in hv if v >= worst_val - δ_worst]

        # Count best
        for (h, _) in near_best
            best_count[h] += 1
        end

        # Count only (unique best)
        if length(near_best) == 1
            h, _ = near_best[1]
            only_count[h] += 1
        end

        # Count worst
        for (h, _) in near_worst
            worst_count[h] += 1
        end
    end

    best_count, only_count, worst_count
end

function timing_stats_by_heuristic(results::QUBOPortfolio.ExecutionResults)
    timing_stats = Dict{String, Tuple{Float64, Float64, Float64, Float64}}()
    heuristic_names = unique(key.heuristic_name for (key, _) in results.results)

    for h_name in heuristic_names
        times = Float64[]
        resources_per_heuristic = QUBOPortfolio.Resource[]

        # Find all results for this heuristic
        for (key, result_vector) in results.results
            if key.heuristic_name == h_name
                resources = QUBOPortfolio.get_results_resources(result_vector)
                time = QUBOPortfolio.get_results_max_time(result_vector)
                push!(resources_per_heuristic, resources)
                push!(times, time)
            end
        end

        @info "Heuristic '$h_name' ran on $(length(times)) runs."
        if !isempty(times)
            avg_time = sum(times) / length(times)
            max_time = maximum(times)
            cpu_hours = sum(r.CPU for r in resources_per_heuristic) / 3600.0
            gpu_hours = sum(r.GPU for r in resources_per_heuristic) / 3600.0
            timing_stats[h_name] = (avg_time, max_time, cpu_hours, gpu_hours)
        else
            timing_stats[h_name] = (NaN, NaN, NaN, NaN)
        end
    end
    return timing_stats
end

function print_latex_performance_table(rows::Vector{Any})
    println(raw"\begin{table}[h!]")
    println(raw"\centering")
    println(raw"\resizebox{\textwidth}{!}{%")
    println(raw"\begin{tabular}{l|rrr|rrr}")
    println(raw"\toprule")
    println(raw"\textbf{Heuristic} & \textbf{Best} & \textbf{Only} & \textbf{Worst} & \textbf{Avg Time (s)} & \textbf{Core Hours} & \textbf{GPU Hours} \\\\")
    println(raw"\midrule")

    for row in rows
        h_name = row[1]
        best = row[2]
        only = row[3]
        worst = row[4]
        avg_time = row[5]
        cpu_hours = row[6]
        gpu_hours = row[7]

        # Format numbers
        avg_time_str = isnan(avg_time) ? "NaN" : Printf.@sprintf("%.3f", avg_time)
        cpu_hours_str = isnan(cpu_hours) ? "NaN" : Printf.@sprintf("%.4f", cpu_hours)
        gpu_hours_str = isnan(gpu_hours) ? "NaN" : Printf.@sprintf("%.4f", gpu_hours)

        # Escape underscores in heuristic names for LaTeX
        safe_name = replace(h_name, "_" => "\\_")

        println("$safe_name & $best & $only & $worst & $avg_time_str & $cpu_hours_str & $gpu_hours_str \\\\")
    end

    println(raw"\bottomrule")
    println(raw"\end{tabular}")
    println(raw"}")
    println(raw"\caption{Performance comparison of heuristics.}")
    println(raw"\label{tab:performance}")
    println(raw"\end{table}")
end

function print_performance_table(results::QUBOPortfolio.ExecutionResults;
                                 filter_heuristics::Union{Vector{String}, Nothing}=nothing,
                                 rtol=1e-9, atol=1e-6)
    best_count, only_count, worst_count = performance_by_heuristic(results; rtol=rtol, atol=atol)
    timing_stats = timing_stats_by_heuristic(results)

    # Combine counts and stats into a sortable structure
    unsorted_rows = []
    for h in keys(best_count)
        if !isnothing(filter_heuristics) && h ∉ filter_heuristics
            continue
        end

        avg_time, max_time, cpu_hours, gpu_hours = get(timing_stats, h, (NaN, NaN, NaN, NaN))

        display_name = replace(h, "SIMULATED_BIFURCATION" => "SB")
        push!(unsorted_rows, (display_name, best_count[h], only_count[h], worst_count[h], avg_time, cpu_hours, gpu_hours))
    end
    rows = sort(unsorted_rows, by = x -> (-x[2], -x[3], x[4], x[1]))

    total_inst = length(unique([key.instance_name for (key, _) in results.results]))
    println("Performance by heuristic over $total_inst instances:")

    # Convert the vector of tuples to a matrix for PrettyTables
    if isempty(rows)
        println("No data to display.")
        return
    end

    data = permutedims(hcat(collect.(rows)...))
    header = ["Heuristic", "Best", "Only", "Worst", "Avg Time (s)", "Core hours", "GPU hours"]

    pretty_table(data, header=header, crop=:none, formatters=(ft_printf("%.3f", [5]), ft_printf("%.4f", [6, 7])))

    println("\n--- LaTeX Table Output ---\n")
    print_latex_performance_table(rows)
end

function plot_histogram_execution_times(results::QUBOPortfolio.ExecutionResults,
                                        heuristic_name::String;
                                        bin_width::Real=5.0,
                                        distribution::Bool=false,
                                        filename="plots/execution_time_histogram.png")
    times = Float64[]
    for (key, result_vector) in results.results
        if key.heuristic_name == heuristic_name
            for result in result_vector
                push!(times, result.time_taken)
            end
        end
    end

    if isempty(times)
        @warn "No execution times found for heuristic '$heuristic_name'. Cannot generate histogram."
        return
    end

    # ======================================================
    # 1. LOG-NORMAL STATISTICS
    # ======================================================
    valid_times = filter(x -> x > 0, times)
    log_times = log.(valid_times)

    mu_log = mean(log_times)
    sigma_log = std(log_times)

    # Convert back to linear scale
    geo_mean = exp(mu_log)
    geo_std  = exp(sigma_log)

    # Calculate 95% interval
    log_lower_95 = geo_mean / (geo_std^2)
    log_upper_95 = geo_mean * (geo_std^2)

    # ======================================================
    # 2. PLOTTING
    # ======================================================

    # Create bins with constant width
    max_val = maximum(times)
    # Ensure bins cover the data and the visible range (0-300)
    limit = max(300.0, max_val)
    bins_vec = 0:bin_width:(limit + bin_width)

    display_name = replace(heuristic_name, "SIMULATED_BIFURCATION" => "SB")

    # Removed explicit font sizing to match plot_dataset.jl defaults
    p = histogram(times, bins=bins_vec, xlabel="Runtime(s)", ylabel="Frequency",
                  title="$display_name",
                  fontfamily="Computer Modern",
                  label="Observed Data", legend=:topright,
                  color=:steelblue, alpha=0.5, linecolor=:white,
                  size=distribution ? (400, 400) : (300, 300), xlims=(0, 300))

    if distribution
        # Plot LogNormal distribution curve
        d = LogNormal(mu_log, sigma_log)
        x_vals = range(0.1, 300, length=500)

        # Scale PDF to match frequency histogram: PDF * Total Count * Bin Width
        scale_factor = length(times) * bin_width
        y_vals = pdf.(d, x_vals) .* scale_factor

        plot!(p, x_vals, y_vals, linewidth=2, color=:black, label="LogNormal Fit")

        # --- A. Add Geometric Mean ---
        vline!(p, [geo_mean], linewidth=2, color=:forestgreen, linestyle=:solid,
               label="Geo. Mean ($(round(geo_mean, digits=1)))")

        # --- B. Add 95% Range Markers (SPLIT) ---
        vline!(p, [log_lower_95], linewidth=2, color=:darkred, linestyle=:dot,
               label="95% Lower ($(round(log_lower_95, digits=1)))")

        vline!(p, [log_upper_95], linewidth=2, color=:darkred, linestyle=:dot,
               label="95% Upper ($(round(log_upper_95, digits=1)))")
    else
        # --- A. Add Geometric Mean ---
        vline!(p, [geo_mean], linewidth=2, color=:forestgreen, linestyle=:solid,
               label="Geo. Mean ($(round(geo_mean, digits=1)))")

        # --- B. Add 95% Range Markers (SPLIT) ---
        vline!(p, [log_lower_95], linewidth=2, color=:darkred, linestyle=:dot,
               label="95% Lower ($(round(log_lower_95, digits=1)))")

        vline!(p, [log_upper_95], linewidth=2, color=:darkred, linestyle=:dot,
               label="95% Upper ($(round(log_upper_95, digits=1)))")
    end

    # --- Formatting Fixes ---
    # Only ensure 100 is visible if relevant (cleaned up logic)
    try
        current_ticks = xticks(p)[1][1]
        if 100 < xlims(p)[2] && 100 ∉ current_ticks
            new_ticks = sort(unique([current_ticks..., 100]))
            xticks!(p, new_ticks)
        end
    catch
        # Fallback mechanism
    end

    display(p)
    savefig(p, filename)
end
function plot_histogram_number_of_heuristics(results::QUBOPortfolio.ExecutionResults,
                                        heuristic_name::String;
                                        skip_portfolio::Bool=false,
                                        filename="plots/number_of_heuristics_histogram.png",
                                        atol=1e-6, rtol=1e-9)
    # 1. Calculate global best per instance across ALL heuristics
    global_bests = Dict{String, Float64}()
    for (key, res_vec) in results.results
        # Get best energy for this specific heuristic run on this instance
        current_min = best_result_from_results(res_vec)
        # Update global best for the instance
        if !haskey(global_bests, key.instance_name) || current_min < global_bests[key.instance_name]
            global_bests[key.instance_name] = current_min
        end
    end

    # 2. Count occurrences and best-result occurrences
    heuristic_counts = Dict{String, Int}()
    heuristic_best_counts = Dict{String, Int}()

    data_found = false
    for (key, result_vector) in results.results
        if key.heuristic_name == heuristic_name
            inst_best = global_bests[key.instance_name]
            # Tolerance threshold
            δ = max(abs(inst_best) * rtol, atol)

            for result in result_vector
                data_found = true
                # The Result struct contains `used_heuristics` vector.
                for h in result.used_heuristics
                    if skip_portfolio && h.type == QUBOPortfolio.QUBOPORTFOLIO
                        continue
                    end

                    h_name = h.name

                    # Skip the main heuristic itself if it appears in the list
                    if h_name == heuristic_name
                        continue
                    end

                    heuristic_counts[h_name] = get(heuristic_counts, h_name, 0) + 1

                    # Check if this sub-heuristic achieved the best global result
                    if result.energy <= inst_best + δ
                        heuristic_best_counts[h_name] = get(heuristic_best_counts, h_name, 0) + 1
                    end
                end
            end
        end
    end

    if !data_found || isempty(heuristic_counts)
        @warn "No data found for heuristic '$heuristic_name'. Cannot generate histogram."
        return
    end

    # 3. Sort heuristics by frequency in descending order
    sorted_data = sort(collect(heuristic_counts), by=x->x[2], rev=true)
    labels = first.(sorted_data)
    freqs = last.(sorted_data)

    # Get the corresponding "best" counts in the same order
    best_freqs = [get(heuristic_best_counts, l, 0) for l in labels]

    # Clean up labels for display
    display_labels = replace.(labels, "SIMULATED_BIFURCATION" => "SB")
    display_name = replace(heuristic_name, "SIMULATED_BIFURCATION" => "SB")
    display_name = replace(display_name, "_" => " ")

    # 4. Plotting
    max_val = isempty(freqs) ? 0 : maximum(freqs)
    # Base bar: Total executions
    p = bar(display_labels, freqs,
            xlabel="Heuristic", ylabel="Frequency",
            title=display_name,
            fontfamily="Computer Modern",
            guidefontsize=14, tickfontsize=10, legendfontsize=10, titlefontsize=16,
            label="Total Executions", legend=:topright,
            color=:steelblue, alpha=0.6, linecolor=:white,
            size=(900, 600),
            xrotation=45,
            bottom_margin=20Plots.mm, left_margin=20Plots.mm, right_margin=15Plots.mm,
            ylims=(0, max_val * 1.15)) # Increase ylim to fit annotations
            # Overlay bar: Best results
            bar!(p, display_labels, best_freqs,
                 label="Best Result (Global)",
                 color=:seagreen, alpha=0.8, linecolor=:white)

            # Add annotations on top of the bars
            anns = Tuple{Float64, Float64, Any}[]
            for (i, (f, b)) in enumerate(zip(freqs, best_freqs))
                # Annotation for Total (Blue)
                # Only show if different from Best to avoid overlap
                if f != b
                    push!(anns, (Float64(i) - 0.5, Float64(f), text(string(f), font("Computer Modern", 8, :black), :bottom)))
                end

                # Annotation for Best (Green)
                if b > 0
                    push!(anns, (Float64(i) - 0.5, Float64(b), text(string(b), font("Computer Modern", 8, :black), :bottom)))
                end
            end
            annotate!(p, anns)


    display(p)
    savefig(p, filename)
end
