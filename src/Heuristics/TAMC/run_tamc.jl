using Base: run, pipeline
using YAML
using Pickle

# Run TAMC (https://github.com/USCqserver/tamc) for a QUBO instance and return dictionary
# of results.
function run_tamc_on_instance(instance::QUBOPortfolio.Instance, hyperparameters::Dict{String, Any}, heuristic_name::String;
    tamc_path::String="/home/cezary/Studia/Magisterka/tamc/target/release/tamc")::Result
    mktempdir() do dir
        time_taken = @elapsed begin
            instance_file_path = joinpath(dir, "instance.txt")
            method_file_path = joinpath(dir, "method.yml")
            output_file_path = joinpath(dir, "output.yaml")
            sample_output_path = joinpath(dir, "sample.pkl")

            # Convert QUBO to Ising
            convert_and_write_qubo_as_ising(instance, instance_file_path)

            # Write hyperparameters to YAML file
            open(method_file_path, "w") do io
                YAML.write(io, hyperparameters)
            end

            # Run the TAMC binary
            run_tamc(method_file_path, instance_file_path, output_file_path, tamc_path, sample_output=sample_output_path)

            # Read the file with results
            ising_result = read_ising_result(QUBOPortfolio.num_qubo_variables(instance), sample_output_path, instance)

            # DISCARD the solver's energy. RECALCULATE it from the returned solution
            # vector to guarantee consistency (the solver's energy is inconsistent)
            recalculated_energy = calculate_energy_from_solution(instance, ising_result.solution)
        end
        return Result(
            energy=recalculated_energy,
            solution=ising_result.solution,
            time_taken=time_taken,
            used_heuristics=[Heuristic(type=QUBOPortfolio.TAMC, name=heuristic_name, hyperparameters=hyperparameters)]
        )
    end
end

# Read the solution bits from the TAMC output file.
function read_ising_result(N::Int, samples_pkl::String, instance::QUBOPortfolio.Instance)::Result
    pkl = Pickle.load(open(samples_pkl))

    energies = pkl["e"][end]

    L = length(pkl["e"][end])
    perm = sortperm(pkl["e"][end])

    states = Array{Int}(undef, N, L)
    states .= 1

    for r ∈ 1:L
        pst = pkl["samples"][end][perm[r]]
        for var ∈ 1:N
            byte, bit = divrem(var - 1, 8)
            flag = (UInt8(pst[byte + 1][1]) >> bit) & 1
            states[var, r] = flag
        end
    end


    # Loop through all the states on the output and get the best one.
    # Disregard the TAMC energies.
    best_state = Int[]
    best_energy = Inf
    for r ∈ 1:L
        state = states[:, perm[r]]
        energy = calculate_energy_from_solution(instance, state)
        if energy < best_energy
            best_energy = energy
            best_state = state
        end
    end

    return Result(energy=Inf, solution=best_state)
end

# Convert an upper-triangular QUBO instance to a symmetric Ising format and write it to a file.
# Assumes the solver uses the H = +J*s*s + h*s convention.
function convert_and_write_qubo_as_ising(instance::Instance, filepath::String; zero_based::Bool=true)
    linear_terms = Dict(collect(QUBOTools.linear_terms(instance.qubo_instance)))
    quadratic_terms = Dict(collect(QUBOTools.quadratic_terms(instance.qubo_instance)))

    ising_couplings = Dict{Tuple{Int, Int}, Float64}()
    for ((v1, v2), weight) in quadratic_terms
        # Formula: J_ij = q_ij / 4
        j_val = weight / 4.0

        # Create a symmetric representation.
        ising_couplings[(v1, v2)] = j_val
        ising_couplings[(v2, v1)] = j_val
    end

    all_vars = union(keys(linear_terms), (v for (v1, v2) in keys(quadratic_terms) for v in (v1, v2))...)
    off_diag_sums = Dict{Int, Float64}(v => 0.0 for v in all_vars)

    for ((v1, v2), weight) in quadratic_terms
        off_diag_sums[v1] += weight
        off_diag_sums[v2] += weight
    end

    ising_biases = Dict{Int, Float64}()
    for v in all_vars
        q_ii = get(linear_terms, v, 0.0)
        sum_q_ij = get(off_diag_sums, v, 0.0)

        # Formula: h_i = q_ii/2 + (sum_{j!=i} q_ij)/4
        ising_biases[v] = q_ii / 2.0 + sum_q_ij / 4.0
    end

    # Write to file
    open(filepath, "w") do io
        for (v, h_val) in sort(collect(ising_biases))
            idx = zero_based ? v - 1 : v
            write(io, "$idx $idx $h_val\n")
        end

        for ((v1, v2), j_val) in sort(collect(ising_couplings))
            if v1 == v2
                continue
            end
            # Only write the upper-triangular part.
            if v1 > v2
                continue
            end
            idx1 = zero_based ? v1 - 1 : v1
            idx2 = zero_based ? v2 - 1 : v2
            write(io, "$idx1 $idx2 $j_val\n")
        end
    end
end

function run_tamc(method_file::String, instance_file::String, output_file::String,
        tamc_path::String;
        sample_output::Union{Nothing,String}=nothing,
        suspects::Union{Nothing,Vector{String}}=nothing)
    if !isfile(tamc_path)
        error("TAMC binary not found at: $tamc_path")
    end
    # Note: the `--qubo` flag for TAMC is buggy and should not be used.
    cmd = `$tamc_path`
    if sample_output !== nothing
        push!(cmd.exec, "--sample-output=$sample_output")
    end
    if suspects !== nothing
        for s in suspects
            push!(cmd.exec, "--suscepts", s)
        end
    end
    push!(cmd.exec, method_file, instance_file, output_file)

    run(pipeline(cmd, stdout=devnull, stderr=devnull))
end
