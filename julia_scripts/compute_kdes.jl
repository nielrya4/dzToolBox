#!/usr/bin/env julia
"""
Compute empirical KDEs for multivariate samples using the SedimentSourceAnalysis package.

Usage: julia compute_kdes.jl <input_excel> <output_json>

This script calls create_input_viz_data() from the SourceAnalysisHelpers package
to compute standardized KDEs for all samples and features.
"""

# Precompiled package in julia_scripts/SourceAnalysisHelpers, installed into the
# Julia project by utils.tensor_factorization.initialize_julia_packages()
using SourceAnalysisHelpers
using JSON

function main()
    if !(2 <= length(ARGS) <= 3)
        println(stderr, "Usage: julia compute_kdes.jl <input_excel> <output_json> [kde_options_json]")
        exit(1)
    end

    input_file = ARGS[1]
    output_file = ARGS[2]
    n_samples, bandwidth_overrides = parse_kde_options(get(ARGS, 3, ""))

    try
        println("Computing empirical KDEs for $input_file")

        # Call the create_input_viz_data function to compute KDEs
        results = Dict{String, Any}(create_input_viz_data(input_file; n_samples, bandwidth_overrides))

        # Add status field
        results["status"] = "success"

        # Write results to JSON file
        open(output_file, "w") do f
            JSON.print(f, results, 2)
        end

        println("KDE computation complete. Results written to $output_file")

    catch e
        println(stderr, "Error during KDE computation: $e")
        println(stderr, stacktrace(catch_backtrace()))

        # Write error to JSON
        error_result = Dict(
            "status" => "error",
            "error" => string(e),
            "stacktrace" => string(stacktrace(catch_backtrace()))
        )

        open(output_file, "w") do f
            JSON.print(f, error_result, 2)
        end

        exit(1)
    end
end

main()
