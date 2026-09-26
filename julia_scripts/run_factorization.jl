#!/usr/bin/env julia
"""
Run tensor factorization with a custom rank using the original dzgrainalyzer code.

Usage: julia run_factorization.jl <input_excel> <rank> <output_json> [kde_options_json]

This script calls rank_sources_custom_rank() from the SourceAnalysisHelpers package
which matches the original dzgrainalyzer implementation exactly.
"""

# Precompiled package in julia_scripts/SourceAnalysisHelpers, installed into the
# Julia project by utils.tensor_factorization.initialize_julia_packages()
using SourceAnalysisHelpers
using JSON

function main()
    if !(3 <= length(ARGS) <= 4)
        println(stderr, "Usage: julia run_factorization.jl <input_excel> <rank> <output_json> [kde_options_json]")
        exit(1)
    end

    input_file = ARGS[1]
    rank = parse(Int, ARGS[2])
    output_file = ARGS[3]
    n_samples, bandwidth_overrides = parse_kde_options(get(ARGS, 4, ""))

    try
        println("Running factorization with rank=$rank on $input_file")

        # Call the original dzgrainalyzer function
        results = rank_sources_custom_rank(input_file, rank; n_samples, bandwidth_overrides)

        # Add status field
        results["status"] = "success"

        # Write results to JSON file
        open(output_file, "w") do f
            JSON.print(f, results, 2)
        end

        println("Factorization complete. Results written to $output_file")

    catch e
        println(stderr, "Error during factorization: $e")
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