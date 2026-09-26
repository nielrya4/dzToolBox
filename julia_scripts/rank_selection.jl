#!/usr/bin/env julia

"""
Call original DZ Grainalyzer rank selection using SedimentSourceAnalysis

Uses linear breakpoint analysis (elbow method) to find optimal rank.
The work is done by select_rank() in the SourceAnalysisHelpers package.

Usage:
    julia rank_selection.jl <input_excel> <min_rank> <max_rank> <output_json> [kde_options_json]
"""

using SourceAnalysisHelpers

# Simple JSON writer - no external package needed
function write_json(io::IO, obj::AbstractDict)
    print(io, "{")
    first_item = true
    for (k, v) in obj
        if !first_item
            print(io, ",")
        end
        first_item = false
        print(io, "\"", k, "\":")
        write_json_value(io, v)
    end
    print(io, "}")
end

function write_json_value(io::IO, v::AbstractString)
    print(io, "\"", replace(string(v), "\"" => "\\\""), "\"")
end

function write_json_value(io::IO, v::Number)
    print(io, v)
end

function write_json_value(io::IO, v::AbstractArray)
    print(io, "[")
    for (i, item) in enumerate(v)
        if i > 1
            print(io, ",")
        end
        write_json_value(io, item)
    end
    print(io, "]")
end

write_json_value(io::IO, v::AbstractDict) = write_json(io, v)

function write_json_value(io::IO, v::Any)
    print(io, "\"", string(v), "\"")
end

# Main execution
function main()
    if length(ARGS) < 4
        println("Usage: julia rank_selection.jl <input_excel> <min_rank> <max_rank> <output_json> [kde_options_json]")
        exit(1)
    end

    input_path = ARGS[1]
    min_rank = parse(Int, ARGS[2])
    max_rank = parse(Int, ARGS[3])
    output_path = ARGS[4]
    n_samples, bandwidth_overrides = parse_kde_options(get(ARGS, 5, ""))

    try
        output = select_rank(input_path; n_samples, bandwidth_overrides)

        # Write JSON
        open(output_path, "w") do f
            write_json(f, output)
        end

        println("Success! Results written to $output_path")
        exit(0)

    catch e
        # Write error to output
        output = Dict(
            "status" => "error",
            "error" => string(e),
            "stacktrace" => string(stacktrace(catch_backtrace()))
        )

        open(output_path, "w") do f
            write_json(f, output)
        end

        println(stderr, "Error: $e")
        for line in stacktrace(catch_backtrace())
            println(stderr, "  ", line)
        end
        exit(1)
    end
end

# Run main
main()
