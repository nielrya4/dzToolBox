# Shared KDE options for the multivariate scripts.
#
# The Python side passes a JSON string like
#   {"n_samples": 128, "bandwidths": [null, 5.0, null]}
# where "bandwidths" lines up with the measurement (sheet) order and null means
# "use default_bandwidth". Overrides are positional rather than keyed by name
# because sheet titles can be truncated/sanitized when the temp workbook is written.

const KDE_INNER_PERCENTILE = 95
const KDE_ALPHA = 0.9
const KDE_DEFAULT_N_SAMPLES = 128

function parse_kde_options(options_json::AbstractString)
    options = isempty(strip(options_json)) ? Dict{String,Any}() : JSON.parse(options_json)
    n_samples = Int(get(options, "n_samples", KDE_DEFAULT_N_SAMPLES))
    n_samples >= 2 || throw(ArgumentError("n_samples must be at least 2, got $n_samples"))
    raw_overrides = get(options, "bandwidths", nothing)
    overrides = raw_overrides === nothing ? Union{Nothing,Float64}[] :
        Union{Nothing,Float64}[b === nothing ? nothing : Float64(b) for b in raw_overrides]
    for b in overrides
        (b === nothing || b > 0) || throw(ArgumentError("bandwidths must be positive, got $b"))
    end
    return n_samples, overrides
end

# Default bandwidths come from the first sink (matching the original dzgrainalyzer),
# then any user overrides replace them measurement by measurement.
function kde_bandwidths(sinks, overrides)
    sink1 = sinks[begin]
    bandwidths = default_bandwidth.(collect(eachmeasurement(sink1)), KDE_ALPHA, KDE_INNER_PERCENTILE)
    is_default = trues(length(bandwidths))
    for (i, b) in enumerate(overrides)
        if b !== nothing && i <= length(bandwidths)
            bandwidths[i] = b
            is_default[i] = false
        end
    end
    return bandwidths, is_default
end

# Same weights as KernelDensity's default UniformWeights{N} (each 1/N, summing to exactly 1.0,
# so results are bit-identical), but without N in the type. UniformWeights{N} makes Julia
# compile the whole KDE pipeline again for every distinct grain count, which can't be
# precompiled and cost several seconds per run.
struct EqualWeights <: AbstractVector{Float64}
    n::Int
end
Base.size(w::EqualWeights) = (w.n,)
Base.getindex(w::EqualWeights, i::Int) = 1 / w.n
Base.sum(::EqualWeights) = 1.0

# SedimentSourceAnalysis's make_densities(::Sink; ...) with the KDE weights swapped for EqualWeights.
function make_sink_densities(sink; bandwidths, inner_percentile)
    density_estimates = Vector{UnivariateKDE}(undef, length(bandwidths))
    for (i, (measurement_values, b)) in enumerate(zip(eachmeasurement(sink), bandwidths))
        measurement_values = filter_inner_percentile(measurement_values, inner_percentile)
        density_estimates[i] = kde(measurement_values; bandwidth=b, weights=EqualWeights(length(measurement_values)))
    end
    return density_estimates
end

# Equivalent to DensityTensor(standardize_KDEs(raw_densities; n_samples)..., sinks), which
# splats the per-sink lists into tuples and so recompiles for every distinct number of sinks.
# Standardizing one measurement at a time with the single-measurement standardize_KDEs does
# the same arithmetic without that.
function standardized_density_tensor(raw_densities, measurement_names, n_samples)
    n_sinks = length(raw_densities)
    n_measurements = length(measurement_names)
    data = Array{Float64,3}(undef, n_sinks, n_measurements, n_samples)
    domains = Vector{StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int}}(undef, n_measurements)
    for m in 1:n_measurements
        densities, domains[m] = standardize_KDEs([raw_densities[s][m] for s in 1:n_sinks]; n_samples)
        for s in 1:n_sinks
            data[s, m, :] = densities[s]
        end
    end
    return DensityTensor(data, domains, measurement_names), domains
end

# Returns the DensityTensor plus a JSON-friendly summary of the parameters used.
function build_density_tensor(sinks, n_samples, overrides)
    bandwidths, is_default = kde_bandwidths(sinks, overrides)
    raw_densities = make_sink_densities.(sinks; bandwidths, inner_percentile=KDE_INNER_PERCENTILE)
    densitytensor, domains = standardized_density_tensor(raw_densities, getmeasurements(sinks[begin]), n_samples)
    kde_parameters = Dict(
        "n_samples" => n_samples,
        "measurements" => [string(m) for m in getmeasurements(sinks[begin])],
        "bandwidths" => collect(Float64, bandwidths),
        "bandwidth_is_default" => collect(Bool, is_default),
    )
    return densitytensor, domains, kde_parameters
end
