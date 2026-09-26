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

# Returns the DensityTensor plus a JSON-friendly summary of the parameters used.
function build_density_tensor(sinks, n_samples, overrides)
    bandwidths, is_default = kde_bandwidths(sinks, overrides)
    raw_densities = make_densities.(sinks; bandwidths, inner_percentile=KDE_INNER_PERCENTILE)
    densities, domains = standardize_KDEs(raw_densities; n_samples)
    densitytensor = DensityTensor(densities, domains, sinks)
    kde_parameters = Dict(
        "n_samples" => n_samples,
        "measurements" => [string(m) for m in getmeasurements(sinks[begin])],
        "bandwidths" => collect(Float64, bandwidths),
        "bandwidth_is_default" => collect(Bool, is_default),
    )
    return densitytensor, domains, kde_parameters
end
