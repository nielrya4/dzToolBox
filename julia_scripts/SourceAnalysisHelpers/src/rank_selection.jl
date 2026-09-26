# Rank selection: factorize at every rank from 1 to min(n_sinks, 10)+1 and pick the elbow
# of the relative error curve with two-line breakpoint analysis.

"""
Two-line Breakpoint Analysis (official algorithm from BlockTensorFactorization.jl)

Fits a piecewise linear model with two segments meeting at a breakpoint:
    f_z(x; a, b, c) = a + b*(min(x,z) - z) + c*(max(x,z) - z)

where z is the breakpoint, a is the value at the breakpoint, b is the slope
before the breakpoint, and c is the slope after the breakpoint.

For each candidate breakpoint z, solves for optimal (a, b, c) via least squares
and computes the squared error. Returns the breakpoint that minimizes the error.

Reference: https://mpf-optimization-laboratory.github.io/BlockTensorFactorization.jl/dev/tutorial/rankestimation/#Two-line-Breakpoint

Returns: (best_breakpoint_index, breakpoint_scores)
"""
function breakpoint_model_coefficients(xs::Vector, ys::Vector, z)
    # Build design matrix M for the model:
    # f_z(x) = a + b*(min(x,z) - z) + c*(max(x,z) - z)
    n = length(xs)
    M = zeros(n, 3)
    for i in 1:n
        M[i, 1] = 1.0                      # coefficient for a
        M[i, 2] = min(xs[i], z) - z        # coefficient for b (left slope)
        M[i, 3] = max(xs[i], z) - z        # coefficient for c (right slope)
    end

    # Solve least squares: coeffs = M \ ys
    coeffs = M \ ys
    return coeffs
end

function breakpoint_error(xs::Vector, ys::Vector, z)
    # Get optimal coefficients for this breakpoint
    coeffs = breakpoint_model_coefficients(xs, ys, z)
    a, b, c = coeffs

    # Compute predictions
    y_pred = [a + b*(min(x, z) - z) + c*(max(x, z) - z) for x in xs]

    # Return squared L2 error
    return sum((ys .- y_pred).^2)
end

function find_linear_breakpoint(x::Vector, y::Vector)
    n = length(x)
    if n < 3
        return 1, zeros(n)
    end

    # Test each point as a candidate breakpoint
    # (excluding first and last points)
    errors = zeros(n)
    errors[1] = Inf
    errors[n] = Inf

    for k in 2:(n-1)
        z = x[k]  # Use actual x value as breakpoint
        errors[k] = breakpoint_error(x, y, z)
    end

    # Find breakpoint with minimum error
    best_idx = argmin(errors)

    # Convert errors to "breakpoint scores" (higher = better)
    # Use max_error - error so higher scores indicate better breakpoints
    max_error = maximum(errors[2:end-1])
    breakpoint_scores = zeros(n)
    for k in 2:(n-1)
        breakpoint_scores[k] = max_error - errors[k]
    end

    return best_idx, breakpoint_scores
end

function select_rank(path::String; n_samples=KDE_DEFAULT_N_SAMPLES, bandwidth_overrides=Union{Nothing,Float64}[])
    sinks = read_raw_data(path)
    densitytensor, domains, kde_parameters = build_density_tensor(sinks, n_samples, bandwidth_overrides)
    Y = array(densitytensor)

    # Ranks to test (EXACTLY like original: 1:min(n_samples, 10)+1)
    ranks = 1:min(size(Y)[1], 10)+1
    println("Using rank range: $ranks (like original DZ Grainalyzer)")
    maxiter = 6000
    tol = 1e-5

    Cs, Fs, all_rel_errors = ([] for _ in 1:3)

    # Normalize fibers (same as original)
    Y_fibres = eachslice(Y, dims=(1, 2))
    Y_fibres ./= sum.(Y_fibres)

    for rank in ranks
        C, F, rel_errors, norm_grad, dist_Ncone = nnmtf(
            Y, rank;
            projection=:nnscale,
            maxiter=maxiter,
            tol=tol,
            rescale_Y=false
        )
        push!(Cs, C)
        push!(Fs, F)
        push!(all_rel_errors, rel_errors)
    end

    relative_errors = map(x -> x[end], all_rel_errors)

    # Exclude last rank from selection (same as original)
    if length(ranks) > 1
        ranks_for_selection = collect(ranks)[1:end-1]
        relative_errors_for_selection = relative_errors[1:end-1]
    else
        ranks_for_selection = collect(ranks)
        relative_errors_for_selection = relative_errors
    end

    best_rank_idx, breakpoint_scores = find_linear_breakpoint(
        Float64.(ranks_for_selection),
        Float64.(relative_errors_for_selection)
    )
    breakpoint_scores_for_selection = breakpoint_scores[1:length(ranks_for_selection)]
    best_rank = ranks_for_selection[best_rank_idx]

    # Calculate R²
    C_best = Cs[best_rank_idx]
    F_best = Fs[best_rank_idx]
    reconstruction = zeros(size(Y))
    for i in 1:size(Y, 1)
        for k in 1:best_rank
            reconstruction[i, :, :] .+= C_best[i, k] * F_best[k, :, :]
        end
    end

    ss_res = sum((Y - reconstruction).^2)
    ss_tot = sum((Y .- mean(Y)).^2)
    r2 = 1 - (ss_res / ss_tot)

    println("Best rank: $best_rank (R² = $(round(r2, digits=4)))")

    # Return truncated arrays (excluding last rank) for visualization
    return Dict(
        "status" => "success",
        "ranks" => ranks_for_selection,
        "relative_errors" => relative_errors_for_selection,
        "breakpoint_scores" => breakpoint_scores_for_selection,
        "best_rank" => best_rank,
        "r2" => r2,
        "sink_names" => [string(s) for s in getsourcenames(densitytensor)],
        "measurement_names" => [string(m) for m in getmeasurements(densitytensor)],
        "kde_parameters" => kde_parameters
    )
end
