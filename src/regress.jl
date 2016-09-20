### functions pertaining to basic regression in MendelBase

# these are the regression models currently supported
# the user must chooses one of these
# regress will check against this list and throw an error for unsupported models 
const acceptable_models = ("linear", "logistic", "Poisson")

export regress

"Subroutine of `regress` to perform linear regression."
function _regress_linear!{T <: AbstractFloat}(
    s :: Vector{T}, # s = - X' * (y - X*β)
    F :: Matrix{T}, # F = X' * X
    r :: Vector{T}, # r = y - X*β
    X :: Matrix{T},
    y :: Vector{T},
    β :: Vector{T},
)
    # compute score
    _compute_linear_score!(s, F, r, X, y, β)

    # solve the linear system
    b = F \ s

    # recompute residuals r = y - X*b
    BLAS.gemv!('N', -one(T), X, b, zero(T), r)
    BLAS.axpy!(one(T), y,  r)

    # final objective function value
    n   = length(r)
    obj = - n / 2 * log(sumabs2(r) / n) - n / 2

    return (b, obj)
end

"Subroutine of `regress` to perform logistic regression."
function _regress_logistic!{T <: AbstractFloat}(
    s :: Vector{T}, # s = - X' * (y - exp(-X*β))
    F :: Matrix{T}, # F = X' * X
    r :: Vector{T}, # r = y - exp(X*β)
    w :: Vector{T}, # w = 1 ./ (1 + exp(-X*β))
    X :: Matrix{T},
    y :: Vector{T},
    β :: Vector{T};
    tol      :: T = convert(eltype(s), 1e-6),
    max_iter :: Int = 10, # max iterations allowed in scoring
    max_step :: Int = 3   # max steps allowed in step halving
)
    # first estimate the intercept
    β[1] = log(mean(y) / (one(T) - mean(y)))

    # initialize objective function values
    obj     = zero(T)
    old_obj = -Inf 

    # loop to estimate parameters using scoring
    for iter = 1:max_iter

        # compute the score and information matrix
        # this will overwrite current score and observed information matrix
        ## clamp!(r, -20*zero(T), 20*zero(T))
        _compute_logistic_score!(s, F, r, w, X, y, β)
        ## clamp!(r, -20*one(T), 20*one(T))

        # compute the scoring increment
        Δβ = F \ s

        # increment the estimate β = β + Δβ
        BLAS.axpy!(one(T), Δβ, β)

        # compute the loglikelihood (objective)
        BLAS.gemv!('N', one(T), X, β, zero(T), r)
##        clamp!(r, -20*one(T), 20*one(T))
        obj = logistic_loglik(r, y)

        # backtrack to produce an increase in the loglikelihood
        steps = 0
        backtrack_scale = one(T)
        while obj < old_obj && steps <= max_step

            # monitor the number of backtracking steps
            steps += 1

            # halve increment
            # β = β - (0.5)^steps * Δβ
            backtrack_scale /= 2
            BLAS.axpy!(-backtrack_scale, Δβ, β)

            # recompute r = X*β
            BLAS.gemv!('N', one(T), X, β, zero(T), r)
##            clamp!(r, -20*zero(T), 20*zero(T))

            # compute the loglikelihood under the appropriate model.
            obj = logistic_loglik(r, y)
        end

        # guard against descent errors
        old_obj > obj && throw(error("Descent failure at iteration $(iter) after $steps backtracking steps!!\nOld_Obj: $(old_obj)\nObj: $(obj)\n"))

        # converged?
        # if not, then save objective and score again
        converged = iter > 1 && abs(obj - old_obj) < tol * (abs(old_obj) + one(T))
        converged && return β, obj
        old_obj = obj
    end

    warn("Maximum iterations $max_iter reached, returning result...")

    return β, obj
end

"Subroutine of `regress` to perform Poisson regression."
function _regress_poisson!{T <: AbstractFloat}(
    s :: Vector{T}, # s = - X' * (y - exp(X*β))
    F :: Matrix{T}, # F = X' * diagm(w)* X
    r :: Vector{T}, # r = y - exp(X*β)
    w :: Vector{T}, # w = exp(X*β) 
    X :: Matrix{T},
    y :: Vector{T},
    β :: Vector{T};
    tol      :: T   = convert(eltype(s), 1e-6),
    max_iter :: Int = 10, # max iterations allowed in scoring
    max_step :: Int = 3,  # max steps allowed in step halving
)
    # first estimate the intercept
    β[1] = log(mean(y))

    # initialize objective function values
    obj     = zero(T)
    old_obj = -Inf

    # loop to estimate parameters using scoring
    for iter = 1:max_iter

        # compute the score and information matrix
        # this will overwrite current score and observed information matrix
        ## clamp!(z, -20*zero(T), 20*zero(T))
        _compute_poisson_score!(s, F, r, w, X, y, β)
        ## clamp!(r, -20*one(T), 20*one(T))

        # compute the scoring increment
        Δβ = F \ s
        BLAS.axpy!(one(T), Δβ, β)

        # compute the loglikelihood (objective)
        BLAS.gemv!('N', one(T), X, β, zero(T), r)
        obj = poisson_loglik(r, y)

        # backtrack to produce an increase in the loglikelihood
        steps = 0
        backtrack_scale = one(T)
        while obj < old_obj && steps <= max_step

            # monitor the number of backtracking steps
            steps += 1

            # halve increment
            # β = β - (0.5)^steps * Δβ
            backtrack_scale /= 2
            BLAS.axpy!(-backtrack_scale, Δβ, β)

            # recompute r = X*β
            BLAS.gemv!('N', one(T), X, β, zero(T), r)
            ## clamp!(r, -20*zero(T), 20*zero(T))

            # compute the loglikelihood under the appropriate model.
            obj = poisson_loglik(r, y)
        end

        # guard against descent errors
        old_obj > obj && throw(error("Descent failure at iteration $(iter)!\n"))

        # converged?
        # if not, then save objective and score again
        converged = iter > 1 && abs(obj - old_obj) < tol * (abs(old_obj) + one(T))
        converged && return (β, obj)
        old_obj = obj
    end

    warn("Maximum iterations $max_iter reached, returning result...")
    return β, obj
end

"""
    regress(X, y, model) -> β::Vector, obj::Float

Using a design matrix `X` and response vector `y`, perform one of `linear`, `logistic`, or `Poisson` regression
with a canonical link function. The choice of link is specified in the `ASCIIString` argument `model`.
Returns the estimated parameter vector `β` and the optimum `obj` of the loss function.
"""
function regress{T <: AbstractFloat}(
    X        :: Matrix{T},
    y        :: Vector{T},
    model    :: ASCIIString;
    tol      :: T   = convert(eltype(X), 1e-6),
    max_iter :: Int = 10, # max iterations allowed in scoring
    max_step :: Int = 3  # max steps allowed in step halving
)
    # problem dimension?
    n,p = size(X)

    # handle input errors
    if model ∉ acceptable_models 
        throw(ArgumentError(
        "The only model choices are 'linear', 'logistic', and 'Poisson'.\n \n"))
    end
    @assert n == length(y)

    s   = zeros(T, p)    # score vector
    F   = zeros(T, p, p) # observed information matrix
    β   = zeros(T, p)    # parameter vector
    r   = zeros(T, n)    # residual vector
    w   = zeros(T, n)    # weight vector
    obj = zero(T)

    # compute regression based on model type specified in `model`
    # all three subroutines return (β, obj)
    model == "linear"   && return _regress_linear!(s, F, r, X, y, β)
    model == "logistic" && return _regress_logistic!(s, F, r, w, X, y, β, tol=tol, max_iter=max_iter, max_step=max_step)
    model == "Poisson"  && return _regress_poisson!(s, F, r, w, X, y, β, tol=tol, max_iter=max_iter, max_step=max_step)

    return β, obj
end
