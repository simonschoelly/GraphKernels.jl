# ================================================================
#     AbstractGraphKernel
# ================================================================

"""
    abstract type AbstractGraphKernel

A kernel function between two graphs.

Subtypes of `AbstractGraphKernel` should implement `preprocessed_form` and `apply_preprocessed`.
When `(k::AbstractGraphKernel)(g1, g2)` is invoked on two graphs, then
```
apply_preprocessed(k, preprocessed_form(k, g1), preprocessed_form(k, g2))
```
is called to calculate the kernel function. Therefore one should implement `preprocessed_form`
that transforms a single graph into a suitable representation and `apply_preprocessed` that
takes the representations for both graphs and calculates the kernel function.

### See also
[`preprocessed_form`](@ref), [`apply_preprocessed`](@ref), [`kernel_matrix`](@ref), [`kernel_matrix_diag`](@ref)

"""
abstract type AbstractGraphKernel end

"""
    preprocessed_form(k::AbstractGraphKernel, g::AbstractGraph) = g

Transform a graph `g` into a suitable form for a graph kernel `k`

When calculating a pairwise kernel matrix for multiple graphs, this preprocessed form
allows us to calculate the transformation only once for each graph, so that we can cache
the result. By default this simply returns `g` without any transformation.

When implementing a custom graph kernel, it might be a good idea to implement this
method.

### See also
[`AbstractGraphKernel`](@ref), [`apply_preprocessed`](@ref)
"""
preprocessed_form(::AbstractGraphKernel, g::AbstractGraph) = g

function (kernel::AbstractGraphKernel)(g1, g2)

    return apply_preprocessed(kernel, preprocessed_form(kernel, g1), preprocessed_form(kernel, g2))
end

## ---------------------------------------------------------------
##       kernelmatrix & kernelmatrix_diag
## ---------------------------------------------------------------

function _map_preprocessed_form(kernel::AbstractGraphKernel, graphs)

    # TODO we should be able to avoid collecting the graphs
    # but currently ThreadX cannot split them otherwise,
    # maybe we can create some wrapper type that is splitable around graphs
    return ThreadsX.map(g -> preprocessed_form(kernel, g), collect(graphs))
end

"""
    kernelmatrix(kernel, graphs)
Return a matrix of running the kernel on all pairs of graphs.

### See also
[`kernelmatrix_diag`](@ref)
"""
function kernelmatrix(kernel::AbstractGraphKernel, graphs)

    pre = _map_preprocessed_form(kernel, graphs)

    # this simply a guard to make the code more type save, maybe we can get
    # rid of it at some point
    return _kernelmatrix_from_preprocessed(kernel, pre)
end

function _kernelmatrix_from_preprocessed(kernel, pre)

    n = length(pre)

    # TODO maybe we should make the matrix only symmetric afterwards
    # so that we avoid false sharing when using multiple threads
    # TODO create some triangle generator instead of allocating a vector
    # TODO apparently ThreadsX can do load balancing so we should consider that here
    G = Matrix{Float64}(undef, n, n)
    indices = [(i, j) for i in 1:n for j in i:n]
    Threads.@threads for idx in indices
        i, j = idx
        @inbounds v = apply_preprocessed(kernel, pre[i], pre[j])
        @inbounds G[i, j] = v
        @inbounds G[j, i] = v
    end

    return G
end

"""
    kernelmatrix_diag(kernel::AbstractGraphKernel, graphs)

Calculate the diagonal of the kernelmatrix matrix of the graphs.

### See also
[`kernelmatrix`](@ref)
"""
function kernelmatrix_diag(kernel::AbstractGraphKernel, graphs)

    n = length(graphs)
    pre = _map_preprocessed_form(kernel, graphs)

    D = Vector{Float64}(undef, n)
    Threads.@threads for i in 1:n
        @inbounds D[i] = apply_preprocessed(kernel, pre[i], pre[i])
    end
    return D
end

"""
    kernelmatrix(kernel::AbstractGraphKernel, graphs1, graphs2)

Calculate a matrix of invoking the kernel on all pairs.
Entry `(i, j)` of the resulting matrix contains `kernel(graphs1[i], graphs2[j]`.
"""
function kernelmatrix(kernel::AbstractGraphKernel, graphs1, graphs2)

    n_rows = length(graphs1)
    n_cols = length(graphs2)

    M = Matrix{Float64}(undef, n_rows, n_cols)

    pre1 = _map_preprocessed_form(kernel, graphs1)
    pre2 = _map_preprocessed_form(kernel, graphs2)

    Threads.@threads for i in 1:n_rows
        for j in 1:n_cols
            @inbounds M[i, j] = apply_preprocessed(kernel, pre1[i], pre2[j])
        end
    end

    return M
end

