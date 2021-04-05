
using LinearAlgebra: eigvecs
using SparseArrays: SparseVector, sparsevec

# ================================================================
#     AbstractGraphKernel
# ================================================================

abstract type AbstractGraphKernel end

preprocessed_form(::AbstractGraphKernel, g::AbstractGraph) = g

function (kernel::AbstractGraphKernel)(g1, g2)

    return apply_preprocessed(kernel, preprocessed_form(kernel, g1), preprocessed_form(kernel, g2))
end

"""
    kernelmatrix(kernel, graphs)
Return a matrix of running the kernel on all pairs of graphs.

"""
function kernelmatrix(kernel::AbstractGraphKernel, graphs)

    n = length(graphs)

    # TODO we should be able to avoid collecting the graphs
    # but currently ThreadX cannot split them otherwise,
    # maybe we should have a custom map function
    pre = ThreadsX.map(g -> preprocessed_form(kernel, g), collect(graphs))
    #pre = map(g -> preprocessed_form(kernel, g), collect(graphs))

    # TODO maybe we should make the matrix only symmetric afterwards
    # so that we avoid false sharing when using multiple threads
    # TODO create some triangle generator instead of allocating a vector
    # TODO apparently ThreadsX can do load balancing so we should consider that here

    return f__(kernel, pre, n)
end

function f__(kernel, pre, n)

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

Calculate the diagonal of the gramm matrix of the kernel on graphs.
"""
function kernelmatrix_diag(kernel::AbstractGraphKernel, graphs)

    n = length(graphs)
    pre = ThreadsX.map(g -> preprocessed_form(kernel, g), collect(graphs))

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

    pre1 = ThreadsX.map(g -> preprocessed_form(kernel, g), collect(graphs1))
    pre2 = ThreadsX.map(g -> preprocessed_form(kernel, g), collect(graphs2))

    Threads.@threads for i in 1:n_rows
        for j in 1:n_cols
            @inbounds M[i, j] = apply_preprocessed(kernel, pre1[i], pre2[j])
        end
    end

    return M
end

# ================================================================
#     BaselineGraphKernel
# ================================================================

# very similar to the No-Graph Baseline Kernel from https://mlai.cs.uni-bonn.de/publications/schulz2019gem.pdf
# but even simpler in that we do not consider any metadata
struct BaselineGraphKernel <: AbstractGraphKernel end

function apply_preprocessed(::BaselineGraphKernel, g1, g2)

    return exp(-(ne(g1) - ne(g2))^2 * (Float64(nv(g1)) - Float64(nv(g2)))^2 )
end

# ================================================================
#     ShortestPathGraphKernel
# ================================================================


"""
    ShortestPathGraphKernel <: AbstractGraphKernel

A graph kernel that compares two graphs `g` and `g'` by comparing all pairs
of vertices `(u, v)` of the `g` and `(u', v')` of `g'` if their shortest distance
is smaller than `tol`. In that case, the vertices `u` and `u'`, as well as `v`, `v'` are
compared with `vertex_kernel`.

# Keywords
- `tol=0.0`: Only pairs of vertices where the shortest distance is at most `tol` are
    compared.
- `dist_key=:`: The key for the edge values to compute the shortest distance with. Can be either
    an `Integer` or a `Symbol` for a key to a specific edge value, `nothing` to use a default distance
    of `1` for each edge, or `:` in which case the default edge weight for that graph type
    is used.
- `vertex_kernel=ConstVertexKernel(1.0)`: The kernel used to compare two vertices.

# References
[Borgwardt, K. M., & Kriegel, H. P.: Shortest-path kernels on graphs](https://www.dbs.ifi.lmu.de/~borgward/papers/BorKri05.pdf)
"""
struct ShortestPathGraphKernel{VK <: AbstractVertexKernel} <: AbstractGraphKernel

    tol::Float64
    vertex_kernel::VK
    dist_key::Union{Int, Symbol, Colon, Nothing}
end

function ShortestPathGraphKernel(;tol=0.0, vertex_kernel=ConstVertexKernel(1.0), dist_key=Colon())

    return ShortestPathGraphKernel(tol, vertex_kernel, dist_key)
end

function preprocessed_form(kernel::ShortestPathGraphKernel, g::AbstractGraph)

    dists = _make_dists(g, kernel.dist_key)

    ds = map(t -> t.dist, dists)
    us =  map(t -> t.u, dists)
    vs =  map(t -> t.v, dists)

    return (g=g, ds=ds, us=us, vs=vs)
end

function apply_preprocessed(kernel::ShortestPathGraphKernel, pre1, pre2)

    g1, ds1, us1, vs1 = pre1
    g2, ds2, us2, vs2 = pre2

    # TODO there might be some issues with unsigned types here
    ε = kernel.tol
    vertex_kernel = kernel.vertex_kernel

    result = 0.0

    len1 = length(ds1)
    len2 = length(ds2)

    i2 = 1
    @inbounds for i1 in Base.OneTo(length(ds1))
        d1 = ds1[i1]
        while i2 <= len2 && d1 > (ds2[i2] + ε)
            i2 += 1
        end
        j2 = i2
        while j2 <= len2 && ds2[j2] <= (d1 + ε)
            result += vertex_kernel(g1, us1[i1], g2, us2[j2])
            result += vertex_kernel(g1, vs1[i1], g2, vs2[j2])
            j2 += 1
        end
    end

    return result

end

function _make_dists(g, dist_key)

    dists = if dist_key === Colon()
        floyd_warshall_shortest_paths(g).dists
    elseif dist_key === nothing
        floyd_warshall_shortest_paths(g, LightGraphs.DefaultDistance(nv(g))).dists
    else
        floyd_warshall_shortest_paths(g, weights(g, dist_key)).dists
    end

    verts = vertices(g)
    tm = typemax(eltype(dists))
    dists_list = [(dist=dists[u, v], u=u, v=v) for u in verts for v in verts if u != v && dists[u, v] != tm]
    sort!(dists_list, by=t->t.dist)
    return dists_list
end

# ================================================================
#     PyramidMatchGraphKernel
# ================================================================

"""
    PyramidMatchGraphKernel <: AbstractGraphKernel
"""
struct PyramidMatchGraphKernel <: AbstractGraphKernel

    d::Int
    L::Int
end

function _embedding(g::AbstractGraph, d::Int)

    n = nv(g)
    A = zeros(max(d, n), max(d, n))
    A[1:n, 1:n] = adjacency_matrix(g)
    # TODO should we scale?
    embedding = eigvecs(A; sortby=x -> -abs(x))[:, 1:d]
    # theoretically clamping is not necessary, this is just a precaution
    # against rounding errors
    clamp!(embedding, -1.0, 1.0)
    return embedding
end

function _make_hists(g::AbstractGraph, d::Int, L::Int)

    # TODO consider using a 3 dimensional SparseArray instead
    # or just calculate the whole histogram on the fly

    # nv(g) x d matrix
    embedding = _embedding(g, d)

    # hist[l, d][i] contains number of points in the i-th bucket
    # for layer l and dimension d
    hists = Matrix{SparseVector{Int, Int}}(undef, L + 1, d)
    for l in 0:L
        for dd in 1:d
            counts = zeros(Int, 2^l)
            for i in 1:2^l
                lo = -1.0 + (i - 1) * 2 / 2^l
                hi = -1.0 + i * 2 / 2^l

                for x in embedding[:, dd]
                    if lo <= x < hi # this currently misses 1.0
                        counts[i] += 1
                    end
                end
            end
            hists[l + 1, dd] = sparsevec(counts)
        end
    end
    return hists
end

function _I(hists1, hists2, d, l)

    return sum(dd -> sum(min.(hists1[l, dd], hists2[l, dd])), 1:d)
end

function preprocessed_form(kernel::PyramidMatchGraphKernel, g::AbstractGraph)

    d = kernel.d
    L = kernel.L
    return _make_hists(g, d, L)
end

function apply_preprocessed(kernel::PyramidMatchGraphKernel, hists1, hists2)

    d = kernel.d
    L = kernel.L
    return _I(hists1, hists2, d, L + 1) +
        sum(l -> 1 / 2^(L - l) * (_I(hists1, hists2, d, l + 1) -
                                  _I(hists1, hists2, d, l + 2)), 0:L-1)
end

# ================================================================
#     NormalizeGraphKernel
# ================================================================

"""
    NormalizeGraphKernel <: AbstractGraphKernel
"""
struct NormalizeGraphKernel{IK<:AbstractGraphKernel} <: AbstractGraphKernel

    inner_kernel::IK
end

function preprocessed_form(kernel::NormalizeGraphKernel, g::AbstractGraph)

    inner = kernel.inner_kernel
    pre_inner = preprocessed_form(inner, g)
    k_ii = apply_preprocessed(inner, pre_inner, pre_inner)
    return (k_ii, pre_inner)
end

function apply_preprocessed(kernel::NormalizeGraphKernel, pre1, pre2)

    inner = kernel.inner_kernel
    k_11, pre_inner1 = pre1
    k_22, pre_inner2 = pre2

    k_12 = apply_preprocessed(inner, pre_inner1, pre_inner2)

    return k_12 / sqrt(k_11 * k_22)
end

