
using LinearAlgebra: eigvecs
using SparseArrays: SparseVector, sparsevec

# ================================================================
#     AbstractGraphKernel
# ================================================================

abstract type AbstractGraphKernel end

"""
    gramm_matrix(kernel, graphs)
Return a matrix of running the kernel on all pairs of graphs.

"""
function gramm_matrix(kernel::AbstractGraphKernel, graphs::AbstractVector{<:AbstractGraph})

    n = length(graphs)

    G = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in i:n
        v = kernel(graphs[i], graphs[j])
        G[i, j] = v
        G[j, i] = v
    end

    return G
end

# ================================================================
#     BaselineGraphKernel
# ================================================================

# very similar to the No-Graph Baseline Kernel from https://mlai.cs.uni-bonn.de/publications/schulz2019gem.pdf
# but even simpler in that we do not consider any metadata
struct BaselineGraphKernel <: AbstractGraphKernel end

function (::BaselineGraphKernel)(g1, g2)

    return exp(-(ne(g1) - ne(g2))^2 * (Float64(nv(g1)) - Float64(nv(g2)))^2 )
end

# ================================================================
#     ShortestPathGraphKernel
# ================================================================

struct ShortestPathGraphKernel{VK <: AbstractVertexKernel} <: AbstractGraphKernel

    tol::Float64
    vertex_kernel::VK
end

function ShortestPathGraphKernel(;tol=0.0, vertex_kernel=ConstVertexKernel(1.0))

    return ShortestPathGraphKernel(tol, vertex_kernel)
end

function (kernel::ShortestPathGraphKernel)(g1, g2)

    dists1 = _make_dists(g1)
    dists2 = _make_dists(g2)

    # TODO there might be some issues with unsigned types here
    ε = kernel.tol

    # TODO this code runs much faster if we use this function barrier here -
    # but is very unclear why
    return _result1(dists1, dists2, g1, g2, kernel.vertex_kernel)
end

function _make_dists(g)

    dists = floyd_warshall_shortest_paths(g).dists
    verts = vertices(g)
    tm = typemax(eltype(dists))
    dists = [(dist=dists[u, v], u=u, v=v) for u in verts for v in verts if u != v && dists[u, v] != tm]
    sort!(dists, by=t->t.dist)
    return dists
end

function _result1(dists1, dists2, g1, g2, vertex_kernel)

    ds1 = map(t -> t.dist, dists1)
    ds2 = map(t -> t.dist, dists2)

    us1 =  map(t -> t.u, dists1)
    us2 =  map(t -> t.u, dists2)

    vs1 =  map(t -> t.v, dists1)
    vs2 =  map(t -> t.v, dists2)

    result = 0.0

    len1 = length(dists1)
    len2 = length(dists2)

    i2 = 1
    @inbounds for i1 in Base.OneTo(length(ds1))
        d1 = ds1[i1]
        while i2 <= len2 && d1 > ds2[i2]
            i2 += 1
        end
        j2 = i2
        while j2 <= len2 &&  ds2[j2] <= d1
            result += vertex_kernel(g1, us1[i1], g2, us2[j2])
            result += vertex_kernel(g1, vs1[i1], g2, vs2[j2])
            j2 += 1
        end
    end

    return result
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

function (kernel::PyramidMatchGraphKernel)(g1, g2)

    d = kernel.d
    L = kernel.L
    hists1 = _make_hists(g1, d, L)
    hists2 = _make_hists(g2, d, L)

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

function (kernel::NormalizeGraphKernel)(g1, g2)

    k_12 = kernel.inner_kernel(g1, g2)
    k_11 = kernel.inner_kernel(g1, g1)
    k_22 = kernel.inner_kernel(g2, g2)

    return k_12 / sqrt(k_11 * k_22)
end

function gramm_matrix(kernel::NormalizeGraphKernel, graphs::AbstractVector{<:AbstractGraph})

    G = gramm_matrix(kernel.inner_kernel, graphs)
    d = diag(G)
    # TODO compare/benchmark if a loop would not be more performant here
    return G ./ sqrt.(d .* d')
end

