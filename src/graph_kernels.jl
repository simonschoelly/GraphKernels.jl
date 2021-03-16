

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

# - dsd
# ************

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

