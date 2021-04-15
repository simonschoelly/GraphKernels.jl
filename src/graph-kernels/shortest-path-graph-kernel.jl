
using LinearAlgebra: eigvecs
using SparseArrays: SparseVector, sparsevec


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


