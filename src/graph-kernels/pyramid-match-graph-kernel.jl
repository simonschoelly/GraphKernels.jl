
"""
    PyramidMatchGraphKernel <: AbstractGraphKernel
"""
struct PyramidMatchGraphKernel <: AbstractGraphKernel

    embedding_dim::Int
    histogram_levels::Int

    function PyramidMatchGraphKernel(;embedding_dim::Integer=6, histogram_levels::Integer=4)

        embedding_dim >= 1 || throw(DomainError(embedding_dim, "embedding_dim must be >= 1"))
        histogram_levels >= 0 || throw(DomainError(embedding_dim, "histogram_levels must be >= 0"))

        return new(embedding_dim, histogram_levels)
    end
end

function preprocessed_form(kernel::PyramidMatchGraphKernel, g::AbstractGraph)

    d = kernel.embedding_dim
    L = kernel.histogram_levels

    # TODO consider using a 3 dimensional SparseArray instead
    # or just calculate the whole histogram on the fly

    # nv(g) x embedding_dim matrix
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

function apply_preprocessed(kernel::PyramidMatchGraphKernel, hists1, hists2)

    d = kernel.embedding_dim
    L = kernel.histogram_levels
    return _I(hists1, hists2, d, L + 1) +
        sum(l -> 1 / 2^(L - l) * (_I(hists1, hists2, d, l + 1) -
                                  _I(hists1, hists2, d, l + 2)), 0:L-1)
end


function _embedding(g::AbstractGraph, embedding_dim::Int)

    n = nv(g)
    A = zeros(max(embedding_dim, n), max(embedding_dim, n))
    A[1:n, 1:n] = adjacency_matrix(g)
    # TODO should we scale?
    embedding = eigvecs(A; sortby=x -> -abs(x))[:, 1:embedding_dim]
    # theoretically clamping is not necessary, this is just a precaution
    # against rounding errors
    clamp!(embedding, -1.0, 1.0)
    return embedding
end

function _I(hists1, hists2, embedding_dim, level)

    return sum(dd -> sum(min.(hists1[level, dd], hists2[level, dd])), 1:embedding_dim)
end


