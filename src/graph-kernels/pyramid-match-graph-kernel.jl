
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

    # TODO consider inplace sort!

    # the j-th column of this matrix contains the j-th coordinates
    # of the embedding vectors sorted in ascending order
    ordered_points = sort(embedding; dims=1)

    return ordered_points
end

function apply_preprocessed(kernel::PyramidMatchGraphKernel, points1, points2)

    # TODO the embedding dim might actually be smaller for some graph
    # we should maybe consider some kind of scaling in such a case
    d = min(size(points1, 2), size(points2, 2))
    L = kernel.histogram_levels
    n1 = size(points1, 1)
    n2 = size(points2, 1)

    hist_intersect = zeros(Int64, L + 1)

    # TODO ensure no int overflow, same result on 32 bit platform

    # TODO we don't need to store hist_intersect as a vector
    # TODO add @inbounds
    hist_intersect[1] = d * min(n1, n2)
    for l in 1:L
        cell_boundaries = range(0.0, 1.0, length=2^l + 1)
        for j in 1:d
            i1 = 1
            i2 = 1
            while i1 <= n1
                # TODO is is possible that searchsortedlast is not implemented
                # efficiently on a StepRangeLen
                cell_num = searchsortedlast(cell_boundaries, points1[i1, j])
                # TODO maybe we should verify here, that cell_num is a valid index
                cell_lower = cell_boundaries[cell_num]
                cell_upper = cell_boundaries[cell_num + 1]
                # TODO maybe we need some correction for rounding errors here

                i1 += 1
                num1 = 1
                # the first loop is just for safety we probably don't need it
                while i1 <= n1 && points1[i1, j] < cell_lower
                    i1 += 1
                end
                # count number of vertices from g1 that fall into the bucket
                # specified [cell_lower, cell_upper) along the j-th dimension
                # of the hypercube
                while i1 <= n1 && points1[i1, j] < cell_upper
                    num1 += 1
                    i1 += 1
                end

                # count number of vertices from g2 that fall into that bucket
                # We could also use binary search here
                while i2 <= n2 && points2[i2, j] < cell_lower
                    i2 += 1
                end
                num2 = 0
                while i2 <= n2 && points2[i2, j] < cell_upper
                    num2 += 1
                    i2 += 1
                end
                hist_intersect[l + 1] += min(num1, num2)
            end
        end
    end

    return hist_intersect[L + 1] + sum(l -> (hist_intersect[l+1] - hist_intersect[l+2]) / 2^(L - l), 0:L-1)
end


function _embedding(g::AbstractGraph, embedding_dim::Int)

    evs = eigvecs(adjacency_matrix(g); sortby=x -> -abs(x))
    embedding = abs.(@view evs[:, 1:min(size(evs, 2), embedding_dim)])
    # theoretically clamping is not necessary, this is just a precaution
    # against rounding errors. Clamping to prevfloat(-1.0) should make some calculations easier
    clamp!(embedding, 0.0, prevfloat(1.0))
    return embedding
end


