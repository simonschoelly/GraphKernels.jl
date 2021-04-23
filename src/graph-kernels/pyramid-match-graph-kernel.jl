
"""
    PyramidMatchGraphKernel <: AbstractGraphKernel

A graph kernel that compares vertex embeddings of graph using pyramid match.

For each graph, the vertices are mapped into a hypercube of dimension `embedding_dim`.
The hypercube is the split into buckets of certain granularity, controlled by `histogram_levels`,
and the number of vertices in a bucket for both graphs are compared.
If vertex labels are provided, only vertices with the same label in a bucket are compared.

# Keywords
- `embedding_dim=6`: The number of dimensions of the space where the vertices are embedded.
    This is an upper bound - in some cases only an embedding of lower dimension is created.
- `histogram_levels=4`: The number of levels for which histograms are compared. For
    each level `l ∈ 0:histogram_levels`, the embeddings are split into 2^l buckets and
    the number of vertices in that bucket are compared separately.
- `vertex_labels`: By which vertex labels the vertices are grouped together. The embedding
    is created on all vertices, but then the histograms are separately compared for each
    group of vertex labels. Finally the results of each group are added together.
    Either `Auto`, `:` (all), or a tuple of vertex value keys.

# References
<https://ojs.aaai.org/index.php/AAAI/article/view/10839>
"""
struct PyramidMatchGraphKernel{VL <: LabelsType} <: AbstractGraphKernel

    embedding_dim::Int
    histogram_levels::Int
    vertex_labels::VL

    function PyramidMatchGraphKernel(;embedding_dim::Integer=6, histogram_levels::Integer=4, vertex_labels=Auto())

        embedding_dim >= 1 || throw(DomainError(embedding_dim, "embedding_dim must be >= 1"))
        histogram_levels >= 0 || throw(DomainError(embedding_dim, "histogram_levels must be >= 0"))

        return new{typeof(vertex_labels)}(Int(embedding_dim), Int(histogram_levels), vertex_labels)
    end
end

function preprocessed_form(kernel::PyramidMatchGraphKernel, g::AbstractGraph)

    embedding = _embedding(g, kernel.embedding_dim)
    d = size(embedding, 2) # d might sometimes be smaller than embedding_dim

    vertex_labels = _labels(kernel.vertex_labels, vertexvals_type(g))

    # TODO, we should deduce the correct type instead of using Any
    vertex_class_sizes = counter(Any)
    for v in vertices(g)
        class_key = tuple((get_vertexval(g, v, i) for i in vertex_labels)...)
        inc!(vertex_class_sizes, class_key)
    end

    # The rows of embedding correspond to vertices. We group vertices together
    # by their labels and associate a sub matrix of the corresponding rows
    # with each label class
    insert_row_index = Dict{Any, Int}()
    points_by_vertex_class = Dict{Any, Matrix{Float64}}()
    for (k, n_rows) in vertex_class_sizes
        insert_row_index[k] = 0
        points_by_vertex_class[k] = Matrix{Float64}(undef, n_rows, d)
    end

    for (v, row) in enumerate(eachrow(embedding))
        k = tuple((get_vertexval(g, v, i) for i in vertex_labels)...)
        row_index = (insert_row_index[k] += 1)
        points_by_vertex_class[k][row_index, :] = row
    end

    for points in values(points_by_vertex_class)
        # the j-th column of this matrix contains the j-th coordinates
        # of the embedding vectors sorted in ascending order
        sort!(points; dims=1)
    end

    return points_by_vertex_class
end

function apply_preprocessed(kernel::PyramidMatchGraphKernel, points_by_class_1, points_by_class_2)

    L = kernel.histogram_levels
    result = 0.0

    for (class, points1) in points_by_class_1

        points2 = get(points_by_class_2, class, nothing)
        points2 == nothing && continue

        # TODO the embedding dim might actually be smaller for some graph
        # we should maybe consider some kind of scaling in such a case
        d = min(size(points1, 2), size(points2, 2))
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

        result += (hist_intersect[L + 1] + sum(l -> (hist_intersect[l+1] - hist_intersect[l+2]) / 2^(L - l), 0:L-1))

    end
    return result
end


function _embedding(g::AbstractGraph, embedding_dim::Int)

    evs = eigvecs(adjacency_matrix(g); sortby=x -> -abs(x))
    embedding = abs.(@view evs[:, 1:min(size(evs, 2), embedding_dim)])
    # theoretically clamping is not necessary, this is just a precaution
    # against rounding errors. Clamping to prevfloat(-1.0) should make some calculations easier
    clamp!(embedding, 0.0, prevfloat(1.0))
    return embedding
end


