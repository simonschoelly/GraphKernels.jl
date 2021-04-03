
struct ReplacedVertexVals{V, V_VAL, E_VALS, G_VALS, G <: AbstractValGraph} <: AbstractValGraph{V, Tuple{V_VAL}, E_VALS, G_VALS}

    graph::G
    vertexvals::Vector{V_VAL}

    function ReplacedVertexVals(graph::AbstractValGraph, vertexvals::Vector)

        # TODO throw proper error
        @assert nv(graph) == length(vertexvals)

        V = eltype(graph)
        V_VAL = eltype(vertexvals)
        E_VALS = edgevals_type(graph)
        G_VALS = graphvals_type(graph)
        G = typeof(graph)
        return new{V, V_VAL, E_VALS, G_VALS, G}(graph, vertexvals)
    end
end


SimpleValueGraphs.nv(g::ReplacedVertexVals) = nv(g.graph)
SimpleValueGraphs.has_edge(g::ReplacedVertexVals, s::Integer, d::Integer) = has_edge(g.graph, s, d)
SimpleValueGraphs.is_directed(::Type{<:ReplacedVertexVals{V, V_VAL, E_VALS, G_VALS, G}}) where {V, V_VAL, E_VALS, G_VALS, G} = is_directed(G)

SimpleValueGraphs.ne(g::ReplacedVertexVals) = ne(g.graph)

SimpleValueGraphs.get_edgeval(g::ReplacedVertexVals, s::Integer, d::Integer, i::Integer) = get_edgeval(g.graph, s, d, i)

function SimpleValueGraphs.get_vertexval(g::ReplacedVertexVals, v::Integer, i::Integer)

    # TODO might verify that this is a correct vertex
    # TODO might verify that i is 1
    return g.vertexvals[v]
end


