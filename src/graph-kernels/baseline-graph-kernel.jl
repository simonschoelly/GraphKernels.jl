#
# very similar to the No-Graph Baseline Kernel from https://mlai.cs.uni-bonn.de/publications/schulz2019gem.pdf
# but even simpler in that we do not consider any metadata
struct BaselineGraphKernel <: AbstractGraphKernel end

function apply_preprocessed(::BaselineGraphKernel, g1, g2)

    return exp(-( (ne(g1) - ne(g2))^2 + (Float64(nv(g1)) - Float64(nv(g2)))^2) )
end

