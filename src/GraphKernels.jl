module GraphKernels

using LightGraphs
using SimpleValueGraphs
using SimpleValueGraphs: AbstractValGraph
using LinearAlgebra: dot, diag
using Statistics: mean, std
using LIBSVM
using Random: MersenneTwister, randperm
using ThreadsX

import LIBSVM: svmtrain, svmpredict

export
    AbstractGraphKernel,

    BaselineGraphKernel,
    ShortestPathGraphKernel,
    PyramidMatchGraphKernel,
    WeisfeilerLehmanGraphKernel,

    NormalizeGraphKernel,

    ConstVertexKernel,
    DiracVertexKernel,
    DotVertexKernel,

    kernelmatrix,
    kernelmatrix_diag,

    k_fold_cross_validation,

    # overridden methods from LIBSVM
    svmtrain,
    svmpredict

include("replacedvertexvals.jl")
include("vertex_kernels.jl")
include("graph-kernels/abstract-graph-kernel.jl")
include("graph-kernels/baseline-graph-kernel.jl")
include("graph-kernels/normalize-graph-kernel.jl")
include("graph-kernels/pyramid-match-graph-kernel.jl")
include("graph-kernels/shortest-path-graph-kernel.jl")
include("graph-kernels/weisfeiler-lehman-graph-kernel.jl")
include("integrations/LIBSVM.jl")

# ================================================================
#     utilities
# ================================================================

"""
    k_fold_cross_validation

Simple k-fold cross validation implementation for quick testing during development.
"""
function k_fold_cross_validation(kernel::AbstractGraphKernel, graphs; k_folds=5, class_key=1, kwargs...)

    n = length(graphs)
    indices = randperm(MersenneTwister(123), n)

    acc_train = Float64[]
    acc_valid = Float64[]
    # TODO there should be better partition function that
    # ensures that not only the last partition is smaller
    for valid_indices in Iterators.partition(1:n, ceil(Int, n // k_folds))
        train_x = [graphs[indices[i]] for i in 1:n if i ∉ valid_indices]
        valid_x = [graphs[indices[i]] for i in valid_indices]

        train_y = [get_graphval(g, class_key) for g in train_x]
        valid_y = [get_graphval(g, class_key) for g in valid_x]

        model = svmtrain(train_x, train_y, kernel; kwargs...)

        train_y_pred = svmpredict(model, train_x)
        valid_y_pred = svmpredict(model, valid_x)

        push!(acc_train, mean(train_y_pred .== train_y))
        push!(acc_valid, mean(valid_y_pred .== valid_y))
    end

    return (mean_train_accuracy=mean(acc_train), std_train_accuracy=std(acc_train),
            mean_valid_accuracy=mean(acc_valid), std_valid_accuracy=std(acc_valid))
end

end
