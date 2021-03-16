
struct GraphSVMModel
    svm::LIBSVM.SVM
    kernel::AbstractGraphKernel
    graphs::AbstractVector
end


function svmtrain(graphs::AbstractVector{<:AbstractGraph}, labels, kernel::AbstractGraphKernel)

    n = length(graphs)

    X = vcat(transpose(1:n), gramm_matrix(kernel, graphs))
    println("Finished calculating gramm_matrix")

    svm = svmtrain(X, labels, kernel=Kernel.Precomputed)

    return GraphSVMModel(svm, kernel, graphs)
end

function svmpredict(model::GraphSVMModel, unpredicted_graphs::AbstractVector{<:AbstractGraph})

    graphs = model.graphs
    kernel = model.kernel

    m = length(graphs)
    n = length(unpredicted_graphs)

    X = Matrix{Float64}(undef, m + 1, n)
    X[1, :] = 1:n

    for i in 1:m, j in 1:n
        X[i+1, j] = kernel(graphs[i], unpredicted_graphs[j])
    end

    return svmpredict(model.svm, X)[1] # for simplicity return only the labels for now
end
