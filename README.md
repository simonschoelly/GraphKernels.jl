# GraphKernels.jl

![](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

A Julia package for kernel functions on graphs.

### Example

```julia
julia> using GraphKernels: ShortestPathGraphKernel, svmtrain, svmpredict

julia> using GraphDatasets: loadgraphs, TUDatasets
julia> using SimpleValueGraphs: get_graphval
julia> using Random: shuffle
julia> using Statistics: mean

# load the MUTAG dataset - it contains 188 graphs of two different classes
julia> graphs = loadgraphs(TUDatasets.MUTAGDataset(); resolve_categories=true)
188-element ValGraphCollection of graphs with
              eltype: Int8
  vertex value types: (chem = String,)
    edge value types: (bond_type = String,)
   graph value types: (class = Int8,)

# shuffle the graphs and split into train and test data
julia> graphs = shuffle(graphs);
julia> X_train, X_test = graphs[begin:120], graphs[121:end];
julia> y_train, y_test = get_graphval.(X_train, :class), get_graphval.(X_test, :class);

# instantiate a ShortestPathGraphKernel
# dist_key is set to nothing so that we use unit distances for all edges
julia> kernel = ShortestPathGraphKernel(;dist_key=nothing)
ShortestPathGraphKernel{ConstVertexKernel}(0.0, ConstVertexKernel(1.0), nothing)

# train a support vector machine with that kernel
julia> model = svmtrain(X_train, y_train, kernel);

# predict classed on the test data
julia> y_test_pred = svmpredict(model, X_test);

# compare with the actual classes and calculate the accuracy
julia> accuracy = mean(y_test .== y_test_pred)
0.8529411764705882
```
