#= 
non MPI version of selection code for the simple example in
Creel (2015) "On Selection of Statistics for Approximate
Bayesian Computing or the Method of Simulated Moments"

to run this, start Julia, and then enter
include("Select.jl")
=#

using Distances
using Distributions
include("SelectionAlgorithm.jl")

function main()

    # run with -np X+1, X should be an even divisor of reps
    reps = 1 # how many repetitions of algorithm

    nparams = 6 # number of parameters in data set
    S = 10000  # size of paramspace sample
    S2 = 1000  # size of test sample
    whichdep = 0; # set to 0 for all parameters, or target to a specific parameter
    
    # data file to read, initial nparams columns are parameters,
    # rest are candidate statistics
    infile = "simdata.30"
    # output file
    outfile = "battery.30"
    # read the data, and split into params and stats
    simdata = readdlm(infile) 
    theta = simdata[1:S+S2,1:nparams]
    Z = simdata[:,nparams+1:end]
    simdata = 0. # save some memory

    # standardize and normalize stats and parameters
    # Z is divided by trimmed std. dev, so that outliers
    # don't compress normal draws
    Z2 = copy(Z)
    dimZ = size(Z,2)
    @inbounds for i = 1:dimZ
        q = quantile(Z2[:,i],0.99)
        # top bound
        test =  Z2[:,i] .< q
        Z2[:,i] = Z2[:,i].*test  + q.*(1. - test)
        q = -quantile(-Z2[:,i],0.99)
        # bottom bound
        test =  Z2[:,i] .> q
        Z2[:,i] = Z2[:,i] .* test + q.*(1. - test)
    end
    stdZ = std(Z2,1)
    Z = (Z .-mean(Z))./stdZ 
    Z = Z[1:S+S2,:]
    theta = theta .- mean(theta)
    s = std(theta,2)
    s = s.*(s.>0.) .+ (s.<=0) # allows for fixed parameters
    theta = theta ./s
    # split into in and out of sample
    Z_in = Z[1:S,:]
    Z_out = Z[S+1:S+S2,:]
    theta_in = theta[1:S,:]
    theta_out = theta[S+1:S+S2,:]
 
    # number of neighbors for SBIL fits
    n, dimZ = size(Z_in)
    #neighbors = round(Int,floor(1.*n^0.25))
    neighbors = 100  # local linear (now the default) needs more neighbors
    # SA controls
    temperature = 0.02 # 0.02 works well, given that parameters are standardized and normalized
    r = 1. # factor to set nt
    nt = r*dimZ # number of evaluations between temp reductions
    rt = 0.75 # rate of temp reduction
    r = 10; # total evals is this number times dimZ
    maxevals = r*dimZ
    
    # do sa
    results = zeros(reps, dimZ+1)
    for i = 1:reps
        selected, obj_value = sa_for_selection(Z_in, Z_out, theta_in, theta_out, temperature, nt, rt, maxevals, neighbors, whichdep)
        results[i,:] = [obj_value vec(selected)']
    end
    writedlm(outfile, results) 
end
# call the main function
main()
