#= 
non MPI version of selection code for the simple example in
Creel (2015) "On Selection of Statistics for Approximate
Bayesian Computing or the Method of Simulated Moments"

to run this, start Julia, and then enter
include("Select.jl")
=#

using Distances
include("SelectionAlgorithm.jl")

function main()

    # run with -np X+1, X should be an even divisor of reps
    reps = 1 # how many repetitions of algorithm

    nparams = 6 # number of parameters in data set
    S = 10000  # size of paramspace sample
    S2 = 1000  # size of test sample
 
    # data file to read, initial nparams columns are parameters,
    # rest are candidate statistics
    infile = "simdata30"
    # output file
    outfile = "selected30.repeat"
    # read the data, and split into params and stats
    simdata = readdlm(infile) 
    theta = simdata[1:S+S2,1:nparams]
    Z = simdata[:,nparams+1:end]
    simdata = 0. # save some memory

    # standardize and normalize stats and parameters
    Z = (Z .-mean(Z))./std(Z,2) 
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
    neighbors = round(Int,floor(1.*n^0.25))
    
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
        selected, obj_value = sa_for_selection(Z_in, Z_out, theta_in, theta_out, temperature, nt, rt, maxevals, neighbors)
        results[i,:] = [obj_value vec(selected)']
    end
    writedlm(outfile, results) 
end
# call the main function
main()
