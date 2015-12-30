# nonparametric regression, using pre-generated weights
# can be local constant, local linear, or local quadratic
#
# y is n X p
# x is n X dimx
# xeval is neval X dimx
# weights is n X neval (different weights for each eval. point)
#
# weights should sum to one by columns, they are typical 
# nonparametric weights from a kernel.
function npreg(y, x, xeval, weights, order)
    weights = sqrt(weights)
    neval, dimx = size(xeval)
    n, dimy = size(y)
    # local constant
    if order==0            
        X = ones(n,1)
        Xeval = ones(neval,1)
    elseif order==1    
    # local linear    
        X = [ones(n,1) x]
        Xeval = [ones(neval,1) xeval]
    else
    # local quadratic
        stacked = [x; xeval]
        # cross products
        CP = Array(Float64, n+neval, Int((dimx-1)*dimx/2))
        cpind = 0
        @inbounds for i = 1:dimx-1
            @inbounds for j = (i+1):dimx
                cpind += 1
                CP[:,cpind] = stacked[:,i].*stacked[:,j]
            end
        end
        ZZ = [ones(n+neval,1) stacked CP]
        X = sub(ZZ,1:n,:)
        Xeval = sub(ZZ,(n+1):n+neval,:)
    end
    # do the fit
    yhat = Array(Float64, neval, dimy)
    @simd for i = 1:neval
        WX = weights[:,i] .* X
        Wy = weights[:,i] .* y
        yhat[i,:] = Xeval[i,:]*(WX\Wy)
    end    
    return yhat
end
