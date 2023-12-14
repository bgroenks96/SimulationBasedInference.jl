"""
    DDM(t,P,T,a,b)

this is my function....
"""
DDM(t, P, T, a::Number, b::Number) = DDM(t, P, T, [a], [b])
function DDM(t, P, T, a::AbstractVector, b::AbstractVector)
    # DDM: A simple degree day snowmelt model
    # Coded for a single cell ensemble run, but could
    # also easily be extended to multiple cells stored as ensemble chunks.
    
    # Hyperparameters:
    Tsnow=1; # Temperature threshold for snowfall (degrees C)
    Tmelt=0; # Temperature threshold for snowmelt (degrees C)
    
    Nt=length(t); # Number of time steps.
    Ne=length(a); # Number of ensemble members (can be just 1).
    D_old=0; # Initial condition (no snow).
    D=zeros(eltype(a), Nt,Ne);
    
    b0=zeros(size(b));
    a0=zeros(size(a));
    
    for j=1:Nt
        Pj=P[j];
        Tj=T[j];
        cansnow=Tj<=Tsnow; # Snow is possible.
        # For simplicity rain does not contribute to SWE
        if cansnow
            bj=b;
        else
            bj=b0;
        end
        Aj=Pj.*bj; # Accumulation for this day
        ddj=Tj-Tmelt; # Degree day for this day
        melting=ddj>0;
        if melting
            aj=a;
        else
            aj=a0;
        end
        Mj=ddj.*aj;
        D_new=D_old.+Aj.-Mj;
        D_new=max.(D_new,0);
        D[j,:].=D_new;
        D_old=D_new;
    end

    return D
end