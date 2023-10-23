using PythonCall

function myfunc(x)
    x = pyconvert(Float64, x)
    println(typeof(x))
    return 2*x
end

pf = pyfunc(myfunc)
pf(2)
