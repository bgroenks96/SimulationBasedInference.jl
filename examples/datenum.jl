using Dates

"""
	DateNumber <: Real

Represents a MATLAB DateNumber type.
"""
struct DateNumber <: Real
	val::Float64
end
Base.convert(::Type{Float64},d::DateNumber) = d.val
Base.convert(::Type{Int64},d::DateNumber) = convert(Int64,d.val)
Base.convert(::DateNumber,val::Float64) = DateNumber(val)
Base.convert(::DateNumber,val::Int64) = DateNumber(float(val))
Base.promote_rule(::Type{DateNumber},::Type{T}) where {T<:Real} = promote_type(Float64,T)

"""
	datenum(d::Dates.DateTime)

Converts a Julia DateTime to a MATLAB style DateNumber.
MATLAB represents time as DateNumber, a double precision floating
point number being the the number of days since January 0, 0000
"""
function todatenum(d::DateTime)
	MATLAB_EPOCH = DateTime(-0001,12,31)
	fac = 1000. * 60. * 60. * 24.
	return DateNumber(value(d - MATLAB_EPOCH) / fac)
end

"""
	todatetime(d::DateNumber)

Converts a MATLAB DateNumber to a Julia DateTime.
"""
function todatetime(d::DateNumber)
	MATLAB_EPOCH = DateTime(-0001,12,31)
	fac = 1000. * 60. * 60. * 24.
	return Millisecond(d*fac) + MATLAB_EPOCH
end