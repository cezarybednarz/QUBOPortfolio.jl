
struct TargetFunction
    name::Symbol
    func::Function
end

const HIGHEST_MEAN = TargetFunction(:highest_mean, results -> mean(r.energy for r in results))
