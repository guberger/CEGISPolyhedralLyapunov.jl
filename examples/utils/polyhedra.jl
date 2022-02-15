using LazySets
using CDDLib
using PyCall
const spatial = pyimport_conda("scipy.spatial", "scipy")

function retrieve_vertices_2d(c_list)
    M = length(c_list)
    Ahrep = zeros(M, 2)
    bhrep = ones(M)
    for j = 1:M
        Ahrep[j, :] = c_list[j]
    end
    poly = HPolytope(Ahrep, bhrep)
    points = vertices_list(poly, backend=CDDLib.Library())
    return convex_hull(points)
end

function retrieve_simplices_3d(c_list)
    M = length(c_list)
    Ahrep = zeros(M, 3)
    bhrep = ones(M)
    for j = 1:M
        Ahrep[j, :] = c_list[j]
    end
    poly = HPolytope(Ahrep, bhrep)
    points = vertices_list(poly, backend=CDDLib.Library())
    M = length(points)
    Points = zeros(M, 3)
    for j = 1:M
        Points[j, :] = points[j]
    end
    hull = spatial.ConvexHull(Points)
    simplices = map(j ->
        map(k -> hull.points[hull.simplices[j, k] + 1, :], 1:3),
        1:size(hull.simplices, 1))
    return simplices
end

norm_poly(x, c_list) = maximum(c -> c'*x, c_list)