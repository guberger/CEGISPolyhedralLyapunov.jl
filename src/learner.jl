@enum StatusCode begin
    LYAPUNOV_FOUND = 0
    LYAPUNOV_INFEASIBLE = 1
    MAX_ITER_REACHED = 2
end

const VT_ = Vector{Float64}
const WT_ = Witness{VT_,Float64,VT_}

function learn_lyapunov(
        pieces::Vector{<:Piece},
        lfs_init::Vector{<:AbstractVector},
        τ, γ, N, xmax, iter_max, solver;
        tol_r=1e-6, tol_η=-1e-6,
        do_print=true, callback_fcn=(args...) -> nothing
    )
    lfs_init_f = map(lf -> Float64.(lf), lfs_init)
    wit_cls = Vector{WT_}[]
    pieces_f = map(
        piece -> Piece(
            Float64.(piece.A),
            map(lf -> Float64.(lf), piece.lfs_dom)
        ), pieces
    )
    Ms = map(piece -> Float64.(I + τ*piece.A), pieces_f)
    nMs = map(M -> opnorm(M, 1), Ms)
    rmax = 2
    iter = 0
    
    while true
        iter += 1
        do_print && println("Iter: ", iter, " - ncl: ", length(wit_cls))
        if iter > iter_max
            println("Max iter exceeded: ", iter)
            break
        end

        lfs::Vector{VT_}, r::Float64 = compute_lfs(
            wit_cls, lfs_init_f, γ, N, rmax, solver
        )

        do_print && println("|-- r generator: ", r)

        if r < tol_r
            println("Lyapunov infeasible")
            return LYAPUNOV_INFEASIBLE, lfs_init_f
        end

        append!(lfs, lfs_init_f)

        x::VT_, η::Float64, qopt::Int, flag::Bool = verify(
            pieces_f, lfs, N, xmax, solver
        )

        @assert flag
        @assert norm(x, Inf) < xmax

        do_print && println("|-- CE: ", x, ", ", γ)

        callback_fcn(iter, wit_cls, lfs, x, qopt)

        if η ≤ tol_η
            println("Valid lyapunov: terminated")
            return LYAPUNOV_FOUND, lfs
        end

        normalize!(x, 2)
        nx = norm(x, 1)
        wit_cl = WT_[]
        for (q, piece) in enumerate(pieces_f)
            in_cl = (q == qopt) || all(lf -> dot(lf, x) ≤ 0, piece.lfs_dom)
            !in_cl && continue
            α = nx*(1 + nMs[q])
            push!(wit_cl, Witness(x, α, Ms[q]*x))
        end
        push!(wit_cls, wit_cl)
    end
    return MAX_ITER_REACHED, lfs_init_f
end