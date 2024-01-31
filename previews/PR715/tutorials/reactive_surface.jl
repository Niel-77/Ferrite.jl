using Ferrite, FerriteGmsh
using BlockArrays, SparseArrays, LinearAlgebra

function assemble_element_mass!(Me::Matrix, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(Me, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δuᵢ = shape_value(cellvalues, q_point, i)
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                δuⱼ = shape_value(cellvalues, q_point, j)
                # Add contribution to Ke
                Me[2*i-1, 2*j-1] += (δuᵢ * δuⱼ) * dΩ
                Me[2*i  , 2*j  ] += (δuᵢ * δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_element_diffusion!(De::Matrix, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(De, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            ∇δuᵢ = shape_gradient(cellvalues, q_point, i)
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇δuⱼ = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                De[2*i-1, 2*j-1] += 2*0.00008 * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
                De[2*i  , 2*j  ] += 2*0.00004 * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_matrices!(M::SparseMatrixCSC, D::SparseMatrixCSC, cellvalues::CellValues, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cellvalues)

    # Allocate the element stiffness matrix and element force vector
    Me = zeros(2*n_basefuncs, 2*n_basefuncs)
    De = zeros(2*n_basefuncs, 2*n_basefuncs)

    # Create an assembler
    M_assembler = start_assemble(M)
    D_assembler = start_assemble(D)
    # Loop over all cels
    for cell in CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        # Compute element contribution
        assemble_element_mass!(Me, cellvalues)
        assemble!(M_assembler, celldofs(cell), Me)

        assemble_element_diffusion!(De, cellvalues)
        assemble!(D_assembler, celldofs(cell), De)
    end
    return nothing
end

function setup_initial_conditions!(u₀::Vector, cellvalues::CellValues, dh::DofHandler)
    u₀ .= ones(ndofs(dh))
    u₀[2:2:end] .= 0.0

    n_basefuncs = getnbasefunctions(cellvalues)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)

        coords = getcoordinates(cell)
        dofs = celldofs(cell)
        uₑ = @view u₀[dofs]
        rv₀ₑ = reshape(uₑ, (2, n_basefuncs))

        for i in 1:n_basefuncs
            if coords[i][3] > 0.9
                rv₀ₑ[1, i] = 0.5
                rv₀ₑ[2, i] = 0.25
            end
        end
    end

    u₀ .+= 0.01*rand(ndofs(dh))
end

function gray_scott_sphere(F, k, Δt, T)
    # We start by setting up grid, dof handler and the matrices for the heat problem.
    gmsh.initialize()

    # Add a unit sphere
    gmsh.model.occ.addSphere(0.0,0.0,0.0,1.0)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    # Generate surface elements
    gmsh.model.mesh.generate(2)

    # Create a grid out of it
    nodes = tonodes()
    elements, _ = toelements(2)
    gmsh.finalize()
    grid = Grid(elements, nodes);

    ip = Lagrange{RefTriangle, 1}()
    ip_geo = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    cellvalues = CellValues(qr, ip, ip_geo^3);

    dh = DofHandler(grid);
    add!(dh, :reactants, ip^2);
    close!(dh);

    M = create_sparsity_pattern(dh; coupling=[true false;false true])
    D = create_sparsity_pattern(dh; coupling=[true false;false true])
    assemble_matrices!(M, D, cellvalues, dh);

    # Since the heat problem is linear and has no time dependent parameters, we precompute the
    # decomposition of the system matrix to speed up the linear system solver.
    A = M + Δt .* D
    Alu = cholesky(A)

    # Now we setup buffers for the time dependent solution and fill the initial condition.
    uₜ   = zeros(ndofs(dh))
    uₜ₋₁ = ones(ndofs(dh))
    setup_initial_conditions!(uₜ₋₁, cellvalues, dh)

    # And prepare output for visualization.
    pvd = paraview_collection("reactive-surface.pvd");
    vtk_grid("reactive-surface-0.0.vtu", dh) do vtk
        vtk_point_data(vtk, dh, uₜ₋₁)
        vtk_save(vtk)
        pvd[0.0] = vtk
    end

    # This is now the main solve loop.
    for (iₜ, t) ∈ enumerate(Δt:Δt:T)
        # First we solve the heat problem
        uₜ .= Alu \ (M * uₜ₋₁)

        # Then we solve the point-wise reaction problem with the solution of
        # the heat problem as initial guess.
        rvₜ = reshape(uₜ, (2, length(grid.nodes)))
        for i ∈ 1:length(grid.nodes)
            r₁ = rvₜ[1, i]
            r₂ = rvₜ[2, i]
            rvₜ[1, i] += Δt*( -r₁*r₂^2 + F *(1 - r₁) )
            rvₜ[2, i] += Δt*(  r₁*r₂^2 - r₂*(F + k ) )
        end

        # The solution is then stored every 10th step to vtk files for
        # later visualization purposes.
        if (iₜ % 10) == 0
            vtk_grid("reactive-surface-$t.vtu", dh) do vtk
                vtk_point_data(vtk, dh, uₜ)
                vtk_save(vtk)
                pvd[t] = vtk
            end
        end

        # Finally we totate the solution to initialize the next timestep.
        uₜ₋₁ .= uₜ
    end

    vtk_save(pvd);
end

# This parametrization gives the spot pattern shown in the gif above.
gray_scott_sphere(0.06, 0.062, 10.0, 32000.0)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
