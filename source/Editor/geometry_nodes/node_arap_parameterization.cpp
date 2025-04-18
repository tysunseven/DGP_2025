#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <memory>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

void arap(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::shared_ptr<MyMesh> iter_mesh)
{
    // TODO: Implement ARAP Parameterization Algorithm.

    /* ------------- ARAP Parameterization Implementation -----------
     ** Implement ARAP mesh parameterization to minimize local distortion.
     **
     ** Steps:
     ** 1. Initial Setup: Use a HW4 parameterization result as initial setup.
     **
     ** 2. Local Phase: For each triangle, compute local orthogonal
     *approximation
     **    (Lt) by computing SVD of Jacobian(Jt) with fixed u.
     **
     ** 3. Global Phase: With Lt fixed, update parameter coordinates(u) by
     *solving
     **    a pre-factored global sparse linear system.
     **
     ** 4. Iteration: Repeat Steps 2 and 3 to refine parameterization.
     **
     */
}

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(arap_parameterization)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Input-2: An embedding result of the mesh. Use the XY coordinates of the
    // embedding as the initialization of the ARAP algorithm
    //
    // Here we use **the result of Assignment 4** as the initialization
    b.add_input<Geometry>("Initialization");

    // Output-1: Like the result of Assignment 4, output the 2D embedding of the
    // mesh
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(arap_parameterization)
{
    // Get the input from params
    auto input = params.get_input<Geometry>("Input");
    auto iters = params.get_input<Geometry>("Initialization");

    // Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>() ||
        !iters.get_component<MeshComponent>()) {
        std::cerr << "ARAP Parameterization: Need Geometry Input." << std::endl;
    }

    /* ----------------------------- Preprocess -------------------------------
    ** Create a halfedge structure (using OpenMesh) for the input mesh. The
    ** half-edge data structure is a widely used data structure in geometric
    ** processing, offering convenient operations for traversing and modifying
    ** mesh elements.
    */

    // Initialization
    auto halfedge_mesh = operand_to_openmesh(&input);
    auto iter_mesh = operand_to_openmesh(&iters);

    // ARAP parameterization
    arap(halfedge_mesh, iter_mesh);

    auto geometry = openmesh_to_operand(iter_mesh.get());

    // Set the output of the nodes
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(arap_parameterization);
NODE_DEF_CLOSE_SCOPE
