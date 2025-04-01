#include <Eigen/Sparse>
#include <cmath>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void tutte_embedding(MyMesh& omesh)
{
    // TODO: Implement Tutte Embedding Algorithm.
    //
    // In this task, you are required to **modify** the original mesh to a
    // 'minimal surface' mesh with the boundary of the input mesh as its
    // boundary.
    //
    // Specifically, the positions of the boundary vertices of the input mesh
    // should be fixed. By solving a global Laplace equation on the mesh,
    // recalculate the coordinates of the vertices inside the mesh to achieve
    // the minimal surface configuration

    /*
     ** Algorithm Pseudocode for Minimal Surface Calculation
     ** ------------------------------------------------------------------------
     ** 1. Initialize mesh with input boundary conditions.
     **    - For each boundary vertex, fix its position.
     **    - For internal vertices, initialize with initial guess if necessary.
     **
     ** 2. Construct Laplacian matrix for the mesh.
     **    - Compute weights for each edge based on the chosen weighting scheme
     **      (e.g., uniform weights for simplicity).
     **    - Assemble the global Laplacian matrix.
     **
     ** 3. Solve the Laplace equation for interior vertices.
     **    - Apply Dirichlet boundary conditions for boundary vertices.
     **    - Solve the linear system (Laplacian * X = 0) to find new positions
     **      for internal vertices.
     **
     ** 4. Update mesh geometry with new vertex positions.
     **    - Ensure the mesh respects the minimal surface configuration.
     **
     ** Note: This pseudocode outlines the general steps for calculating a
     ** minimal surface mesh given fixed boundary conditions using the Laplace
     ** equation. The specific implementation details may vary based on the mesh
     ** representation and numerical methods used.
     **
     */

    return;
}

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(tutte)
{
    // Function content omitted
    b.add_input<Geometry>("Input");

    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(tutte)
{
    // Function content omitted

    // Get the input from params
    auto input = params.get_input<Geometry>("Input");

    // Avoid processing the node when there is no input
    if (!input.get_component<MeshComponent>()) {
        std::cerr << "Tutte Parameterization: Need Geometry Input."
                  << std::endl;
        return false;
    }

    auto mesh = input.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    for (int i = 0; i < vertices.size(); i++) {
        omesh.add_vertex(
            OpenMesh::Vec3f(vertices[i][0], vertices[i][1], vertices[i][2]));
    }

    // Add faces
    size_t start = 0;
    for (int face_vertex_count : face_vertex_counts) {
        std::vector<OpenMesh::VertexHandle> face;
        face.reserve(face_vertex_count);
        for (int j = 0; j < face_vertex_count; j++) {
            face.push_back(
                OpenMesh::VertexHandle(face_vertex_indices[start + j]));
        }
        omesh.add_face(face);
        start += face_vertex_count;
    }

    omesh.request_vertex_normals();
    omesh.request_face_normals();
    omesh.update_normals();

    // Perform Tutte Embedding
    tutte_embedding(omesh);

    // Convert back to Geometry
    pxr::VtArray<pxr::GfVec3f> tutte_vertices;
    for (const auto& v : omesh.vertices()) {
        const auto& p = omesh.point(v);
        tutte_vertices.push_back(pxr::GfVec3f(p[0], p[1], p[2]));
    }
    pxr::VtArray<int> tutte_faceVertexIndices;
    pxr::VtArray<int> tutte_faceVertexCounts;
    for (const auto& f : omesh.faces()) {
        size_t count = 0;
        for (const auto& vf : f.vertices()) {
            tutte_faceVertexIndices.push_back(vf.idx());
            count += 1;
        }
        tutte_faceVertexCounts.push_back(count);
    }

    Geometry tutte_geometry;
    auto tutte_mesh = std::make_shared<MeshComponent>(&tutte_geometry);

    tutte_mesh->set_vertices(tutte_vertices);
    tutte_mesh->set_face_vertex_indices(tutte_faceVertexIndices);
    tutte_mesh->set_face_vertex_counts(tutte_faceVertexCounts);
    tutte_geometry.attach_component(tutte_mesh);
    // Set the output of the nodes
    params.set_output("Output", tutte_geometry);
    return true;
}

NODE_DECLARATION_UI(tutte);
NODE_DEF_CLOSE_SCOPE
