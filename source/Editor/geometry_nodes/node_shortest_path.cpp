#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <cstddef>
#include <string>
#include <queue>
#include <vector>
#include <limits>

#include "GCore/Components/MeshOperand.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

// Return true if the shortest path exists, and fill in the shortest path
// vertices and the distance. Otherwise, return false.
bool find_shortest_path(
    const MyMesh::VertexHandle& start_vertex_handle,
    const MyMesh::VertexHandle& end_vertex_handle,
    const MyMesh& omesh,
    std::list<size_t>& shortest_path_vertex_indices,
    float& distance)
{
    if (!start_vertex_handle.is_valid() || !end_vertex_handle.is_valid() ||
        start_vertex_handle.idx() >= omesh.n_vertices() ||
        end_vertex_handle.idx() >= omesh.n_vertices()) {
        return false;
    }

    const size_t num_vertices = omesh.n_vertices();
    std::vector<float> dist(
        num_vertices, std::numeric_limits<float>::infinity());
    std::vector<MyMesh::VertexHandle> prev(num_vertices);

    auto cmp = [](const std::pair<float, MyMesh::VertexHandle>& a,
                  const std::pair<float, MyMesh::VertexHandle>& b) {
        return a.first > b.first;
    };
    std::priority_queue<
        std::pair<float, MyMesh::VertexHandle>,
        std::vector<std::pair<float, MyMesh::VertexHandle>>,
        decltype(cmp)>
        pq(cmp);

    dist[start_vertex_handle.idx()] = 0.0f;
    pq.emplace(0.0f, start_vertex_handle);

    while (!pq.empty()) {
        float current_dist = pq.top().first;
        MyMesh::VertexHandle u = pq.top().second;
        pq.pop();

        if (u == end_vertex_handle)
            break;
        if (current_dist > dist[u.idx()])
            continue;

        for (MyMesh::ConstVertexVertexIter vv_it = omesh.cvv_iter(u);
             vv_it.is_valid();
             ++vv_it) {
            MyMesh::VertexHandle v = *vv_it;
            MyMesh::Point u_pos = omesh.point(u);
            MyMesh::Point v_pos = omesh.point(v);
            float edge_length = (u_pos - v_pos).length();

            float new_dist = current_dist + edge_length;

            if (new_dist < dist[v.idx()]) {
                dist[v.idx()] = new_dist;
                prev[v.idx()] = u;
                pq.emplace(new_dist, v);
            }
        }
    }

    if (dist[end_vertex_handle.idx()] ==
        std::numeric_limits<float>::infinity()) {
        return false;
    }

    shortest_path_vertex_indices.clear();
    distance = dist[end_vertex_handle.idx()];
    MyMesh::VertexHandle current = end_vertex_handle;

    while (current != start_vertex_handle) {
        shortest_path_vertex_indices.push_front(current.idx());
        current = prev[current.idx()];
        if (!current.is_valid())
            return false;
    }
    shortest_path_vertex_indices.push_front(start_vertex_handle.idx());

    return true;
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(shortest_path)
{
    b.add_input<std::string>("Picked Mesh [0] Name");
    b.add_input<std::string>("Picked Mesh [1] Name");
    b.add_input<Geometry>("Picked Mesh");
    b.add_input<size_t>("Picked Vertex [0] Index");
    b.add_input<size_t>("Picked Vertex [1] Index");

    b.add_output<std::list<size_t>>("Shortest Path Vertex Indices");
    b.add_output<float>("Shortest Path Distance");
}

NODE_EXECUTION_FUNCTION(shortest_path)
{
    auto picked_mesh_0_name =
        params.get_input<std::string>("Picked Mesh [0] Name");
    auto picked_mesh_1_name =
        params.get_input<std::string>("Picked Mesh [1] Name");
    // Ensure that the two picked meshes are the same
    if (picked_mesh_0_name != picked_mesh_1_name) {
        std::cerr << "Ensure that the two picked meshes are the same"
                  << std::endl;
        return false;
    }

    auto mesh = params.get_input<Geometry>("Picked Mesh")
                    .get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();
    auto face_vertex_indices = mesh->get_face_vertex_indices();

    // Convert the mesh to OpenMesh
    MyMesh omesh;

    // Add vertices
    std::vector<OpenMesh::VertexHandle> vhandles;
    vhandles.reserve(vertices.size());

    for (auto vertex : vertices) {
        omesh.add_vertex(OpenMesh::Vec3f(vertex[0], vertex[1], vertex[2]));
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

    auto start_vertex_index =
        params.get_input<size_t>("Picked Vertex [0] Index");
    auto end_vertex_index = params.get_input<size_t>("Picked Vertex [1] Index");

    // Turn the vertex indices into OpenMesh vertex handles
    OpenMesh::VertexHandle start_vertex_handle(start_vertex_index);
    OpenMesh::VertexHandle end_vertex_handle(end_vertex_index);

    // The indices of the vertices on the shortest path, including the start and
    // end vertices
    std::list<size_t> shortest_path_vertex_indices;

    // The distance of the shortest path
    float distance = 0.0f;

    if (find_shortest_path(
            start_vertex_handle,
            end_vertex_handle,
            omesh,
            shortest_path_vertex_indices,
            distance)) {
        params.set_output(
            "Shortest Path Vertex Indices", shortest_path_vertex_indices);
        params.set_output("Shortest Path Distance", distance);
        return true;
    }
    else {
        params.set_output("Shortest Path Vertex Indices", std::list<size_t>());
        params.set_output("Shortest Path Distance", 0.0f);
        return false;
    }

    return true;
}

NODE_DECLARATION_UI(shortest_path);
NODE_DECLARATION_REQUIRED(shortest_path);

NODE_DEF_CLOSE_SCOPE
