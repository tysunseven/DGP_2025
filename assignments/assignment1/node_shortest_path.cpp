#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <cstddef>
#include <queue>
#include <string>

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
    // TODO: Implement the shortest path algorithm
    // You need to fill in `shortest_path_vertex_indices` and `distance`

    using VertexHandle = MyMesh::VertexHandle;

    // 检查顶点句柄是否有效
    if (!omesh.is_valid_handle(start_vertex_handle) ||
        !omesh.is_valid_handle(end_vertex_handle)) {
        return false;
    }

    // 存储距离和前驱节点的映射
    std::map<VertexHandle, float> dist;
    std::map<VertexHandle, VertexHandle> prev;

    // 优先队列，用于Dijkstra算法
    using PQElement = std::pair<float, VertexHandle>;
    std::priority_queue<
        PQElement,
        std::vector<PQElement>,
        std::greater<PQElement>>
        pq;

    // 初始化所有顶点的距离为无穷大
    for (auto vh : omesh.vertices()) {
        dist[vh] = std::numeric_limits<float>::infinity();
    }

    // 设置起点距离为0
    dist[start_vertex_handle] = 0.0f;
    pq.push({ 0.0f, start_vertex_handle });

    // Dijkstra算法主循环
    while (!pq.empty()) {
        auto [d, current] = pq.top();
        pq.pop();

        // 如果找到终点，结束搜索
        if (current == end_vertex_handle) {
            break;
        }

        // 如果当前距离大于已知最短距离，跳过
        if (d > dist[current]) {
            continue;
        }

        // 遍历当前顶点的所有邻接顶点
        for (auto voh_it = omesh.cvoh_iter(current); voh_it.is_valid();
             ++voh_it) {
            VertexHandle next = omesh.to_vertex_handle(*voh_it);

            // 计算边的长度（两点之间的欧氏距离）
            OpenMesh::Vec3f p1 = omesh.point(current);
            OpenMesh::Vec3f p2 = omesh.point(next);
            float edge_length = (p2 - p1).length();

            // 计算通过当前顶点到达邻接顶点的总距离
            float alt_dist = dist[current] + edge_length;

            // 如果找到更短的路径，更新距离和前驱
            if (alt_dist < dist[next]) {
                dist[next] = alt_dist;
                prev[next] = current;
                pq.push({ alt_dist, next });
            }
        }
    }

    // 如果没有找到路径到终点
    if (dist[end_vertex_handle] == std::numeric_limits<float>::infinity()) {
        return false;
    }

    // 重建最短路径
    shortest_path_vertex_indices.clear();
    VertexHandle current = end_vertex_handle;

    while (true) {
        shortest_path_vertex_indices.push_front(current.idx());
        if (current == start_vertex_handle) {
            break;
        }
        current = prev[current];
    }

    // 设置总距离
    distance = dist[end_vertex_handle];

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
