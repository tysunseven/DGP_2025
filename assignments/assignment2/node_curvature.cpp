#include <corecrt_math_defines.h>
#include <pxr/base/vt/array.h>

#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "OpenMesh/Core/Geometry/Vector11T.hh"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

void compute_mean_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& mean_curvature)
{
    // TODO: Implement the mean curvature computation
    //  You need to fill in `mean_curvature`

    mean_curvature.resize(omesh.n_vertices());
    for (auto v_it = omesh.vertices_begin(); v_it != omesh.vertices_end();
         ++v_it) {
        auto vertex_handle = *v_it;
        // 计算顶点周围三角形的Laplace-Beltrami Operator
        auto laplace = MyMesh::Point(0.0f, 0.0f, 0.0f);

        // 遍历由顶点出发的半边
        for (auto voh_it = omesh.cvoh_iter(vertex_handle); voh_it.is_valid();
             ++voh_it) {
            auto he_handle = *voh_it;

            std::vector<MyMesh::Point> face_vertices;
            face_vertices.resize(3);
            float cot_sum = 0.0f;

            // 先计算一侧的面
            // 求出该边对应的面的三个顶点坐标，我们需要求的是face_vertices[2]处的cotangent
            if (!omesh.is_boundary(he_handle)) {
                face_vertices[0] =
                    omesh.point(omesh.from_vertex_handle(he_handle));
                face_vertices[1] =
                    omesh.point(omesh.to_vertex_handle(he_handle));
                face_vertices[2] =
                    omesh.point(omesh.to_vertex_handle(he_handle.next()));
                float cos_1 = OpenMesh::dot(
                    (face_vertices[0] - face_vertices[2]).normalize(),
                    (face_vertices[1] - face_vertices[2]).normalize());
                float cot_1 = cos_1 / sqrt(1 - cos_1 * cos_1);
                cot_sum += cot_1;
            }

            // 计算另一侧的面
            // 求出该边对应的面的三个顶点坐标，我们需要求的是face_vertices[2]处的cotangent
            auto he_handle_oppo = omesh.opposite_halfedge_handle(he_handle);
            if (!omesh.is_boundary(he_handle_oppo)) {
                face_vertices[0] =
                    omesh.point(omesh.from_vertex_handle(he_handle_oppo));
                face_vertices[1] =
                    omesh.point(omesh.to_vertex_handle(he_handle_oppo));
                face_vertices[2] =
                    omesh.point(omesh.to_vertex_handle(he_handle_oppo.next()));
                float cos_2 = OpenMesh::dot(
                    (face_vertices[0] - face_vertices[2]).normalize(),
                    (face_vertices[1] - face_vertices[2]).normalize());
                float cot_2 = cos_2 / sqrt(1 - cos_2 * cos_2);
                cot_sum += cot_2;
            }

            // 求和
            MyMesh::Point edge_vector =
                omesh.point(vertex_handle) -
                omesh.point(omesh.to_vertex_handle(he_handle));
            laplace += cot_sum * edge_vector;
        }
        // 计算面积
        float area = 0.0f;
        for (auto vf_it = omesh.cvf_iter(vertex_handle); vf_it.is_valid();
             ++vf_it) {
            auto face_handle = *vf_it;
            area += omesh.calc_face_area(face_handle);
        }
        area /= 3.0f;  // 平均面积

        laplace /= (2.0f * area);  // Laplace-Beltrami Operator

        // 计算平均曲率
        auto H = laplace.length() / 2.0f;

        // 确定符号
        auto normal = omesh.normal(vertex_handle);
        if (OpenMesh::dot(normal, laplace) < 0.0f) {
            H = -H;
        }
        mean_curvature[vertex_handle.idx()] = H;
    }
}

void compute_gaussian_curvature(
    const MyMesh& omesh,
    pxr::VtArray<float>& gaussian_curvature)
{
    // TODO: Implement the Gaussian curvature computation
    //  You need to fill in `gaussian_curvature`

    gaussian_curvature.resize(omesh.n_vertices());
    for (auto v_it = omesh.vertices_begin(); v_it != omesh.vertices_end();
         ++v_it) {
        auto vertex_handle = *v_it;

        // 计算顶点周围三角形的角度和
        float angle_sum = 0.0f;
        // 遍历顶点的相邻面
        for (auto vf_it = omesh.cvf_iter(vertex_handle); vf_it.is_valid();
             ++vf_it) {
            auto face_handle = *vf_it;
            // 获取面上的顶点
            auto fv_it = omesh.cfv_iter(face_handle);
            auto v_0 = *fv_it++;
            auto v_1 = *fv_it++;
            auto v_2 = *fv_it++;

            // 计算角度
            if (vertex_handle == v_0) {
                auto edge1 = omesh.point(v_1) - omesh.point(v_0);
                auto edge2 = omesh.point(v_2) - omesh.point(v_0);
                angle_sum +=
                    acos(OpenMesh::dot(edge1.normalize(), edge2.normalize()));
            }
            else if (vertex_handle == v_1) {
                auto edge1 = omesh.point(v_0) - omesh.point(v_1);
                auto edge2 = omesh.point(v_2) - omesh.point(v_1);
                angle_sum +=
                    acos(OpenMesh::dot(edge1.normalize(), edge2.normalize()));
            }
            else if (vertex_handle == v_2) {
                auto edge1 = omesh.point(v_0) - omesh.point(v_2);
                auto edge2 = omesh.point(v_1) - omesh.point(v_2);
                angle_sum +=
                    acos(OpenMesh::dot(edge1.normalize(), edge2.normalize()));
            }
        }
        // 计算角度差
        float angle_defect = 2.0f * M_PI;
        if (omesh.is_boundary(vertex_handle)) {
            angle_defect = M_PI;
        }
        angle_defect -= angle_sum;

        // 计算面积
        float area = 0.0f;
        for (auto vf_it = omesh.cvf_iter(vertex_handle); vf_it.is_valid();
             ++vf_it) {
            auto face_handle = *vf_it;
            area += omesh.calc_face_area(face_handle);
        }
        area /= 3.0f;  // 平均面积

        gaussian_curvature[vertex_handle.idx()] =
            angle_defect / area;  // 高斯曲率 = 角度差 / 面积
    }
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(mean_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Mean Curvature");
}

NODE_EXECUTION_FUNCTION(mean_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

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

    // Request vertex and face normals
    omesh.request_vertex_normals();
    omesh.request_face_normals();
    omesh.update_normals();

    // Compute mean curvature
    pxr::VtArray<float> mean_curvature;
    mean_curvature.reserve(omesh.n_vertices());

    compute_mean_curvature(omesh, mean_curvature);

    params.set_output("Mean Curvature", mean_curvature);

    return true;
}

NODE_DECLARATION_UI(mean_curvature);

NODE_DECLARATION_FUNCTION(gaussian_curvature)
{
    b.add_input<Geometry>("Mesh");
    b.add_output<pxr::VtArray<float>>("Gaussian Curvature");
}

NODE_EXECUTION_FUNCTION(gaussian_curvature)
{
    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto face_vertex_indices = mesh->get_face_vertex_indices();
    auto face_vertex_counts = mesh->get_face_vertex_counts();

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

    // Compute Gaussian curvature
    pxr::VtArray<float> gaussian_curvature;
    gaussian_curvature.reserve(omesh.n_vertices());

    compute_gaussian_curvature(omesh, gaussian_curvature);

    params.set_output("Gaussian Curvature", gaussian_curvature);

    return true;
}

NODE_DECLARATION_UI(gaussian_curvature);

NODE_DEF_CLOSE_SCOPE
