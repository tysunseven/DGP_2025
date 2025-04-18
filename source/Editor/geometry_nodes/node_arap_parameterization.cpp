#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <memory>
#include <vector>

#include "GCore/Components/MeshOperand.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

double compute_cotangent(MyMesh::Point v1, MyMesh::Point v2, MyMesh::Point v3)
{
    MyMesh::Point e1 = v1 - v2;
    MyMesh::Point e2 = v3 - v2;
    double dot = e1.dot(e2);
    double cross = e1.cross(e2).norm();
    return dot / (cross + 1e-6);  // 避免除零
}

void arap(
    std::shared_ptr<MyMesh> halfedge_mesh,
    std::shared_ptr<MyMesh> iter_mesh)
{
    // Step 1: 保存初始参数化坐标
    std::vector<Eigen::Vector2d> initial_uv;
    for (auto vh : iter_mesh->vertices()) {
        auto p = iter_mesh->point(vh);
        initial_uv.emplace_back(p[0], p[1]);
    }

    const int n_vertices = iter_mesh->n_vertices();
    const int num_iterations = 50;

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Step 2: 局部阶段 - 计算每个面的Lt矩阵
        std::vector<Eigen::Matrix2d> Lt_list(iter_mesh->n_faces());

        for (auto fh : iter_mesh->faces()) {
            // 获取三个顶点
            auto he = fh.halfedge();
            auto vh0 = he.from();
            auto vh1 = he.to();
            he = he.next();
            auto vh2 = he.to();

            // 当前UV坐标
            Eigen::Vector2d u0(
                iter_mesh->point(vh0)[0], iter_mesh->point(vh0)[1]);
            Eigen::Vector2d u1(
                iter_mesh->point(vh1)[0], iter_mesh->point(vh1)[1]);
            Eigen::Vector2d u2(
                iter_mesh->point(vh2)[0], iter_mesh->point(vh2)[1]);

            // 初始UV坐标
            Eigen::Vector2d u0_init = initial_uv[vh0.idx()];
            Eigen::Vector2d u1_init = initial_uv[vh1.idx()];
            Eigen::Vector2d u2_init = initial_uv[vh2.idx()];

            // 中心化坐标
            Eigen::Vector2d centroid_uv = (u0 + u1 + u2) / 3.0;
            Eigen::Vector2d centroid_init = (u0_init + u1_init + u2_init) / 3.0;

            Eigen::Matrix2d S =
                (u0_init - centroid_init) * (u0 - centroid_uv).transpose() +
                (u1_init - centroid_init) * (u1 - centroid_uv).transpose() +
                (u2_init - centroid_init) * (u2 - centroid_uv).transpose();

            // SVD分解
            Eigen::JacobiSVD<Eigen::Matrix2d> svd(
                S, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2d U = svd.matrixU();
            Eigen::Matrix2d V = svd.matrixV();

            Eigen::Matrix2d Lt = V * U.transpose();
            if (Lt.determinant() < 0) {
                V.col(1) *= -1;
                Lt = V * U.transpose();
            }
            Lt_list[fh.idx()] = Lt;
        }

        // Step 3: 全局阶段 - 构建线性系统
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplets;
        Eigen::SparseMatrix<double> A(2 * n_vertices, 2 * n_vertices);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * n_vertices);

        // 遍历所有边
        for (auto eh : iter_mesh->edges()) {
            auto he0 = eh.h0();
            auto he1 = eh.h1();

            // 获取相邻面
            std::vector<Eigen::Matrix2d> Lts;
            if (he0.face().is_valid())
                Lts.push_back(Lt_list[he0.face().idx()]);
            if (he1.face().is_valid())
                Lts.push_back(Lt_list[he1.face().idx()]);

            if (Lts.empty())
                continue;

            // 获取顶点
            auto v0 = he0.from();
            auto v1 = he0.to();
            int i = v0.idx();
            int j = v1.idx();

            // 计算余切权重
            double weight = 0.0;
            for (auto he : { he0, he1 }) {
                if (he.face().is_valid()) {
                    auto v_prev = he.prev().from();
                    auto v_curr = he.from();
                    auto v_next = he.to();

                    MyMesh::Point p_prev = halfedge_mesh->point(v_prev);
                    MyMesh::Point p_curr = halfedge_mesh->point(v_curr);
                    MyMesh::Point p_next = halfedge_mesh->point(v_next);

                    weight += compute_cotangent(p_prev, p_curr, p_next);
                }
            }
            weight = std::abs(weight) / 2.0;  // 取绝对值防止负权重

            // 添加到矩阵A
            for (int k = 0; k < 2; ++k) {
                triplets.emplace_back(2 * i + k, 2 * i + k, weight);
                triplets.emplace_back(2 * i + k, 2 * j + k, -weight);
                triplets.emplace_back(2 * j + k, 2 * i + k, -weight);
                triplets.emplace_back(2 * j + k, 2 * j + k, weight);
            }

            // 处理每个面的贡献
            Eigen::Vector2d e_init = initial_uv[j] - initial_uv[i];
            for (auto& Lt : Lts) {
                Eigen::Vector2d term = weight * Lt * e_init;
                b[2 * i] += term.x();
                b[2 * i + 1] += term.y();
                b[2 * j] -= term.x();
                b[2 * j + 1] -= term.y();
            }
        }

        // 求解线性系统
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();

        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed" << std::endl;
            break;
        }

        Eigen::VectorXd x = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solving failed" << std::endl;
            break;
        }

        // 更新顶点坐标
        for (auto vh : iter_mesh->vertices()) {
            int idx = vh.idx();
            iter_mesh->set_point(
                vh, MyMesh::Point(x[2 * idx], x[2 * idx + 1], 0.0));
        }
    }
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
