# Tutte Embedding 算法详解

Tutte Embedding 是一种重要的网格参数化算法，它可以将具有单个边界的网格映射到平面上，确保内部顶点位于其相邻顶点的质心位置。下面将逐段分析 `tutte_embedding` 函数的实现逻辑。

## 1. 函数初始化

```cpp
void tutte_embedding(MyMesh& omesh)
{
    // 初始化
    int n_vertices = omesh.n_vertices();
    std::vector<int> ori2mat(n_vertices, 0);
```

这部分代码完成了初始化工作：
- 获取输入网格的顶点总数 `n_vertices`
- 创建一个映射数组 `ori2mat`，用于建立原始网格顶点索引到线性系统矩阵索引的映射关系，初始值全部设为0

## 2. 边界顶点标记

```cpp
    // 标记边界顶点
    for (const auto& halfedge_handle : omesh.halfedges())
        if (halfedge_handle.is_boundary()) {
            ori2mat[halfedge_handle.to().idx()] = -1;
            ori2mat[halfedge_handle.from().idx()] = -1;
        }
```

这部分代码标记了所有的边界顶点：
- 遍历网格中的所有半边
- 如果某个半边是边界半边（位于网格边界），则将该半边的起点和终点顶点在 `ori2mat` 中标记为 -1
- 边界顶点的位置在 Tutte 嵌入中保持不变（通常会被预先设置为边界形状，如圆形或多边形）

## 3. 内部顶点映射建立

```cpp
    // 构建内部点的索引映射
    int n_internals = 0;
    for (int i = 0; i < n_vertices; i++)
        if (ori2mat[i] != -1)
            ori2mat[i] = n_internals++;
```

这部分代码为内部顶点（非边界顶点）建立新的索引映射：
- 初始化内部顶点计数器 `n_internals` 为0
- 遍历所有顶点，对于非边界顶点（即 `ori2mat[i] != -1` 的顶点）：
  - 将其在 `ori2mat` 中的值设为当前的 `n_internals` 值
  - 然后 `n_internals` 自增
- 最终 `n_internals` 存储了内部顶点的总数量

## 4. 线性系统初始化

```cpp
    Eigen::SparseMatrix<double> A(n_internals, n_internals);
    Eigen::VectorXd bx(n_internals);
    Eigen::VectorXd by(n_internals);
    Eigen::VectorXd bz(n_internals);
```

这部分代码初始化了线性方程组的数据结构：
- `A` 是一个稀疏矩阵，大小为内部顶点数 × 内部顶点数，用于存储系数矩阵
- `bx`, `by`, `bz` 是三个向量，大小均为内部顶点数，分别用于存储 x、y、z 坐标的右侧向量

它们用于求解方程

$$
Ax = b
$$

## 5. 构建线性系统

```cpp
    // 构建系数矩阵和向量
    for (const auto& vertex_handle : omesh.vertices()) {
        int mat_idx = ori2mat[vertex_handle.idx()];
        if (mat_idx == -1)
            continue;
        bx(mat_idx) = 0;
        by(mat_idx) = 0;
        bz(mat_idx) = 0;

        int Aii = 0;
        for (const auto& halfedge_handle : vertex_handle.outgoing_halfedges()) {
            const auto& v1 = halfedge_handle.to();
            int mat_idx1 = ori2mat[v1.idx()];
            Aii++;
            if (mat_idx1 == -1) {
                // 边界点对右侧向量的贡献
                bx(mat_idx) += omesh.point(v1)[0];
                by(mat_idx) += omesh.point(v1)[1];
                bz(mat_idx) += omesh.point(v1)[2];
            }
            else
                // 内部点对系数矩阵的贡献
                A.coeffRef(mat_idx, mat_idx1) = -1;
        }
        A.coeffRef(mat_idx, mat_idx) = Aii;
    }
```

这部分代码构建了 Laplace 方程的离散形式，表达了 Tutte 嵌入的核心约束：每个内部顶点坐标都是其相邻顶点的凸线性组合。此处我们使用均匀权重。

对于每个内部顶点$v_i$，记其相邻的内部顶点为$v_{in1}, v_{in2}, \ldots, v_{inm}$，相邻的边界顶点$v_{b1}, v_{b2}, \ldots, v_{bn}$，则有以下约束关系：
$$
v_i = \frac{1}{m+n} \sum_{j=1}^{m} v_{inj} + \frac{1}{m+n} \sum_{k=1}^{n} v_{b_k}
$$

改写为$A_ix=b_i$的形式，让$A_{ii}$表示顶点$v_i$的度数，$A_{ij}$表示顶点$i$与相邻顶点$j$的关系，则有：

$$
A_{ij} = \begin{cases}
    -1 & \text{if } j \text{ 是 } v_i \text{ 的相邻顶点} \\
    degree(v_i) & \text{if } i = j \\
    0 & \text{otherwise}
\end{cases}
$$

$$
b_i = \sum_{k=1}^{n} v_{b_k} 
$$

具体实现：
- 遍历所有顶点，跳过边界顶点（`mat_idx == -1`）
- 对于每个内部顶点：
  - 初始化其在右侧向量 `bx`, `by`, `bz` 中的值为0
  - 初始化顶点的度数计数器 `Aii` 为0
  - 遍历该顶点的所有出射半边，获取相邻顶点 `v1`：
    - 增加度数计数 `Aii++`
    - 如果相邻顶点是边界顶点（`mat_idx1 == -1`）：
      - 将其坐标添加到当前内部顶点对应的右侧向量 `bx`, `by`, `bz` 中
    - 如果相邻顶点是内部顶点：
      - 设置系数矩阵 `A` 中对应位置的值为 -1，表示相邻关系
  - 设置系数矩阵 `A` 对角线上的元素为顶点的度数 `Aii`

这一步构建的线性方程组实质上是：A·x = b，其中 x 是内部顶点的坐标，b 包含了边界顶点对内部顶点的影响。

## 6. 求解线性方程组

```cpp
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        solver;
    solver.compute(A);
    Eigen::VectorXd ux = bx;
    ux = solver.solve(ux);
    Eigen::VectorXd uy = by;
    uy = solver.solve(uy);
    Eigen::VectorXd uz = bz;
    uz = solver.solve(uz);
```

这部分代码求解线性方程组：
- 创建一个 SparseLU 求解器，使用 COLAMDOrdering 优化稀疏矩阵计算
- 对矩阵 `A` 进行分解计算
- 分别求解 x、y、z 三个坐标的线性方程组：A·ux = bx、A·uy = by、A·uz = bz
- 结果存储在 `ux`、`uy`、`uz` 向量中，它们包含了所有内部顶点的新坐标

## 7. 更新顶点位置

```cpp
    // 更新顶点新位置
    for (const auto& vertex_handle : omesh.vertices()) {
        int idx = ori2mat[vertex_handle.idx()];
        if (idx != -1) {
            omesh.point(vertex_handle)[0] = ux(idx);
            omesh.point(vertex_handle)[1] = uy(idx);
            omesh.point(vertex_handle)[2] = uz(idx);
        }
    }
}
```

这部分代码将计算得到的新坐标更新到网格中：
- 遍历所有顶点
- 对于内部顶点（`idx != -1`），将其坐标更新为求解得到的新坐标
- 边界顶点的位置保持不变

## 总结

Tutte Embedding 算法通过求解 Laplace 方程，使得每个内部顶点位于其相邻顶点的几何中心，从而得到一个无翻转、无重叠的网格平面参数化。算法的关键步骤包括：
1. 识别并固定边界顶点（通常设为凸多边形，例如圆或正多边形）
2. 构建线性方程组，使每个内部顶点等于其相邻顶点的平均位置
3. 求解线性方程组获取内部顶点的新坐标
4. 更新内部顶点的位置

该实现保持了 3D 坐标系统，但典型的 Tutte Embedding 通常会将网格映射到 2D 平面（即 z 坐标设为0）。