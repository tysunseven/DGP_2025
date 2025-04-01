# OpenMesh双边法线滤波算法解析

## 基本结构

首先定义了使用OpenMesh库的三角网格类型：
```cpp
typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;
```

## 核心函数解析

### 1. 面积计算函数 - getFaceArea

```cpp
void getFaceArea(MyMesh &mesh, std::vector<float> &area)
```

该函数计算网格中每个三角面的面积：
- 遍历每个面，获取其三个顶点
- 计算两条边的向量
- 使用叉乘计算三角形面积：S = 0.5 * |edge1 × edge2|
- 其中 `%` 运算符表示向量叉积

### 2. 面质心计算函数 - getFaceCentroid

```cpp
void getFaceCentroid(MyMesh &mesh, std::vector<MyMesh::Point> &centroid)
```

计算每个面的质心位置，直接调用OpenMesh的`calc_face_centroid`方法。

### 3. 面法线计算函数 - getFaceNormal

```cpp
void getFaceNormal(MyMesh &mesh, std::vector<MyMesh::Normal> &normals)
```

- 请求和更新面法线
- 遍历所有面获取其法线向量

### 4. 邻接面查找函数 - getFaceNeighbor

```cpp
void getFaceNeighbor(MyMesh &mesh, MyMesh::FaceHandle fh, std::vector<MyMesh::FaceHandle> &face_neighbor)
```

通过以下步骤查找给定面的邻接面：
- 遍历给定面的所有顶点
- 对于每个顶点，查找与其相关的所有面
- 排除当前面本身，将其他面添加到邻居集合中
- 使用`std::set`确保不会添加重复的面

### 5. 全局邻接关系构建函数 - getAllFaceNeighbor

```cpp
void getAllFaceNeighbor(MyMesh &mesh, std::vector<std::vector<MyMesh::FaceHandle>> &all_face_neighbor, bool include_central_face)
```

为网格中的所有面构建邻接关系，`include_central_face`控制是否将中心面本身也包含在邻居列表中。

### 6. 顶点位置更新函数 - updateVertexPosition

```cpp
void updateVertexPosition(MyMesh &mesh, std::vector<MyMesh::Normal> &filtered_normals, int iteration_number, bool fixed_boundary)
```

根据过滤后的法线更新顶点位置：
- 迭代指定次数，每次迭代使网格逐渐向目标表面逼近
- 在每次迭代中：
  - 计算面质心作为参考点
  - 对每个顶点：
    - 如果是边界顶点且`fixed_boundary=true`，则保持不变（保护模型边界形状）
    - 否则，计算顶点位移：
      ```cpp
      temp_point += temp_normal * (temp_normal | (temp_centroid - p));
      ```
      这个公式的意义是：
      1. `temp_centroid - p`: 顶点到面质心的向量
      2. `temp_normal | (...)`: 将该向量投影到法线方向上，得到点到面的有符号距离
      3. `temp_normal * (...)`: 将该距离转换为法线方向的位移向量
      4. 对所有相邻面的这种位移求平均，作为顶点的总体位移
  - 批量更新所有顶点位置，让网格逐步向目标曲面演化

这个过程实际上是一种网格变形，使顶点沿过滤后的法线方向移动，最终使网格表面更符合平滑的曲面特性。

### 7. 空间域参数计算函数 - getSigmaC

```cpp
float getSigmaC(MyMesh &mesh, std::vector<MyMesh::Point> &face_centroid, float multiple_sigma_c)
```

计算空间域滤波的标准差参数sigma_c：
- 计算相邻面质心之间的平均距离
- 乘以用户指定的倍数作为空间域参数

### 8. 法线过滤核心函数 - update_filtered_normals_local_scheme

```cpp
void update_filtered_normals_local_scheme(MyMesh &mesh, std::vector<MyMesh::Normal> &filtered_normals, float multiple_sigma_c, int normal_iteration_number, float sigma_s)
```

实现局部双边滤波方案更新面法线，这是算法的核心：
- 收集网格的邻接关系、原始法线、面积和质心信息
- 计算空间域参数sigma_c作为距离权重因子
- 迭代进行法线滤波（迭代能够增强滤波效果）：
  - 对每个面i，通过其邻近面j的法线加权平均来更新法线：
  
  ```cpp
  // 权重计算
  float spatial_distance = (ci - cj).length();  // 质心间距离
  float spatial_weight = std::exp(-0.5 * spatial_distance * spatial_distance / (sigma_c * sigma_c));  // 空间域权重
  float range_distance = (ni - nj).length();  // 法线差异
  float range_weight = std::exp(-0.5 * range_distance * range_distance / (sigma_s * sigma_s));  // 值域权重
  float weight = face_area[index_j] * spatial_weight * range_weight;  // 总权重 = 面积 × 空间权重 × 值域权重
  ```

    ![空间域权重](../../images/analysis_3_1.png)

    ![值域权重](../../images/analysis_3_2.png)

    ![新法线](../../images/analysis_3_3.png)
  
  - 关键计算包括：
    - **面积权重**：面积越大的面贡献越大
    - **空间域权重**：基于面质心距离的高斯衰减，距离越近影响越大
    - **值域权重**：基于法线方向差异的高斯衰减，法线方向越接近影响越大
  - 根据这三种权重的乘积进行加权平均，然后归一化得到新法线
  - 每轮迭代后用新计算的法线替换原法线，用于下一轮计算

该函数通过局部加权平均实现了法线的自适应平滑，能够在保留特征的同时减少噪声。

### 9. 主滤波函数 - bilateral_normal_filtering

```cpp
void bilateral_normal_filtering(MyMesh &mesh, float sigma_s, int normal_iteration_number, float multiple_sigma_c)
```

双边法线滤波的主函数，整合整个处理流程：
- 首先过滤法线
- 然后根据过滤后的法线更新顶点位置

## 算法原理

双边法线滤波是结合空间域和值域滤波的技术：
- **空间域滤波**：考虑面之间的几何距离，距离近的面影响更大
- **值域滤波**：考虑法线方向差异，方向相似的面影响更大

这种方法能够有效地平滑噪声同时保留锐利特征，因为：
- 在平坦区域：法线相似，充分平滑
- 在边缘特征处：法线差异大，保留边缘

## 双边滤波原理深入解析

双边滤波（Bilateral Filtering）是一种非线性、边缘保持的平滑滤波器，在这个网格处理算法中被应用于法线向量的平滑：

### 基本思想

1. **组合滤波**：双边滤波同时考虑两个域的权重：
   - 空间域（Spatial Domain）：基于几何距离
   - 值域（Range Domain）：基于特征值差异（这里是法线方向）

2. **权重函数**：通常使用高斯函数来计算权重，如代码中：
   ```cpp
   exp(-0.5 * distance² / sigma²)
   ```
   
   其中sigma参数控制权重衰减的速率：
   - sigma_c：控制空间域高斯函数的宽度，决定距离影响范围
   - sigma_s：控制值域高斯函数的宽度，决定法线差异容忍度

### 为什么能保持特征

双边法线滤波在网格处理中的优势在于：

1. **自适应性**：在平坦区域（法线相似），主要受空间域权重影响，充分平滑；在边缘区域（法线差异大），值域权重迅速降低，减少平滑，从而保留特征

2. **局部性**：考虑每个面的邻域，根据局部特征决定滤波强度，适应不同区域的几何特点

3. **迭代优化**：通过多次迭代，逐步改善法线分布，避免过度平滑

### 参数影响

- **sigma_s（值域参数）**：控制法线方向差异敏感度
  - 值越小，越能保留细微特征但噪声去除效果较弱
  - 值越大，平滑效果越强，但可能模糊锐利特征

- **sigma_c（空间域参数）**：控制空间影响范围
  - 通常根据网格尺度自适应计算（如代码中的getSigmaC函数）
  - 由multiple_sigma_c参数调节

- **迭代次数**：影响平滑程度和计算成本
  - 次数越多，效果越明显，但计算时间增加

通过法线滤波和顶点位置更新的组合应用，算法能有效地改善网格质量，平滑噪声的同时保留模型的重要几何特征。
