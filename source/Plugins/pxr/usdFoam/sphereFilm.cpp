//
// Copyright 2016 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
#include "pxr/usd/usdFoam/sphereFilm.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<UsdFoamSphereFilm,
        TfType::Bases< UsdGeomPointBased > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("SphereFilm")
    // to find TfType<UsdFoamSphereFilm>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, UsdFoamSphereFilm>("SphereFilm");
}

/* virtual */
UsdFoamSphereFilm::~UsdFoamSphereFilm()
{
}

/* static */
UsdFoamSphereFilm
UsdFoamSphereFilm::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return UsdFoamSphereFilm();
    }
    return UsdFoamSphereFilm(stage->GetPrimAtPath(path));
}

/* static */
UsdFoamSphereFilm
UsdFoamSphereFilm::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("SphereFilm");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return UsdFoamSphereFilm();
    }
    return UsdFoamSphereFilm(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind UsdFoamSphereFilm::_GetSchemaKind() const
{
    return UsdFoamSphereFilm::schemaKind;
}

/* static */
const TfType &
UsdFoamSphereFilm::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<UsdFoamSphereFilm>();
    return tfType;
}

/* static */
bool 
UsdFoamSphereFilm::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
UsdFoamSphereFilm::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
UsdFoamSphereFilm::GetSphereCentersAttr() const
{
    return GetPrim().GetAttribute(UsdFoamTokens->sphereCenters);
}

UsdAttribute
UsdFoamSphereFilm::CreateSphereCentersAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(UsdFoamTokens->sphereCenters,
                       SdfValueTypeNames->Point3fArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
UsdFoamSphereFilm::GetSphereRadiiAttr() const
{
    return GetPrim().GetAttribute(UsdFoamTokens->sphereRadii);
}

UsdAttribute
UsdFoamSphereFilm::CreateSphereRadiiAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(UsdFoamTokens->sphereRadii,
                       SdfValueTypeNames->FloatArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
UsdFoamSphereFilm::GetPolygonIndicesAttr() const
{
    return GetPrim().GetAttribute(UsdFoamTokens->polygonIndices);
}

UsdAttribute
UsdFoamSphereFilm::CreatePolygonIndicesAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(UsdFoamTokens->polygonIndices,
                       SdfValueTypeNames->IntArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
UsdFoamSphereFilm::GetPolygonPointsAttr() const
{
    return GetPrim().GetAttribute(UsdFoamTokens->polygonPoints);
}

UsdAttribute
UsdFoamSphereFilm::CreatePolygonPointsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(UsdFoamTokens->polygonPoints,
                       SdfValueTypeNames->Point3fArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

namespace {
static inline TfTokenVector
_ConcatenateAttributeNames(const TfTokenVector& left,const TfTokenVector& right)
{
    TfTokenVector result;
    result.reserve(left.size() + right.size());
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
    return result;
}
}

/*static*/
const TfTokenVector&
UsdFoamSphereFilm::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        UsdFoamTokens->sphereCenters,
        UsdFoamTokens->sphereRadii,
        UsdFoamTokens->polygonIndices,
        UsdFoamTokens->polygonPoints,
    };
    static TfTokenVector allNames =
        _ConcatenateAttributeNames(
            UsdGeomPointBased::GetSchemaAttributeNames(true),
            localNames);

    if (includeInherited)
        return allNames;
    else
        return localNames;
}

PXR_NAMESPACE_CLOSE_SCOPE

// ===================================================================== //
// Feel free to add custom code below this line. It will be preserved by
// the code generator.
//
// Just remember to wrap code in the appropriate delimiters:
// 'PXR_NAMESPACE_OPEN_SCOPE', 'PXR_NAMESPACE_CLOSE_SCOPE'.
// ===================================================================== //
// --(BEGIN CUSTOM CODE)--
