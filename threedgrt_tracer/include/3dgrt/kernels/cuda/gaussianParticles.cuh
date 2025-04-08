// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <optix.h>

#include <3dgrt/mathUtils.h>
#include <3dgrt/particleDensity.h>

void rotationMatrixTranspose(const float4& q, float33& ret) {
    const float r = q.x;
    const float x = q.y;
    const float y = q.z;
    const float z = q.w;

    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float rx = r * x;
    const float ry = r * y;
    const float rz = r * z;

    // Compute rotation matrix from quaternion
    ret[0] = make_float3((1.f - 2.f * (yy + zz)), 2.f * (xy + rz), 2.f * (xz - ry));
    ret[1] = make_float3(2.f * (xy - rz), (1.f - 2.f * (xx + zz)), 2.f * (yz + rx));
    ret[2] = make_float3(2.f * (xz + ry), 2.f * (yz - rx), (1.f - 2.f * (xx + yy)));
}

static constexpr float SH_C0   = 0.28209479177387814f;
static constexpr float SH_C1   = 0.4886025119029199f;
static constexpr float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                  -1.0925484305920792f, 0.5462742152960396f};
static constexpr float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                  -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

static inline __device__ float3
radianceFromSpH(int deg, const float3* sphCoefficients, const float3& rdir, bool clamped = true) {
    float3 rad = SH_C0 * sphCoefficients[0];
    if (deg > 0) {
        const float3& dir = rdir;

        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;
        rad           = rad - SH_C1 * y * sphCoefficients[1] + SH_C1 * z * sphCoefficients[2] -
              SH_C1 * x * sphCoefficients[3];

        if (deg > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            rad = rad + SH_C2[0] * xy * sphCoefficients[4] + SH_C2[1] * yz * sphCoefficients[5] +
                  SH_C2[2] * (2.0f * zz - xx - yy) * sphCoefficients[6] +
                  SH_C2[3] * xz * sphCoefficients[7] + SH_C2[4] * (xx - yy) * sphCoefficients[8];

            if (deg > 2) {
                rad = rad + SH_C3[0] * y * (3.0f * xx - yy) * sphCoefficients[9] +
                      SH_C3[1] * xy * z * sphCoefficients[10] +
                      SH_C3[2] * y * (4.0f * zz - xx - yy) * sphCoefficients[11] +
                      SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sphCoefficients[12] +
                      SH_C3[4] * x * (4.0f * zz - xx - yy) * sphCoefficients[13] +
                      SH_C3[5] * z * (xx - yy) * sphCoefficients[14] +
                      SH_C3[6] * x * (xx - 3.0f * yy) * sphCoefficients[15];
            }
        }
    }
    rad += 0.5f;
    return clamped ? maxf3(rad, make_float3(0.0f)) : rad;
}

static inline __device__ void addSphCoeffGrd(float3* sphCoefficientsGrad, int idx, const float3& val) {
    atomicAdd(&sphCoefficientsGrad[idx].x, val.x);
    atomicAdd(&sphCoefficientsGrad[idx].y, val.y);
    atomicAdd(&sphCoefficientsGrad[idx].z, val.z);
}

static inline __device__ float3 radianceFromSpHBwd(
    int deg, const float3* sphCoefficients, const float3& rdir, float weight, const float3& rayRadGrd, float3* sphCoefficientsGrad) {
    // radiance unclamped
    const float3 gradu = radianceFromSpH(deg, sphCoefficients, rdir, false);

    // clamped radiance
    float3 grad = make_float3(gradu.x > 0.0f ? gradu.x : 0.0f,
                              gradu.y > 0.0f ? gradu.y : 0.0f,
                              gradu.z > 0.0f ? gradu.z : 0.0f);

    //
    float3 dL_dRGB = rayRadGrd * weight;
    dL_dRGB.x *= (gradu.x > 0.0f ? 1 : 0);
    dL_dRGB.y *= (gradu.y > 0.0f ? 1 : 0);
    dL_dRGB.z *= (gradu.z > 0.0f ? 1 : 0);

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    // ---> rayRad = weight * grad = weight * explu(gsph0 * SH_C0 +
    // 0.5,SHRadMinBound) with explu(x,a) = x if x > a else a*e(x-a)
    // ===> d_rayRad / d_gsph0 =   weight * SH_C0
    addSphCoeffGrd(sphCoefficientsGrad, 0, SH_C0 * dL_dRGB);

    if (deg > 0) {
        // const float3 sphdiru = gpos - rori;
        // const float3 sphdir = safe_normalize(sphdiru);
        const float3& sphdir = rdir;

        float x = sphdir.x;
        float y = sphdir.y;
        float z = sphdir.z;

        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;

        addSphCoeffGrd(sphCoefficientsGrad, 1, dRGBdsh1 * dL_dRGB);
        addSphCoeffGrd(sphCoefficientsGrad, 2, dRGBdsh2 * dL_dRGB);
        addSphCoeffGrd(sphCoefficientsGrad, 3, dRGBdsh3 * dL_dRGB);

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);

            addSphCoeffGrd(sphCoefficientsGrad, 4, dRGBdsh4 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 5, dRGBdsh5 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 6, dRGBdsh6 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 7, dRGBdsh7 * dL_dRGB);
            addSphCoeffGrd(sphCoefficientsGrad, 8, dRGBdsh8 * dL_dRGB);

            if (deg > 2) {
                float dRGBdsh9  = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);

                addSphCoeffGrd(sphCoefficientsGrad, 9, dRGBdsh9 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 10, dRGBdsh10 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 11, dRGBdsh11 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 12, dRGBdsh12 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 13, dRGBdsh13 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 14, dRGBdsh14 * dL_dRGB);
                addSphCoeffGrd(sphCoefficientsGrad, 15, dRGBdsh15 * dL_dRGB);
            }
        }
    }

    return grad;
}

static inline __device__ void fetchParticleDensity(
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    float3& particlePosition,
    float3& particleScale,
    float33& particleInvRotation,
    float& particleDensity) {
    const ParticleDensity particleData = particlesDensity[particleIdx];

    particlePosition = particleData.position;
    particleScale    = particleData.scale;
    rotationMatrixTranspose(particleData.quaternion, particleInvRotation);
    particleDensity = particleData.density;
}

static inline __device__ void fetchParticleSphCoefficients(
    const int32_t particleIdx,
    const float* particlesSphCoefficients,
    float3* sphCoefficients) {
    const uint32_t particleOffset = particleIdx * SPH_MAX_NUM_COEFFS * 3;
#pragma unroll
    for (unsigned int i = 0; i < SPH_MAX_NUM_COEFFS; ++i) {
        const int offset   = i * 3;
        sphCoefficients[i] = make_float3(
            particlesSphCoefficients[particleOffset + offset + 0],
            particlesSphCoefficients[particleOffset + offset + 1],
            particlesSphCoefficients[particleOffset + offset + 2]);
    }
}

template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponseGrd(float grayDist, float gres, float gresGrd) {
    /// generalized gaussian of degree b : scaling a = -4.5/3^b
    /// d_e^{a*|x|^b}/d_x^2 = a*(0.5*b)*x^{b-2}*e^{a*|x|^b}
    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        constexpr float s      = -0.000685871056241 * (0.5f * 8);
        const float grayDistSq = grayDist * grayDist;
        return s * grayDistSq * grayDist * gres * gresGrd;
    }
    case 5: // Quintic
    {
        constexpr float s = -0.0185185185185 * (0.5f * 5);
        return s * grayDist * sqrtf(grayDist) * gres * gresGrd;
    }
    case 4: // Tesseractic
    {
        constexpr float s = -0.0555555555556 * (0.5f * 4);
        return s * grayDist * gres * gresGrd;
    }
    case 3: // Cubic
    {
        constexpr float s = -0.166666666667 * (0.5f * 3);
        return s * sqrtf(grayDist) * gres * gresGrd;
    }
    case 1: // Laplacian
    {
        constexpr float s = -1.5f * (0.5f * 1);
        return s * sqrtf(grayDist) * gres * gresGrd;
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487;
        return gres > 0.f ? (0.5f * s * rsqrtf(grayDist)) * gresGrd : 0.f;
    }
    default: // Quadratic
    {
        constexpr float s = -0.5f;
        return s * gres * gresGrd;
    }
    }
}

template <int GeneralizedGaussianDegree = 2>
static inline __device__ float particleResponse(float grayDist) {
    /// generalized gaussian of degree n : scaling is s = -4.5/3^n
    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        constexpr float s      = -0.000685871056241f;
        const float grayDistSq = grayDist * grayDist;
        return expf(s * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        constexpr float s = -0.0185185185185f;
        return expf(s * grayDist * grayDist * sqrtf(grayDist));
    }
    case 4: // Tesseractic
    {
        constexpr float s = -0.0555555555556f;
        return expf(s * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        constexpr float s = -0.166666666667f;
        return expf(s * grayDist * sqrtf(grayDist));
    }
    case 1: // Laplacian
    {
        constexpr float s = -1.5f;
        return expf(s * sqrtf(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = -0.329630334487f;
        return fmaxf(1.f + s * sqrtf(grayDist), 0.f);
    }
    default: // Quadratic
    {
        constexpr float s = -0.5f;
        return expf(s * grayDist);
    }
    }
}

template <int GeneralizedGaussianDegree = 2, bool clamped>
static inline __device__ float particleScaledResponse(float grayDist, float modulatedMinResponse, float responseModulation = 1.0f) {

    const float minResponse    = fminf(modulatedMinResponse / responseModulation, 0.97f);
    const float logMinResponse = clamped ? logf(minResponse) : modulatedMinResponse;

    switch (GeneralizedGaussianDegree) {
    case 8: // Zenzizenzizenzic
    {
        const float grayDistSq = grayDist * grayDist;
        return expf(logMinResponse * grayDistSq * grayDistSq);
    }
    case 5: // Quintic
    {
        return expf(logMinResponse * grayDist * grayDist * sqrtf(grayDist));
    }
    case 4: // Tesseractic
    {
        return expf(logMinResponse * grayDist * grayDist);
    }
    case 3: // Cubic
    {
        return expf(logMinResponse * grayDist * sqrtf(grayDist));
    }
    case 1: // Laplacian
    {
        return expf(logMinResponse * sqrtf(grayDist));
    }
    case 0: // Linear
    {
        /* static const */ float s = (1.0f - minResponse) / 3.0f;
        return fmaxf(1.f + s * sqrtf(grayDist), 0.f);
    }
    default: // Quadratic
    {
        return expf(logMinResponse * grayDist);
    }
    }
}

template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false>
__device__ inline bool processHit(
    const float3& rayOrigin,
    const float3& rayDirection,
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    const float* particlesSphCoefficients,
    const float minParticleKernelDensity,
    const float minParticleAlpha,
    const int32_t sphEvalDegree,
    float* transmittance,
    float3* radiance,
    float* depth,
    float3* normal) {
    float3 particlePosition;
    float3 particleScale;
    float33 particleInvRotation;
    float particleDensity;

    fetchParticleDensity(
        particleIdx,
        particlesDensity,
        particlePosition,
        particleScale,
        particleInvRotation,
        particleDensity);

    const float3 giscl   = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gposcr  = (gposc * particleInvRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = rayDirection * particleInvRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);

    const float3 gcrod   = SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro);
    const float grayDist = dot(gcrod, gcrod);

    const float gres   = particleResponse<ParticleKernelDegree>(grayDist);
    const float galpha = fminf(0.99f, gres * particleDensity);

    const bool acceptHit = (gres > minParticleKernelDensity) && (galpha > minParticleAlpha);
    if (acceptHit) {
        const float weight = galpha * (*transmittance);

        // distance to the gaussian center projection on the ray
        const float3 grds = particleScale * grd * (SurfelPrimitive ? -gro.z / grd.z : dot(grd, -1 * gro));
        const float hitT  = sqrtf(dot(grds, grds));

        // radiance from sph coefficients
        float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
        fetchParticleSphCoefficients(
            particleIdx,
            particlesSphCoefficients,
            &sphCoefficients[0]);
        const float3 grad = radianceFromSpH(sphEvalDegree, &sphCoefficients[0], rayDirection);

        *radiance += grad * weight;
        *transmittance *= (1 - galpha);
        *depth += hitT * weight;

        if (normal) {
            constexpr float ellispoidSqRadius = 9.0f;
            const float3 particleScaleRotated = (particleInvRotation * particleScale);
            *normal += weight * (SurfelPrimitive ? make_float3(0, 0, (grd.z > 0 ? 1 : -1) * particleScaleRotated.z) : safe_normalize((gro + grd * (dot(grd, -1 * gro) - sqrtf(ellispoidSqRadius - grayDist))) * particleScaleRotated));
        }
    }

    return acceptHit;
}

__device__ inline bool intersectCustomParticle(
    const float3& rayOrigin,
    const float3& rayDirection,
    const int32_t particleIdx,
    const ParticleDensity* particlesDensity,
    const float minHitDistance,
    const float maxHitDistance,
    const float maxParticleSquaredDistance,
    float& hitDistance) {
    float3 particlePosition;
    float3 particleScale;
    float33 particleInvRotation;
    float particleDensity;
    fetchParticleDensity(
        particleIdx,
        particlesDensity,
        particlePosition,
        particleScale,
        particleInvRotation,
        particleDensity);

    const float3 giscl   = make_float3(1 / particleScale.x, 1 / particleScale.y, 1 / particleScale.z);
    const float3 gposc   = (rayOrigin - particlePosition);
    const float3 gposcr  = (gposc * particleInvRotation);
    const float3 gro     = giscl * gposcr;
    const float3 rayDirR = rayDirection * particleInvRotation;
    const float3 grdu    = giscl * rayDirR;
    const float3 grd     = safe_normalize(grdu);

    // distance to the gaussian center projection on the ray
    const float grp   = -dot(grd, gro);
    const float3 grds = particleScale * grd * grp;
    hitDistance       = (grp < 0.f ? -1.f : 1.f) * sqrtf(dot(grds, grds));

    if ((hitDistance > minHitDistance) && (hitDistance < maxHitDistance)) {
        const float3 gcrod   = cross(grd, gro);
        const float grayDist = dot(gcrod, gcrod);
        return (grayDist < maxParticleSquaredDistance);
    }
    return false;
}

__device__ inline bool intersectInstanceParticle(
    const float3& particleRayOrigin,
    const float3& particleRayDirection,
    const int32_t particleIdx,
    const float minHitDistance,
    const float maxHitDistance,
    const float maxParticleSquaredDistance,
    float& hitDistance) {
    const float numerator   = -dot(particleRayOrigin, particleRayDirection);
    const float denominator = 1.f / dot(particleRayDirection, particleRayDirection);
    hitDistance             = numerator * denominator;
    if ((hitDistance > minHitDistance) && (hitDistance < maxHitDistance)) {
        const float3 gcrod = cross(safe_normalize(particleRayDirection), particleRayOrigin);
        return (dot(gcrod, gcrod) * denominator < maxParticleSquaredDistance);
    }
    return false;
}

template <int ParticleKernelDegree = 4, bool SurfelPrimitive = false>
__device__ inline void processHitBwd(
    const float3& rayOrigin,
    const float3& rayDirection,
    int32_t particleIdx,
    const ParticleDensity* particleDensityPtr,
    ParticleDensity* particleDensityGradPtr,
    const float* particleRadiancePtr,
    float* particleRadianceGradPtr,
    float minParticleKernelDensity,
    float minParticleAlpha,
    float minTransmittance,
    int32_t sphEvalDegree,
    float integratedTransmittance,
    float& transmittance,
    float transmittanceGrad,
    float3 integratedRadiance,
    float3& radiance,
    float3 radianceGrad,
    float integratedDepth,
    float& depth,
    float depthGrad,
    float3* rayOriginGrad = nullptr,
    float3* rayDirectionGrad = nullptr) {
    float3 particlePosition;
    float3 gscl;
    float33 particleInvRotation;
    float particleDensity;
    float4 grot;

    {
        const ParticleDensity particleData = particleDensityPtr[particleIdx];
        particlePosition                   = particleData.position;
        gscl                               = particleData.scale;
        grot                               = particleData.quaternion;
        rotationMatrixTranspose(grot, particleInvRotation);
        particleDensity = particleData.density;
    }

    // project ray in the gaussian
    // giscl = Gaussian Inverse Scale，高斯粒子的逆尺度向量
    const float3 giscl   = make_float3(1 / gscl.x, 1 / gscl.y, 1 / gscl.z);
    // gposc = Gaussian Position Center，射线原点到高斯粒子中心的向量
    const float3 gposc   = (rayOrigin - particlePosition);
    // gposcr = Gaussian Position Center Rotated，旋转后的射线原点到高斯粒子中心的向量
    const float3 gposcr  = (gposc * particleInvRotation);
    // gro = Gaussian Ray Origin，高斯粒子本地坐标系中的射线原点
    const float3 gro     = giscl * gposcr;
    // rayDirR = Ray Direction Rotated，旋转后的射线方向
    const float3 rayDirR = rayDirection * particleInvRotation;
    // grdu = Gaussian Ray Direction Unscaled，高斯粒子本地坐标系中未归一化的射线方向
    const float3 grdu    = giscl * rayDirR;
    // grd = Gaussian Ray Direction，高斯粒子本地坐标系中归一化的射线方向
    const float3 grd     = safe_normalize(grdu);
    // gcrod = Gaussian Cross Ray Origin Direction，计算射线与高斯粒子中心的距离向量
    // 对于体积粒子：使用射线方向与射线原点的叉积
    // 对于表面粒子：计算射线与表面的交点位置
    const float3 gcrod   = SurfelPrimitive ? gro + grd * -gro.z / grd.z : cross(grd, gro); // 表示射线与高斯粒子中心的平方距离
    // grayDist = Gaussian ray Distance，射线与高斯粒子中心的平方距离
    const float grayDist = dot(gcrod, gcrod);

    
    // gres = Gaussian Response，高斯核函数对该距离的响应值    
    const float gres   = particleResponse<ParticleKernelDegree>(grayDist);
    // galpha = Gaussian Alpha，高斯粒子在该点的不透明度
    const float galpha = fminf(0.99f, gres * particleDensity);

    if ((gres > minParticleKernelDensity) && (galpha > minParticleAlpha)) {
        ParticleDensity& particleDensityGrad = particleDensityGradPtr[particleIdx];

        // grdd = Gaussian Ray Direction Displacement，射线方向上的位移量
        const float3 grdd   = grd * (SurfelPrimitive ? -gro.z / grd.z : dot(grd, -1 * gro));
        // grds = Gaussian Ray Direction Scaled，尺度缩放后的射线方向位移
        const float3 grds   = gscl * grdd;
        // gsqdist = Gaussian Squared Distance，平方距离
        const float gsqdist = dot(grds, grds);
        // gdist = Gaussian Distance，距离值
        const float gdist   = sqrtf(gsqdist);

        const float weight = galpha * transmittance;

        const float nextTransmit = (1 - galpha) * transmittance;
        // prevTrm 是 "previous transmittance"（之前的透射率）的缩写。在高斯粒子渲染的上下文中，它表示光线到达当前粒子之前累积的透明度。
        // 具体来说：
        // transmittance 变量表示射线在当前位置的累积透明度
        // 当射线穿过场景中的多个高斯粒子时，每次遇到一个粒子，透明度都会被更新
        // transmittance *= (1 - galpha)，即透明度会乘以 (1 - 当前粒子的不透明度)
        // 在注释中的公式里：
        // hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
        // prevTrm 就是当前处理的粒子之前的累积透明度值
        // galpha * prevTrm 是当前粒子的权重贡献
        // (1-galpha) * prevTrm 是穿过当前粒子后的剩余透明度
        // 这个透明度值非常重要，因为它决定了每个粒子对最终图像的贡献程度。透明度越高（越接近1），表示之前遇到的粒子越透明，当前粒子的贡献就越大；透明度越低，表示之前的粒子已经阻挡了大部分光线，当前粒子的贡献就越小。
        // 这是体积渲染中的关键概念，用于实现半透明物体的正确混合和深度感知。
        // ---> hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
        depth += weight * gdist;
        const float residualHitT =
            fmaxf((nextTransmit <= minTransmittance ? 0 : (integratedDepth - depth) / nextTransmit),
                  0);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> hitT = accumulatedHitT + galpha * prevTrm * gdist + (1-galpha) * prevTrm * residualHitT
        //
        // ===> d_hitT / d_galpha = gdist * prevTrm - residualHitT * prevTrm
        //                        = (gdist - residualHitT) * prevTrm
        //
        const float galphaRayHitGrd = (gdist - residualHitT) * transmittance * depthGrad;
        //
        // ===> d_hitT / d_gsqdist = weight / (2*gdist)
        // ===> d_gsqdist / d_grds =  2 * grds
        const float3 grdsRayHitGrd = gsqdist > 0.0f ? ((2 * grds * weight) / (2 * gdist)) * depthGrad : make_float3(0.0f);

        // ---> grds = gscl * grd * dot(grd, -1 * gro)
        //
        // ===> d_grds / d_gscl =  grd * dot(grd, -1 * gro)
        const float3 gsclRayHitGrd = grdd * grdsRayHitGrd;
        // ===> d_grds / d_grd =  - gscl * grd * (2 dot(grd, -1 * gro)
        const float3 grdRayHitGrd = -gscl * make_float3(2 * grd.x * gro.x + grd.y * gro.y + grd.z * gro.z, grd.x * gro.x + 2 * grd.y * gro.y + grd.z * gro.z, grd.x * gro.x + grd.y * gro.y + 2 * grd.z * gro.z) * grdsRayHitGrd;
        //
        // ===> d_grds / d_gro = - gscl * grd * grd
        const float3 groRayHitGrd = -gscl * grd * grd * grdsRayHitGrd;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_galpha = prevTrm * nextTrm = residualTrm
        const float residualTrm     = galpha < 0.999999f ? integratedTransmittance / (1 - galpha) : transmittance;
        const float galphaRayDnsGrd = residualTrm * -transmittanceGrad;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // compute the gradient wrt to the sph coefficients and position (through the sph view
        // direction)
        float3 sphCoefficients[SPH_MAX_NUM_COEFFS];
        fetchParticleSphCoefficients(
            particleIdx,
            particleRadiancePtr,
            &sphCoefficients[0]);
        const float3 grad = radianceFromSpHBwd(sphEvalDegree, &sphCoefficients[0], rayDirection, weight, radianceGrad, (float3*)&particleRadianceGradPtr[particleIdx * SPH_MAX_NUM_COEFFS * 3]);

        // >>> rayRadiance = accumulatedRayRad + weigth * rayRad + (1-galpha)*transmit * residualRayRad
        const float3 rayRad = weight * grad;
        radiance += rayRad;
        const float3 residualRayRad = maxf3((nextTransmit <= minTransmittance ? make_float3(0) : (integratedRadiance - radiance) / nextTransmit),
                                            make_float3(0));

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_gdns = residualTrm * gres
        //
        // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1-galpha) * transmit *
        // residualRayRad
        //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1-gdns*gres) *
        //                  transmit * residualRayRad
        // ===> d_rayRad / d_gdns = gres * transmit * grad - gres * transmit * residualRayRad
        atomicAdd(
            &particleDensityGrad.density,
            gres * (galphaRayHitGrd + galphaRayDnsGrd + transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +
                    transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +
                    transmittance * (grad.z - residualRayRad.z) * radianceGrad.z));

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDns = 1 - prevTrm * (1-galpha) * nextTrm
        //             = 1 - (1-galpha) * prevTrm * nextTrm
        // ===> d_rayDns / d_gres = residualTrm * gdns
        //
        // ---> rayRadiance = accumulatedRayRad + galpha * transmit * grad + (1 - galpha) * transmit *
        // residualRayRad
        //                  = accumulatedRayRad + gdns * gres * transmit * grad + (1 - gdns * gres) *
        //                  transmit * residualRayRad
        // ===> d_rayRad / d_gres = gdns * transmit * grad - gdns * transmit * residualRayRad
        const float gresGrd =
            particleDensity * (galphaRayHitGrd + galphaRayDnsGrd + transmittance * (grad.x - residualRayRad.x) * radianceGrad.x +
                               transmittance * (grad.y - residualRayRad.y) * radianceGrad.y +
                               transmittance * (grad.z - residualRayRad.z) * radianceGrad.z);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gres = exp(-0.0555 * grayDist * grayDist)
        // ===> d_gres / d_grayDist = -0.111 * grayDist * exp(-0.555 * grayDist * grayDist)
        //                          = -0.111 * grayDist * gres
        const float grayDistGrd = particleResponseGrd<PARTICLE_KERNEL_DEGREE>(grayDist, gres, gresGrd);

        float3 grdGrd, groGrd;
        if (SurfelPrimitive) {
            const float3 surfelNm    = make_float3(0, 0, 1);
            const float doSurfelGro  = dot(surfelNm, gro);
            const float dotSurfelGrd = dot(surfelNm, grd); // cannot be null otherwise no hit
            const float ghitT        = -doSurfelGro / dotSurfelGrd;
            const float3 ghitPos     = gro + grd * ghitT;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> grayDist = dot(ghitPos, ghitPos)
            //               = ghitPos.x^2 + ghitPos.y^2 + ghitPos.z^2
            // ===> d_grayDist / d_ghitPos = 2*ghitPos
            const float3 ghitPosGrd = 2 * ghitPos * grayDistGrd;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> ghitPos = gro + grd * ghitT
            //
            // ===> d_ghitPos / d_gro = 1
            // ===> d_ghitPos / d_grd = ghitT
            groGrd = ghitPosGrd;
            grdGrd = ghitT * ghitPosGrd;
            // ===> d_ghitPos / d_ghitT = grd
            const float ghitTGrd = sum(grd * ghitPosGrd);

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> ghitT = -dot(surfelNm, gro) / dot(surfNm, grd)
            //
            // ===> d_ghitT / d_gro = -surfelNm / dot(surfNm, grd)
            // ===> d_ghitT / d_dotSurfelGrd = dot(surfelNm, gro) / dotSurfelGrd^2
            groGrd += (-surfelNm * ghitTGrd) / dotSurfelGrd;
            const float dotSurfelGrdGrd = (doSurfelGro * ghitTGrd) / (dotSurfelGrd * dotSurfelGrd);
            // ===> d_dotSurfelGrd / d_grd = surfelNm
            grdGrd += surfelNm * dotSurfelGrdGrd;
        } else {
            const float3 gcrod = cross(grd, gro);

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> grayDist = dot(gcrod, gcrod)
            //               = gcrod.x^2 + gcrod.y^2 + gcrod.z^2
            // ===> d_grayDist / d_gcrod = 2*gcrod
            const float3 gcrodGrd = 2 * gcrod * grayDistGrd;

            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // ---> gcrod = cross(grd, gro)
            // ---> gcrod.x = grd.y * gro.z - grd.z * gro.y
            // ---> gcrod.y = grd.z * gro.x - grd.x * gro.z
            // ---> gcrod.z = grd.x * gro.y - grd.y * gro.x
            grdGrd = make_float3(gcrodGrd.z * gro.y - gcrodGrd.y * gro.z,
                                 gcrodGrd.x * gro.z - gcrodGrd.z * gro.x,
                                 gcrodGrd.y * gro.x - gcrodGrd.x * gro.y);
            groGrd = make_float3(gcrodGrd.y * grd.z - gcrodGrd.z * grd.y,
                                 gcrodGrd.z * grd.x - gcrodGrd.x * grd.z,
                                 gcrodGrd.x * grd.y - gcrodGrd.y * grd.x);
        }

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gro = (1/gscl)*gposcr
        // ===> d_gro / d_gscl = -gposcr/(gscl*gscl)
        // ===> d_gro / d_gposcr = (1/gscl)
        const float3 gsclGrdGro = make_float3((-gposcr.x / (gscl.x * gscl.x)),
                                              (-gposcr.y / (gscl.y * gscl.y)),
                                              (-gposcr.z / (gscl.z * gscl.z))) *
                                  (groGrd + groRayHitGrd);
        const float3 gposcrGrd = giscl * (groGrd + groRayHitGrd);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gposcr = matmul(gposc, grotMat)
        // ===> d_gposcr / d_gposc = matmul_bw_vec(grotMat)
        // ===> d_gposcr / d_grotmat = matmul_bw_mat(gposc)
        const float3 gposcGrd     = matmul_bw_vec(particleInvRotation, gposcrGrd);
        const float4 grotGrdPoscr = matmul_bw_quat(gposc, gposcrGrd, grot);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> gposc = rayOri - gpos
        // ===> d_gposc / d_gpos = -1
        const float3 rayMoGPosGrd = -gposcGrd;
        atomicAdd(&particleDensityGrad.position.x, rayMoGPosGrd.x);
        atomicAdd(&particleDensityGrad.position.y, rayMoGPosGrd.y);
        atomicAdd(&particleDensityGrad.position.z, rayMoGPosGrd.z);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grd = safe_normalize(grdu)
        // ===> d_grd / d_grdu = safe_normalize_bw(grd)
        const float3 grduGrd = safe_normalize_bw(grdu, grdGrd + grdRayHitGrd);

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> grdu = (1/gscl)*rayDirR
        // ===> d_grdu / d_gscl = -rayDirR/(gscl*gscl)
        // ===> d_grdu / d_rayDirR = (1/gscl)
        atomicAdd(&particleDensityGrad.scale.x, gsclRayHitGrd.x + gsclGrdGro.x + (-rayDirR.x / (gscl.x * gscl.x)) * grduGrd.x);
        atomicAdd(&particleDensityGrad.scale.y, gsclRayHitGrd.y + gsclGrdGro.y + (-rayDirR.y / (gscl.y * gscl.y)) * grduGrd.y);
        atomicAdd(&particleDensityGrad.scale.z, gsclRayHitGrd.z + gsclGrdGro.z + (-rayDirR.z / (gscl.z * gscl.z)) * grduGrd.z);
        const float3 rayDirRGrd = giscl * grduGrd;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // ---> rayDirR = matmul(rayDir, grotMat)
        // ===> d_rayDirR / d_grotmat = matmul_bw_mat(rayDir, grotMat)
        const float4 grotGrdRayDirR = matmul_bw_quat(rayDirection, rayDirRGrd, grot);
        atomicAdd(&particleDensityGrad.quaternion.x, grotGrdPoscr.x + grotGrdRayDirR.x);
        atomicAdd(&particleDensityGrad.quaternion.y, grotGrdPoscr.y + grotGrdRayDirR.y);
        atomicAdd(&particleDensityGrad.quaternion.z, grotGrdPoscr.z + grotGrdRayDirR.z);
        atomicAdd(&particleDensityGrad.quaternion.w, grotGrdPoscr.w + grotGrdRayDirR.w);

        // 计算Loss对射线原点和方向的梯度
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // rayOrigin梯度计算
        // ---> gposc = rayOrigin - particlePosition
        // ===> d_gposc / d_rayOrigin = 1
        // 所以 rayOriginGrad = gposcGrd
        
        // 创建局部变量存储梯度，并输出到指针（如果提供了的话）
        float3 localRayOriginGrad = gposcGrd;

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        // rayDirection梯度计算
        // ---> rayDirR = rayDirection * particleInvRotation
        // ===> d_rayDirR / d_rayDirection = particleInvRotation
        float3 localRayDirectionGrad = matmul_bw_vec(particleInvRotation, rayDirRGrd);

        // 从球谐光照计算中获取方向梯度
        float3 rayDirGradFromSpH = make_float3(0.f, 0.f, 0.f);
        if(weight > 0.0f) {
            rayDirGradFromSpH = grad * weight * radianceGrad;
            localRayDirectionGrad -= rayDirGradFromSpH;
        }

        // 累加梯度，而不是直接赋值
        if (rayOriginGrad != nullptr) {
            rayOriginGrad->x -= localRayOriginGrad.x;
            rayOriginGrad->y -= localRayOriginGrad.y;
            rayOriginGrad->z -= localRayOriginGrad.z;
        }

        if (rayDirectionGrad != nullptr) {
            rayDirectionGrad->x -= localRayDirectionGrad.x;
            rayDirectionGrad->y -= localRayDirectionGrad.y;
            rayDirectionGrad->z -= localRayDirectionGrad.z;
        }

        transmittance = nextTransmit;
    }
}
