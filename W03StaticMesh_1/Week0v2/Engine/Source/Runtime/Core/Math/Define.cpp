#include "Define.h"
#define SIMD 1
#define AVX 0

// 단위 행렬 정의
const FMatrix FMatrix::Identity = { {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
} };

// 행렬 덧셈
FMatrix FMatrix::operator+(const FMatrix& Other) const {
    FMatrix Result;
    for (int32 i = 0; i < 4; i++)
    {
#if SIMD
        Result.Row[i] = _mm_add_ps(Row[i], Other.Row[i]);
#else
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = M[i][j] + Other.M[i][j];
#endif
    }
    return Result;
}

// 행렬 뺄셈
FMatrix FMatrix::operator-(const FMatrix& Other) const {
    FMatrix Result;
    for (int32 i = 0; i < 4; i++)
    {
#if SIMD
        Result.Row[i] = _mm_sub_ps(Row[i], Other.Row[i]);
#else
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = M[i][j] - Other.M[i][j];
#endif
    }
    return Result;
}

// 행렬 곱셈
FMatrix FMatrix::operator*(const FMatrix& Other) const {
    FMatrix Result = {};
    for (int32 i = 0; i < 4; i++)
    {
#if SIMD
        __m128 R = Row[i];
        __m128 R0 = _mm_mul_ps(_mm_shuffle_ps(R, R, 0x00), Other.Row[0]);
        __m128 R1 = _mm_mul_ps(_mm_shuffle_ps(R, R, 0x55), Other.Row[1]);
        __m128 R2 = _mm_mul_ps(_mm_shuffle_ps(R, R, 0xAA), Other.Row[2]);
        __m128 R3 = _mm_mul_ps(_mm_shuffle_ps(R, R, 0xFF), Other.Row[3]);
        Result.Row[i] = _mm_add_ps(_mm_add_ps(R0, R1), _mm_add_ps(R2, R3));
#else
        for (int32 j = 0; j < 4; j++)
            for (int32 k = 0; k < 4; k++)
                Result.M[i][j] += M[i][k] * Other.M[k][j];
#endif
    }
    return Result;

    //FMatrix R;
    //
    //// B를 column 단위로 분해
    //__m128 B0 = Other.Row[0];
    //__m128 B1 = Other.Row[1];
    //__m128 B2 = Other.Row[2];
    //__m128 B3 = Other.Row[3];
    //
    //// A의 Row들과 B의 columns 내적
    //for (int i = 0; i < 4; ++i) {
    //    __m128 r = Row[i];
    //    R.Row[i] = _mm_add_ps(
    //        _mm_add_ps(
    //            _mm_mul_ps(_mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 0, 0, 0)), B0),
    //            _mm_mul_ps(_mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 1, 1, 1)), B1)
    //        ),
    //        _mm_add_ps(
    //            _mm_mul_ps(_mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 2, 2, 2)), B2),
    //            _mm_mul_ps(_mm_shuffle_ps(r, r, _MM_SHUFFLE(3, 3, 3, 3)), B3)
    //        )
    //    );
    //}
    //return R;
}

// 스칼라 곱셈
FMatrix FMatrix::operator*(float Scalar) const {
    FMatrix Result;
    for (int32 i = 0; i < 4; i++)
    {
#if SIMD
        Result.Row[i] = _mm_mul_ps(Row[i], _mm_set1_ps(Scalar));
#else
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = M[i][j] * Scalar;
#endif
    }
    return Result;
}

// 스칼라 나눗셈
FMatrix FMatrix::operator/(float Scalar) const {
    assert(Scalar != 0.f);
    FMatrix Result;
    for (int32 i = 0; i < 4; i++)
    {
#if SIMD
        Result.Row[i] = _mm_div_ps(Row[i], _mm_set1_ps(Scalar));
#else
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = M[i][j] / Scalar;
#endif
    }
    return Result;
}

float* FMatrix::operator[](int Row) {
    return M[Row];
}

const float* FMatrix::operator[](int Row) const
{
    return M[Row];
}

// 전치 행렬
FMatrix FMatrix::Transpose(const FMatrix& Mat) {
    FMatrix Result;
#if SIMD
    // x,x x,y x,z x,w
    // y,x y,y y,z y,w
    // z,x z,y z,z z,w
    // w,x w,y w,z w,w
    // x,x x,y y,x y,y (0[0], 0[1] 1[0] 1[1])
    // x,z x,w y,z y,w (0[2], 0[3] 1[2] 1[3])
    // z,x z,y w,x w,y (2[0], 2[1] 3[0] 3[1])
    // z,z z,w w,z w,w (2[2], 2[3] 3[2] 3[3])
    __m128 TempRow0 = _mm_shuffle_ps((Mat.Row[0]), (Mat.Row[1]), 0x44); // 0x44 = 0b01 00 01 00 (1, 0, 1, 0)
    __m128 TempRow1 = _mm_shuffle_ps((Mat.Row[0]), (Mat.Row[1]), 0xEE); // 0xEE = 0b11 10 11 10 (3, 2, 3, 2)
    __m128 TempRow2 = _mm_shuffle_ps((Mat.Row[2]), (Mat.Row[3]), 0x44); // 0x44 = 0b01 00 01 00 (1, 0, 1, 0)
    __m128 TempRow3 = _mm_shuffle_ps((Mat.Row[2]), (Mat.Row[3]), 0xEE); // 0xEE = 0b11 10 11 10 (3, 2, 3, 2)
    // x,x y,x z,x w,x (0[0], 0[2], 2[0], 2[2])
    // x,y y,y z,y w,y (0[1], 0[3], 2[1], 2[3])
    // x,z y,z z,z w,z (1[0], 1[2], 3[0], 3[2])
    // x,w y,w z,w w,w (1[1], 1[3], 3[1], 3[3])
    Result.Row[0] = _mm_shuffle_ps(TempRow0, TempRow2, 0x88); // 0x88 = 0b10 00 10 00 (2, 0, 2, 0)
    Result.Row[1] = _mm_shuffle_ps(TempRow0, TempRow2, 0xDD); // 0xDD = 0b11 01 11 01 (3, 1, 3, 1)
    Result.Row[2] = _mm_shuffle_ps(TempRow1, TempRow3, 0x88); // 0x88 = 0b10 00 10 00 (2, 0, 2, 0)
    Result.Row[3] = _mm_shuffle_ps(TempRow1, TempRow3, 0xDD); // 0xDD = 0b11 01 11 01 (3, 1, 3, 1)
#elif AVX
    __m256 ymm2 = _mm256_loadu_ps(reinterpret_cast<const float*>(Mat.M));          // ymm2 = { x0, y0, z0, w0, x1, y1, z1, w1 }
    __m256 ymm3 = _mm256_loadu_ps(reinterpret_cast<const float*>(Mat.M) + 8);      // ymm3 = { x2, y2, z2, w2, x3, y3, z3, w3 }
    __m256 ymm0 = _mm256_permute2f128_ps(ymm2, ymm3, 0x20);                      // ymm0 = { x0, y0, z0, w0, x2, y2, z2, w2 }
    __m256 ymm1 = _mm256_permute2f128_ps(ymm2, ymm3, 0x31);                      // ymm1 = { x1, y1, z1, w1, x3, y3, z3, w3 }

    // shuffle to each ymm contain the x, z or y, w elements of every row
    ymm2 = _mm256_shuffle_ps(ymm0, ymm1, 0x88);                                  // ymm2 = { x0, z0, x1, z1, x2, z2, x3, z3 }
    ymm3 = _mm256_shuffle_ps(ymm0, ymm1, 0xDD);                                  // ymm3 = { y0, w0, y1, w1, y2, w2, y3, w3 }

    __m256 ymm4 = _mm256_insertf128_ps(ymm2, _mm256_castps256_ps128(ymm3), 1);   // ymm4 = { x0, z0, x1, z1, y0, w0, y1, w1 }
    __m256 ymm5 = _mm256_permute2f128_ps(ymm2, ymm3, 0x31);                      // ymm5 = { x2, z2, x3, z3, y2, w2, y3, w3 }

    __m256 ymm6 = _mm256_shuffle_ps(ymm4, ymm5, 0x88);                           // ymm6 = { x0, x1, x2, x3, y0, y1, y2, y3 }
    __m256 ymm7 = _mm256_shuffle_ps(ymm4, ymm5, 0xDD);                           // ymm7 = { z0, z1, z2, z3, w0, w1, w2, w3 }

    _mm256_storeu_ps(reinterpret_cast<float*>(Result.M), ymm6);
    _mm256_storeu_ps(reinterpret_cast<float*>(Result.M) + 8, ymm7);
#else
    for (int32 i = 0; i < 4; i++)
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = Mat.M[j][i];
#endif
    return Result;
}

// 행렬식 계산 (라플라스 전개, 4x4 행렬)
float FMatrix::Determinant(const FMatrix& Mat)
{
#if SIMD
    /**
         * M = [ A B ] (A, B, C, D는 2x2 행렬)
         *     [ C D ]
         * Det(M) = Det(A - B * inv(D) * C) * Det(D)
         */

         // 4x4 행렬을 2x2 블록으로 분할
    __m128 A, B, C, D;
    Split4x4To2x2(Mat, A, B, C, D);

    // 행렬식 계산: det(M) = det(A - B * inv(D) * C) * det(D)
    float DetD = Det2x2(D);

    // D가 특이 행렬이면 다른 방법으로 계산
    if (fabs(DetD) < 1e-6f) {
        // 여인수 전개 방식으로 대체 가능
        // 여기서는 간략화를 위해 0을 반환
        return 0.0f;
    }

    //__m128 InvD = Inv2x2(D);                // D의 역행렬
    __m128 InvD = InverseDet2x2(D, DetD);
    __m128 Temp1 = Mul2x2(InvD, C);    // inv(D) * C
    __m128 Temp2 = Mul2x2(B, Temp1);   // B * inv(D) * C
    __m128 Schur = Sub2x2(A, Temp2);     // A - B * inv(D) * C

    float DetSchur = Det2x2(Schur);

    return DetSchur * DetD;
#else
    float det = 0.0f;
    for (int32 i = 0; i < 4; i++) {
        float subMat[3][3];
        for (int32 j = 1; j < 4; j++) {
            int32 colIndex = 0;
            for (int32 k = 0; k < 4; k++) {
                if (k == i) continue;
                subMat[j - 1][colIndex] = Mat.M[j][k];
                colIndex++;
            }
        }
        float minorDet =
            subMat[0][0] * (subMat[1][1] * subMat[2][2] - subMat[1][2] * subMat[2][1]) -
            subMat[0][1] * (subMat[1][0] * subMat[2][2] - subMat[1][2] * subMat[2][0]) +
            subMat[0][2] * (subMat[1][0] * subMat[2][1] - subMat[1][1] * subMat[2][0]);
        det += (i % 2 == 0 ? 1 : -1) * Mat.M[0][i] * minorDet;
    }
    return det;
#endif
}

FMatrix FMatrix::Inverse(const FMatrix& Mat) {
#if SIMD
    // Cramer's Rule
    // https://github.com/microsoft/DirectXMath/blob/main/Inc/DirectXMathMatrix.inl

    //float det = Determinant(Mat);
    //if (fabs(det) < 1e-6) {
    //    return Identity;
    //}

    // Transpose matrix
    FMatrix MatT = Transpose(Mat);

    __m128 V00 = XM_PERMUTE_PS(MatT.Row[2], _MM_SHUFFLE(1, 1, 0, 0));
    __m128 V10 = XM_PERMUTE_PS(MatT.Row[3], _MM_SHUFFLE(3, 2, 3, 2));
    __m128 V01 = XM_PERMUTE_PS(MatT.Row[0], _MM_SHUFFLE(1, 1, 0, 0));
    __m128 V11 = XM_PERMUTE_PS(MatT.Row[1], _MM_SHUFFLE(3, 2, 3, 2));
    __m128 V02 = _mm_shuffle_ps(MatT.Row[2], MatT.Row[0], _MM_SHUFFLE(2, 0, 2, 0));
    __m128 V12 = _mm_shuffle_ps(MatT.Row[3], MatT.Row[1], _MM_SHUFFLE(3, 1, 3, 1));

    __m128 D0 = _mm_mul_ps(V00, V10);
    __m128 D1 = _mm_mul_ps(V01, V11);
    __m128 D2 = _mm_mul_ps(V02, V12);

    V00 = XM_PERMUTE_PS(MatT.Row[2], _MM_SHUFFLE(3, 2, 3, 2));
    V10 = XM_PERMUTE_PS(MatT.Row[3], _MM_SHUFFLE(1, 1, 0, 0));
    V01 = XM_PERMUTE_PS(MatT.Row[0], _MM_SHUFFLE(3, 2, 3, 2));
    V11 = XM_PERMUTE_PS(MatT.Row[1], _MM_SHUFFLE(1, 1, 0, 0));
    V02 = _mm_shuffle_ps(MatT.Row[2], MatT.Row[0], _MM_SHUFFLE(3, 1, 3, 1));
    V12 = _mm_shuffle_ps(MatT.Row[3], MatT.Row[1], _MM_SHUFFLE(2, 0, 2, 0));

    D0 = XM_FNMADD_PS(V00, V10, D0);
    D1 = XM_FNMADD_PS(V01, V11, D1);
    D2 = XM_FNMADD_PS(V02, V12, D2);

    // V11 = D0Y,D0W,D2Y,D2Y
    V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 1, 3, 1));
    V00 = XM_PERMUTE_PS(MatT.Row[1], _MM_SHUFFLE(1, 0, 2, 1));
    V10 = _mm_shuffle_ps(V11, D0, _MM_SHUFFLE(0, 3, 0, 2));
    V01 = XM_PERMUTE_PS(MatT.Row[0], _MM_SHUFFLE(0, 1, 0, 2));
    V11 = _mm_shuffle_ps(V11, D0, _MM_SHUFFLE(2, 1, 2, 1));

    // V13 = D1Y,D1W,D2W,D2W
    __m128 V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 3, 3, 1));
    V02 = XM_PERMUTE_PS(MatT.Row[3], _MM_SHUFFLE(1, 0, 2, 1));
    V12 = _mm_shuffle_ps(V13, D1, _MM_SHUFFLE(0, 3, 0, 2));
    __m128 V03 = XM_PERMUTE_PS(MatT.Row[2], _MM_SHUFFLE(0, 1, 0, 2));
    V13 = _mm_shuffle_ps(V13, D1, _MM_SHUFFLE(2, 1, 2, 1));

    __m128 C0 = _mm_mul_ps(V00, V10);
    __m128 C2 = _mm_mul_ps(V01, V11);
    __m128 C4 = _mm_mul_ps(V02, V12);
    __m128 C6 = _mm_mul_ps(V03, V13);

    // V11 = D0X,D0Y,D2X,D2X
    V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(0, 0, 1, 0));
    V00 = XM_PERMUTE_PS(MatT.Row[1], _MM_SHUFFLE(2, 1, 3, 2));
    V10 = _mm_shuffle_ps(D0, V11, _MM_SHUFFLE(2, 1, 0, 3));
    V01 = XM_PERMUTE_PS(MatT.Row[0], _MM_SHUFFLE(1, 3, 2, 3));
    V11 = _mm_shuffle_ps(D0, V11, _MM_SHUFFLE(0, 2, 1, 2));
    // V13 = D1X,D1Y,D2Z,D2Z
    V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(2, 2, 1, 0));
    V02 = XM_PERMUTE_PS(MatT.Row[3], _MM_SHUFFLE(2, 1, 3, 2));
    V12 = _mm_shuffle_ps(D1, V13, _MM_SHUFFLE(2, 1, 0, 3));
    V03 = XM_PERMUTE_PS(MatT.Row[2], _MM_SHUFFLE(1, 3, 2, 3));
    V13 = _mm_shuffle_ps(D1, V13, _MM_SHUFFLE(0, 2, 1, 2));

    C0 = XM_FNMADD_PS(V00, V10, C0);
    C2 = XM_FNMADD_PS(V01, V11, C2);
    C4 = XM_FNMADD_PS(V02, V12, C4);
    C6 = XM_FNMADD_PS(V03, V13, C6);

    V00 = XM_PERMUTE_PS(MatT.Row[1], _MM_SHUFFLE(0, 3, 0, 3));
    // V10 = D0Z,D0Z,D2X,D2Y
    V10 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 0, 2, 2));
    V10 = XM_PERMUTE_PS(V10, _MM_SHUFFLE(0, 2, 3, 0));
    V01 = XM_PERMUTE_PS(MatT.Row[0], _MM_SHUFFLE(2, 0, 3, 1));
    // V11 = D0X,D0W,D2X,D2Y
    V11 = _mm_shuffle_ps(D0, D2, _MM_SHUFFLE(1, 0, 3, 0));
    V11 = XM_PERMUTE_PS(V11, _MM_SHUFFLE(2, 1, 0, 3));
    V02 = XM_PERMUTE_PS(MatT.Row[3], _MM_SHUFFLE(0, 3, 0, 3));
    // V12 = D1Z,D1Z,D2Z,D2W
    V12 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 2, 2, 2));
    V12 = XM_PERMUTE_PS(V12, _MM_SHUFFLE(0, 2, 3, 0));
    V03 = XM_PERMUTE_PS(MatT.Row[2], _MM_SHUFFLE(2, 0, 3, 1));
    // V13 = D1X,D1W,D2Z,D2W
    V13 = _mm_shuffle_ps(D1, D2, _MM_SHUFFLE(3, 2, 3, 0));
    V13 = XM_PERMUTE_PS(V13, _MM_SHUFFLE(2, 1, 0, 3));

    V00 = _mm_mul_ps(V00, V10);
    V01 = _mm_mul_ps(V01, V11);
    V02 = _mm_mul_ps(V02, V12);
    V03 = _mm_mul_ps(V03, V13);
    __m128 C1 = _mm_sub_ps(C0, V00);
    C0 = _mm_add_ps(C0, V00);
    __m128 C3 = _mm_add_ps(C2, V01);
    C2 = _mm_sub_ps(C2, V01);
    __m128 C5 = _mm_sub_ps(C4, V02);
    C4 = _mm_add_ps(C4, V02);
    __m128 C7 = _mm_add_ps(C6, V03);
    C6 = _mm_sub_ps(C6, V03);

    C0 = _mm_shuffle_ps(C0, C1, _MM_SHUFFLE(3, 1, 2, 0));
    C2 = _mm_shuffle_ps(C2, C3, _MM_SHUFFLE(3, 1, 2, 0));
    C4 = _mm_shuffle_ps(C4, C5, _MM_SHUFFLE(3, 1, 2, 0));
    C6 = _mm_shuffle_ps(C6, C7, _MM_SHUFFLE(3, 1, 2, 0));
    C0 = XM_PERMUTE_PS(C0, _MM_SHUFFLE(3, 1, 2, 0));
    C2 = XM_PERMUTE_PS(C2, _MM_SHUFFLE(3, 1, 2, 0));
    C4 = XM_PERMUTE_PS(C4, _MM_SHUFFLE(3, 1, 2, 0));
    C6 = XM_PERMUTE_PS(C6, _MM_SHUFFLE(3, 1, 2, 0));

    // dot product to calculate determinant
    __m128 vTemp2 = C0;
    __m128 vTemp = _mm_mul_ps(MatT.Row[0], vTemp2);
    vTemp2 = _mm_shuffle_ps(vTemp2, vTemp, _MM_SHUFFLE(1, 0, 0, 0));
    vTemp2 = _mm_add_ps(vTemp2, vTemp);
    vTemp = _mm_shuffle_ps(vTemp, vTemp2, _MM_SHUFFLE(0, 3, 0, 0));
    vTemp = _mm_add_ps(vTemp, vTemp2);
    vTemp = XM_PERMUTE_PS(vTemp, _MM_SHUFFLE(2, 2, 2, 2));

    vTemp = _mm_div_ps(_mm_set_ps1(1.f), vTemp);
    FMatrix mResult;
    mResult.Row[0] = _mm_mul_ps(C0, vTemp);
    mResult.Row[1] = _mm_mul_ps(C2, vTemp);
    mResult.Row[2] = _mm_mul_ps(C4, vTemp);
    mResult.Row[3] = _mm_mul_ps(C6, vTemp);
    return mResult;
#else
    // 역행렬 (가우스-조던 소거법)
    float det = Determinant(Mat);
    if (fabs(det) < 1e-6) {
        return Identity;
    }

    FMatrix Inv;
    float invDet = 1.0f / det;

    // 여인수 행렬 계산 후 전치하여 역행렬 계산
    for (int32 i = 0; i < 4; i++) {
        for (int32 j = 0; j < 4; j++) {
            float subMat[3][3];
            int32 subRow = 0;
            for (int32 r = 0; r < 4; r++) {
                if (r == i) continue;
                int32 subCol = 0;
                for (int32 c = 0; c < 4; c++) {
                    if (c == j) continue;
                    subMat[subRow][subCol] = Mat.M[r][c];
                    subCol++;
                }
                subRow++;
            }
            float minorDet =
                subMat[0][0] * (subMat[1][1] * subMat[2][2] - subMat[1][2] * subMat[2][1]) -
                subMat[0][1] * (subMat[1][0] * subMat[2][2] - subMat[1][2] * subMat[2][0]) +
                subMat[0][2] * (subMat[1][0] * subMat[2][1] - subMat[1][1] * subMat[2][0]);

            Inv.M[j][i] = ((i + j) % 2 == 0 ? 1 : -1) * minorDet * invDet;
        }
    }
    return Inv;
#endif
}

FMatrix FMatrix::CreateRotation(float roll, float pitch, float yaw)
{
    float radRoll = roll * (3.14159265359f / 180.0f);
    float radPitch = pitch * (3.14159265359f / 180.0f);
    float radYaw = yaw * (3.14159265359f / 180.0f);

    float cosRoll = cos(radRoll), sinRoll = sin(radRoll);
    float cosPitch = cos(radPitch), sinPitch = sin(radPitch);
    float cosYaw = cos(radYaw), sinYaw = sin(radYaw);

    // Z축 (Yaw) 회전
    FMatrix rotationZ = { {
        { cosYaw, sinYaw, 0, 0 },
        { -sinYaw, cosYaw, 0, 0 },
        { 0, 0, 1, 0 },
        { 0, 0, 0, 1 }
    } };

    // Y축 (Pitch) 회전
    FMatrix rotationY = { {
        { cosPitch, 0, -sinPitch, 0 },
        { 0, 1, 0, 0 },
        { sinPitch, 0, cosPitch, 0 },
        { 0, 0, 0, 1 }
    } };

    // X축 (Roll) 회전
    FMatrix rotationX = { {
        { 1, 0, 0, 0 },
        { 0, cosRoll, sinRoll, 0 },
        { 0, -sinRoll, cosRoll, 0 },
        { 0, 0, 0, 1 }
    } };

    // DirectX 표준 순서: Z(Yaw) → Y(Pitch) → X(Roll)  
    return rotationX * rotationY * rotationZ;  // 이렇게 하면  오른쪽 부터 적용됨
}

// 스케일 행렬 생성
FMatrix FMatrix::CreateScale(float scaleX, float scaleY, float scaleZ)
{
    return { {
        { scaleX, 0, 0, 0 },
        { 0, scaleY, 0, 0 },
        { 0, 0, scaleZ, 0 },
        { 0, 0, 0, 1 }
    } };
}

FMatrix FMatrix::CreateTranslationMatrix(const FVector& position)
{
    FMatrix translationMatrix = FMatrix::Identity;
#if SIMD
    // Identity이므로 M[3][3] = 1
    translationMatrix.Row[3] = _mm_set_ps(1.0f, position.z, position.y, position.x);
#else
    translationMatrix.M[3][0] = position.x;
    translationMatrix.M[3][1] = position.y;
    translationMatrix.M[3][2] = position.z;
#endif
    return translationMatrix;
}

#if SIMD
float FMatrix::Det2x2(__m128 Mat)
{
    // Mat = [a, b, c, d]
    // 대각 원소 곱: AD
    float AD = _mm_cvtss_f32(_mm_mul_ss(_mm_shuffle_ps(Mat, Mat, _MM_SHUFFLE(3, 3, 3, 0)),
        _mm_shuffle_ps(Mat, Mat, _MM_SHUFFLE(3, 3, 3, 3))));
    /**
     * 뒤의 3 요소는 쓰이지 않음... -> 비효율
     * a | d d d
     * d | d d d
     * a * d
     */
     // 역대각 원소 곱: BC
    float BC = _mm_cvtss_f32(_mm_mul_ss(_mm_shuffle_ps(Mat, Mat, _MM_SHUFFLE(3, 3, 3, 1)),
        _mm_shuffle_ps(Mat, Mat, _MM_SHUFFLE(3, 3, 3, 2))));
    return AD - BC;
}

__m128 FMatrix::Inverse2x2(__m128 Mat)
{
    // Mat = [a, b, c, d]
    // 행렬식 계산
    float Det = Det2x2(Mat);

    if (fabs(Det) < 1e-6f) {
        // 행렬식이 0에 가까우면 특이 행렬로 간주
        return _mm_setzero_ps();
    }

    return InverseDet2x2(Mat, Det);
}

__m128 FMatrix::InverseDet2x2(__m128 Mat, float Det)
{
    // 역행렬 = 1/Det * [d, -b, -c, a]
    float InvDet = 1.0f / Det;
    __m128 InvDetVec = _mm_set1_ps(InvDet);

    // [d, -b, -c, a] 구성
    __m128 Adj = _mm_shuffle_ps(Mat, Mat, _MM_SHUFFLE(0, 1, 2, 3));     // [d, c, b, a]     ???? not [d, b, c, a]
    //__m128 Adj = _mm_shuffle_ps(Mat, Mat, _MM_SHUFFLE(0, 2, 1, 3));             // [d, b, c, a]
    __m128 Sign = _mm_set_ps(1.0f, -1.0f, -1.0f, 1.0f);                 // [1, -1, -1, 1]

    return _mm_mul_ps(_mm_mul_ps(Adj, Sign), InvDetVec);
}

__m128 FMatrix::Mul2x2(__m128 LMat, __m128 RMat)
{
    __m128 Row0 = _mm_add_ps(
        _mm_mul_ps(_mm_shuffle_ps(LMat, LMat, _MM_SHUFFLE(0, 0, 0, 0)), _mm_shuffle_ps(RMat, RMat, _MM_SHUFFLE(1, 0, 1, 0))),
        _mm_mul_ps(_mm_shuffle_ps(LMat, LMat, _MM_SHUFFLE(1, 1, 1, 1)), _mm_shuffle_ps(RMat, RMat, _MM_SHUFFLE(3, 2, 3, 2)))
    );
    __m128 Row1 = _mm_add_ps(
        _mm_mul_ps(_mm_shuffle_ps(LMat, LMat, _MM_SHUFFLE(2, 2, 2, 2)), _mm_shuffle_ps(RMat, RMat, _MM_SHUFFLE(1, 0, 1, 0))),
        _mm_mul_ps(_mm_shuffle_ps(LMat, LMat, _MM_SHUFFLE(3, 3, 3, 3)), _mm_shuffle_ps(RMat, RMat, _MM_SHUFFLE(3, 2, 3, 2)))
    );

    return _mm_shuffle_ps(
        _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(0, 0, 0, 0)),
        _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(1, 1, 1, 1)),
        _MM_SHUFFLE(2, 0, 2, 0)
    );
}

__m128 FMatrix::Sub2x2(__m128 LMat, __m128 RMat)
{
    return _mm_sub_ps(LMat, RMat);
}

void FMatrix::Split4x4To2x2(const FMatrix& Mat, __m128& A, __m128& B, __m128& C, __m128& D)
{
    // 각 2x2 블록 구성
    // A: 왼쪽 상단 2x2
    // B: 오른쪽 상단 2x2
    // C: 왼쪽 하단 2x2
    // D: 오른쪽 하단 2x2

    // 상단 두 행 처리
    __m128 Row0 = Mat.Row[0]; // [m00, m01, m02, m03]
    __m128 Row1 = Mat.Row[1]; // [m10, m11, m12, m13]

    // 하단 두 행 처리
    __m128 Row2 = Mat.Row[2]; // [m20, m21, m22, m23]
    __m128 Row3 = Mat.Row[3]; // [m30, m31, m32, m33]

    // 4x4 행렬을 2x2 블록 4개로 분할
    A = _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(1, 0, 1, 0)); // [m00, m01, m10, m11]
    B = _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(3, 2, 3, 2)); // [m02, m03, m12, m13]
    C = _mm_shuffle_ps(Row2, Row3, _MM_SHUFFLE(1, 0, 1, 0)); // [m20, m21, m30, m31]
    D = _mm_shuffle_ps(Row2, Row3, _MM_SHUFFLE(3, 2, 3, 2)); // [m22, m23, m32, m33]
}
#endif

FVector FMatrix::TransformVector(const FVector& Vec, const FMatrix& Mat)
{
    FVector Result;
#if SIMD
    __m128 VX = _mm_set_ps1(Vec.x);
    __m128 VY = _mm_set_ps1(Vec.y);
    __m128 VZ = _mm_set_ps1(Vec.z);

    __m128 R = _mm_mul_ps(VX, Mat.Row[0]);
    R = _mm_add_ps(R, _mm_mul_ps(VY, Mat.Row[1]));
    R = _mm_add_ps(R, _mm_mul_ps(VZ, Mat.Row[2]));

    _mm_storeu_ps(reinterpret_cast<float*>(&Result), R);
#else
    // 4x4 행렬을 사용하여 벡터 변환 (W = 0으로 가정, 방향 벡터)
    result.x = Vec.x * Mat.M[0][0] + Vec.y * Mat.M[1][0] + Vec.z * Mat.M[2][0] + 0.0f * Mat.M[3][0];
    result.y = Vec.x * Mat.M[0][1] + Vec.y * Mat.M[1][1] + Vec.z * Mat.M[2][1] + 0.0f * Mat.M[3][1];
    result.z = Vec.x * Mat.M[0][2] + Vec.y * Mat.M[1][2] + Vec.z * Mat.M[2][2] + 0.0f * Mat.M[3][2];
#endif

    return Result;
}

// FVector4를 변환하는 함수
FVector4 FMatrix::TransformVector(const FVector4& Vec, const FMatrix& Mat)
{
    FVector4 Result;
#if SIMD
    __m128 VX = _mm_set_ps1(Vec.x);
    __m128 VY = _mm_set_ps1(Vec.y);
    __m128 VZ = _mm_set_ps1(Vec.z);
    __m128 VW = _mm_set_ps1(Vec.a);

    __m128 R = _mm_mul_ps(VX, Mat.Row[0]);
    R = _mm_add_ps(R, _mm_mul_ps(VY, Mat.Row[1]));
    R = _mm_add_ps(R, _mm_mul_ps(VZ, Mat.Row[2]));
    R = _mm_add_ps(R, _mm_mul_ps(VW, Mat.Row[3]));

    _mm_storeu_ps(reinterpret_cast<float*>(&Result), R);
#else
    result.x = Vec.x * Mat.M[0][0] + Vec.y * Mat.M[1][0] + Vec.z * Mat.M[2][0] + Vec.a * Mat.M[3][0];
    result.y = Vec.x * Mat.M[0][1] + Vec.y * Mat.M[1][1] + Vec.z * Mat.M[2][1] + Vec.a * Mat.M[3][1];
    result.z = Vec.x * Mat.M[0][2] + Vec.y * Mat.M[1][2] + Vec.z * Mat.M[2][2] + Vec.a * Mat.M[3][2];
    result.a = Vec.x * Mat.M[0][3] + Vec.y * Mat.M[1][3] + Vec.z * Mat.M[2][3] + Vec.a * Mat.M[3][3];
#endif
    return Result;
}

FVector FMatrix::TransformPosition(const FVector& Vec) const
{
#if SIMD
    FVector Result;
    __m128 VX = _mm_set_ps1(Vec.x);
    __m128 VY = _mm_set_ps1(Vec.y);
    __m128 VZ = _mm_set_ps1(Vec.z);
    __m128 VW = _mm_set_ps1(1.f);
    __m128 R = _mm_mul_ps(VX, Row[0]);
    R = _mm_add_ps(R, _mm_mul_ps(VY, Row[1]));
    R = _mm_add_ps(R, _mm_mul_ps(VZ, Row[2]));
    R = _mm_add_ps(R, _mm_mul_ps(VW, Row[3]));
    // w 값으로 나누기
    __m128 W = _mm_shuffle_ps(R, R, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 Mask = _mm_cmpneq_ps(W, _mm_setzero_ps());   // w가 0이 아닌 경우
    R = _mm_blendv_ps(R, _mm_div_ps(R, W), Mask);

    _mm_storeu_ps(reinterpret_cast<float*>(&Result), R);
    return Result;
#else
	float x = M[0][0] * vector.x + M[1][0] * vector.y + M[2][0] * vector.z + M[3][0];
	float y = M[0][1] * vector.x + M[1][1] * vector.y + M[2][1] * vector.z + M[3][1];
	float z = M[0][2] * vector.x + M[1][2] * vector.y + M[2][2] * vector.z + M[3][2];
	float w = M[0][3] * vector.x + M[1][3] * vector.y + M[2][3] * vector.z + M[3][3];
	return w != 0.0f ? FVector{ x / w, y / w, z / w } : FVector{ x, y, z };
#endif
}
