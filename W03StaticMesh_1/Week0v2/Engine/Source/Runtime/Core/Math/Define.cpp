#include "Define.h"
//#define SIMD

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
#ifdef SIMD
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
#ifdef SIMD
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
#ifdef SIMD
        Result.Row[i] = MulVecMat(Row[i], Other);
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
    //// A의 row들과 B의 columns 내적
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

#ifdef SIMD
// 벡터 곱셈
__m128 FMatrix::MulVecMat(const __m128& Vector, const FMatrix& Matrix) const
{
    __m128 VX = _mm_shuffle_ps(Vector, Vector, 0x00);   // 0x00 = 0b00 00 00 00 (X)
    __m128 VY = _mm_shuffle_ps(Vector, Vector, 0x55);   // 0x55 = 0b01 01 01 01 (Y)
    __m128 VZ = _mm_shuffle_ps(Vector, Vector, 0xAA);   // 0xAA = 0b10 10 10 10 (Z)
    __m128 VW = _mm_shuffle_ps(Vector, Vector, 0xFF);   // 0xFF = 0b11 11 11 11 (W)

    __m128 R =             _mm_mul_ps(VX, Matrix.Row[0]);
    R = _mm_add_ps(R, _mm_mul_ps(VY, Matrix.Row[1]));
    R = _mm_add_ps(R, _mm_mul_ps(VZ, Matrix.Row[2]));
    R = _mm_add_ps(R, _mm_mul_ps(VW, Matrix.Row[3]));

    return R;
}
#endif

// 스칼라 곱셈
FMatrix FMatrix::operator*(float Scalar) const {
    FMatrix Result;
    for (int32 i = 0; i < 4; i++)
    {
#ifdef SIMD
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
#ifdef SIMD
        Result.Row[i] = _mm_div_ps(Row[i], _mm_set1_ps(Scalar));
#else
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = M[i][j] / Scalar;
#endif
    }
    return Result;
}

float* FMatrix::operator[](int row) {
    return M[row];
}

const float* FMatrix::operator[](int row) const
{
    return M[row];
}

// 전치 행렬
FMatrix FMatrix::Transpose(const FMatrix& Mat) {
    FMatrix Result;
#ifdef SIMD
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
#ifdef SIMD
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

// 역행렬 (가우스-조던 소거법)
FMatrix FMatrix::Inverse(const FMatrix& Mat) {
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
    translationMatrix.M[3][0] = position.x;
    translationMatrix.M[3][1] = position.y;
    translationMatrix.M[3][2] = position.z;
    return translationMatrix;
}

#ifdef SIMD
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

FVector FMatrix::TransformVector(const FVector& v, const FMatrix& m)
{
    FVector result;

    // 4x4 행렬을 사용하여 벡터 변환 (W = 0으로 가정, 방향 벡터)
    result.x = v.x * m.M[0][0] + v.y * m.M[1][0] + v.z * m.M[2][0] + 0.0f * m.M[3][0];
    result.y = v.x * m.M[0][1] + v.y * m.M[1][1] + v.z * m.M[2][1] + 0.0f * m.M[3][1];
    result.z = v.x * m.M[0][2] + v.y * m.M[1][2] + v.z * m.M[2][2] + 0.0f * m.M[3][2];


    return result;
}

// FVector4를 변환하는 함수
FVector4 FMatrix::TransformVector(const FVector4& v, const FMatrix& m)
{
    FVector4 result;
    result.x = v.x * m.M[0][0] + v.y * m.M[1][0] + v.z * m.M[2][0] + v.a * m.M[3][0];
    result.y = v.x * m.M[0][1] + v.y * m.M[1][1] + v.z * m.M[2][1] + v.a * m.M[3][1];
    result.z = v.x * m.M[0][2] + v.y * m.M[1][2] + v.z * m.M[2][2] + v.a * m.M[3][2];
    result.a = v.x * m.M[0][3] + v.y * m.M[1][3] + v.z * m.M[2][3] + v.a * m.M[3][3];
    return result;
}

FVector FMatrix::TransformPosition(const FVector& vector) const
{
#ifdef SIMD
    FMatrix T = FMatrix::Transpose(*this);

    __m128 vec = _mm_set_ps(vector.x, vector.y, vector.z, 1.f);
    float* xx = _mm_mul_ps(vec, T.Row[0]).m128_f32;
    float* yy = _mm_mul_ps(vec, T.Row[1]).m128_f32;
    float* zz = _mm_mul_ps(vec, T.Row[2]).m128_f32;
    float* ww = _mm_mul_ps(vec, T.Row[3]).m128_f32;
    float w = ww[0] + ww[1] + ww[2] + ww[3];
    if (w != 0.f) {
        return FVector(
            (xx[0] + xx[1] + xx[2] + xx[3]) / w,
            (yy[0] + yy[1] + yy[2] + yy[3]) / w,
            (zz[0] + zz[1] + zz[2] + zz[3]) / w
        );
    }
    else {
        return FVector(
            xx[0] + xx[1] + xx[2] + xx[3],
            yy[0] + yy[1] + yy[2] + yy[3],
            zz[0] + zz[1] + zz[2] + zz[3]
        );
    }
#else
	float x = M[0][0] * vector.x + M[1][0] * vector.y + M[2][0] * vector.z + M[3][0];
	float y = M[0][1] * vector.x + M[1][1] * vector.y + M[2][1] * vector.z + M[3][1];
	float z = M[0][2] * vector.x + M[1][2] * vector.y + M[2][2] * vector.z + M[3][2];
	float w = M[0][3] * vector.x + M[1][3] * vector.y + M[2][3] * vector.z + M[3][3];
	return w != 0.0f ? FVector{ x / w, y / w, z / w } : FVector{ x, y, z };
#endif
}
