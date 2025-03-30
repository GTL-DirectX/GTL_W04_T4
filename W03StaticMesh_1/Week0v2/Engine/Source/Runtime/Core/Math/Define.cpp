#include "Define.h"
#define AVX 0
#define SSE 1

// 단위 행렬 정의
const FMatrix FMatrix::Identity = { {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
} };

// 행렬 덧셈
FMatrix FMatrix::operator+(const FMatrix& Other) const
{
    FMatrix Result;
#if AVX
    Result.Row256[0] = _mm256_add_ps(Row256[0], Other.Row256[0]);
    Result.Row256[1] = _mm256_add_ps(Row256[1], Other.Row256[1]);
#elif SSE
    Result.Row[0] = _mm_add_ps(Row[0], Other.Row[0]);
    Result.Row[1] = _mm_add_ps(Row[1], Other.Row[1]);
    Result.Row[2] = _mm_add_ps(Row[2], Other.Row[2]);
    Result.Row[3] = _mm_add_ps(Row[3], Other.Row[3]);
#else
    for (int32 i = 0; i < 4; i++)
    {
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = M[i][j] + Other.M[i][j];
    }
#endif
    return Result;
}

// 행렬 뺄셈
FMatrix FMatrix::operator-(const FMatrix& Other) const
{
        FMatrix Result;
#if AVX
        Result.Row256[0] = _mm256_sub_ps(Row256[0], Other.Row256[0]);
        Result.Row256[1] = _mm256_sub_ps(Row256[1], Other.Row256[1]);
#elif SSE
        Result.Row[0] = _mm_sub_ps(Row[0], Other.Row[0]);
        Result.Row[1] = _mm_sub_ps(Row[1], Other.Row[1]);
        Result.Row[2] = _mm_sub_ps(Row[2], Other.Row[2]);
        Result.Row[3] = _mm_sub_ps(Row[3], Other.Row[3]);
#else
        for (int32 i = 0; i < 4; i++)
        {
            for (int32 j = 0; j < 4; j++)
                Result.M[i][j] = M[i][j] - Other.M[i][j];
        }
#endif
        return Result;
    }

// 행렬 곱셈
FMatrix FMatrix::operator*(const FMatrix& Other) const
{
    FMatrix Result = {};
#if AVX
    const __m256 T0 = Row256[0];                                            // t0 = a00, a01, a02, a03, a10, a11, a12, a13
    const __m256 T1 = Row256[1];                                            // t1 = a20, a21, a22, a23, a30, a31, a32, a33
    const __m256 U0 = Other.Row256[0];                                      // u0 = b00, b01, b02, b03, b10, b11, b12, b13
    const __m256 U1 = Other.Row256[1];                                      // u1 = b20, b21, b22, b23, b30, b31, b32, b33

    __m256 A0 = _mm256_shuffle_ps(T0, T0, _MM_SHUFFLE(0, 0, 0, 0));         // a0 = a00, a00, a00, a00, a10, a10, a10, a10
    __m256 A1 = _mm256_shuffle_ps(T1, T1, _MM_SHUFFLE(0, 0, 0, 0));         // a1 = a20, a20, a20, a20, a30, a30, a30, a30
    __m256 B0 = _mm256_permute2f128_ps(U0, U0, 0x00);                       // b0 = b00, b01, b02, b03, b00, b01, b02, b03  
    __m256 C0 = _mm256_mul_ps(A0, B0);                                      // c0 = a00*b00  a00*b01  a00*b02  a00*b03  a10*b00  a10*b01  a10*b02  a10*b03
    __m256 C1 = _mm256_mul_ps(A1, B0);                                      // c1 = a20*b00  a20*b01  a20*b02  a20*b03  a30*b00  a30*b01  a30*b02  a30*b03

    A0 = _mm256_shuffle_ps(T0, T0, _MM_SHUFFLE(1, 1, 1, 1));                // a0 = a01, a01, a01, a01, a11, a11, a11, a11
    A1 = _mm256_shuffle_ps(T1, T1, _MM_SHUFFLE(1, 1, 1, 1));                // a1 = a21, a21, a21, a21, a31, a31, a31, a31
    B0 = _mm256_permute2f128_ps(U0, U0, 0x11);                              // b0 = b10, b11, b12, b13, b10, b11, b12, b13
    const __m256 C2 = _mm256_mul_ps(A0, B0);                                // c2 = a01*b10  a01*b11  a01*b12  a01*b13  a11*b10  a11*b11  a11*b12  a11*b13
    const __m256 C3 = _mm256_mul_ps(A1, B0);                                // c3 = a21*b10  a21*b11  a21*b12  a21*b13  a31*b10  a31*b11  a31*b12  a31*b13

    A0 = _mm256_shuffle_ps(T0, T0, _MM_SHUFFLE(2, 2, 2, 2));                // a0 = a02, a02, a02, a02, a12, a12, a12, a12
    A1 = _mm256_shuffle_ps(T1, T1, _MM_SHUFFLE(2, 2, 2, 2));                // a1 = a22, a22, a22, a22, a32, a32, a32, a32
    __m256 B1 = _mm256_permute2f128_ps(U1, U1, 0x00);                       // b0 = b20, b21, b22, b23, b20, b21, b22, b23
    __m256 C4 = _mm256_mul_ps(A0, B1);                                      // c4 = a02*b20  a02*b21  a02*b22  a02*b23  a12*b20  a12*b21  a12*b22  a12*b23
    __m256 C5 = _mm256_mul_ps(A1, B1);                                      // c5 = a22*b20  a22*b21  a22*b22  a22*b23  a32*b20  a32*b21  a32*b22  a32*b23

    A0 = _mm256_shuffle_ps(T0, T0, _MM_SHUFFLE(3, 3, 3, 3));                // a0 = a03, a03, a03, a03, a13, a13, a13, a13
    A1 = _mm256_shuffle_ps(T1, T1, _MM_SHUFFLE(3, 3, 3, 3));                // a1 = a23, a23, a23, a23, a33, a33, a33, a33
    B1 = _mm256_permute2f128_ps(U1, U1, 0x11);                              // b0 = b30, b31, b32, b33, b30, b31, b32, b33
    const __m256 C6 = _mm256_mul_ps(A0, B1);                                // c6 = a03*b30  a03*b31  a03*b32  a03*b33  a13*b30  a13*b31  a13*b32  a13*b33
    const __m256 C7 = _mm256_mul_ps(A1, B1);                                // c7 = a23*b30  a23*b31  a23*b32  a23*b33  a33*b30  a33*b31  a33*b32  a33*b33

    C0 = _mm256_add_ps(C0, C2);                                             // c0 = c0 + c2 (two terms, first two rows)
    C4 = _mm256_add_ps(C4, C6);                                             // c4 = c4 + c6 (the other two terms, first two rows)
    C1 = _mm256_add_ps(C1, C3);                                             // c1 = c1 + c3 (two terms, second two rows)
    C5 = _mm256_add_ps(C5, C7);                                             // c5 = c5 + c7 (the other two terms, second two rose)

    // Finally complete addition of all four terms and return the results
    Result.Row256[0] = _mm256_add_ps(C0, C4);   // n0 = a00*b00+a01*b10+a02*b20+a03*b30  a00*b01+a01*b11+a02*b21+a03*b31  a00*b02+a01*b12+a02*b22+a03*b32  a00*b03+a01*b13+a02*b23+a03*b33
                                                //      a10*b00+a11*b10+a12*b20+a13*b30  a10*b01+a11*b11+a12*b21+a13*b31  a10*b02+a11*b12+a12*b22+a13*b32  a10*b03+a11*b13+a12*b23+a13*b33
    Result.Row256[1] = _mm256_add_ps(C1, C5);   // n1 = a20*b00+a21*b10+a22*b20+a23*b30  a20*b01+a21*b11+a22*b21+a23*b31  a20*b02+a21*b12+a22*b22+a23*b32  a20*b03+a21*b13+a22*b23+a23*b33
                                                //      a30*b00+a31*b10+a32*b20+a33*b30  a30*b01+a31*b11+a32*b21+a33*b31  a30*b02+a31*b12+a32*b22+a33*b32  a30*b03+a31*b13+a32*b23+a33*b33
#elif SSE
    for (int32 i = 0; i < 4; i++)
    {
        __m128 R = Row[i];
        __m128 R0 = _mm_mul_ps(_mm_shuffle_ps(R, R, 0x00), Other.Row[0]);
        __m128 R1 = _mm_mul_ps(_mm_shuffle_ps(R, R, 0x55), Other.Row[1]);
        __m128 R2 = _mm_mul_ps(_mm_shuffle_ps(R, R, 0xAA), Other.Row[2]);
        __m128 R3 = _mm_mul_ps(_mm_shuffle_ps(R, R, 0xFF), Other.Row[3]);
        Result.Row[i] = _mm_add_ps(_mm_add_ps(R0, R1), _mm_add_ps(R2, R3));
    }
#else
    for (int32 i = 0; i < 4; i++)
    {
        for (int32 j = 0; j < 4; j++)
            for (int32 k = 0; k < 4; k++)
                Result.M[i][j] += M[i][k] * Other.M[k][j];
    }
#endif
    return Result;
}

// 스칼라 곱셈
FMatrix FMatrix::operator*(float Scalar) const {
    FMatrix Result;
#if AVX
    const __m256 S = _mm256_set1_ps(Scalar);
    Result.Row256[0] = _mm256_mul_ps(Row256[0], S);
    Result.Row256[1] = _mm256_mul_ps(Row256[1], S);
#elif SSE
    for (int32 i = 0; i < 4; i++)
    {
        Result.Row[i] = _mm_mul_ps(Row[i], _mm_set1_ps(Scalar));
    }
#else
    for (int32 i = 0; i < 4; i++)
    {
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = M[i][j] * Scalar;
    }
#endif
    return Result;
}

// 스칼라 나눗셈
FMatrix FMatrix::operator/(float Scalar) const {
    assert(Scalar != 0.f);
    FMatrix Result;
#if AVX
    const __m256 S = _mm256_set1_ps(Scalar);
    Result.Row256[0] = _mm256_div_ps(Row256[0], S);
    Result.Row256[1] = _mm256_div_ps(Row256[1], S);
#elif SSE
    for (int32 i = 0; i < 4; i++)
    {
        Result.Row[i] = _mm_div_ps(Row[i], _mm_set1_ps(Scalar));
    }
#else
    for (int32 i = 0; i < 4; i++)
    {
        for (int32 j = 0; j < 4; j++)
            Result.M[i][j] = M[i][j] / Scalar;
    }
#endif
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
    //TODO: Transpose using AVX
#if SSE or AVX
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
#if SSE or AVX
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
#if AVX
    // https://gist.github.com/recp/8ccc5ad0d19f5516de55f9bf7b5045b2
    __m256  Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13;
    __m256  Yt0, Yt1, Yt2;
    __m256  T0, T1, T2;
    __m256  R1, R2;
    __m256  Flpsign;
    __m256i Yi1, Yi2, Yi3;

    Y0 = Mat.Row256[0]; /* h g f e d c b a */
    Y1 = Mat.Row256[1]; /* p o n m l k j i */

    Y2 = _mm256_permute2f128_ps(Y1, Y1, 0x00); /* l k j i l k j i */
    Y3 = _mm256_permute2f128_ps(Y1, Y1, 0x11); /* p o n m p o n m */
    Y4 = _mm256_permute2f128_ps(Y0, Y0, 0x03); /* d c b a h g f e */
    Y13 = _mm256_permute2f128_ps(Y4, Y4, 0x00); /* h g f e h g f e */

    Yi1 = _mm256_set_epi32(0, 0, 0, 0, 0, 1, 1, 2);
    Yi2 = _mm256_set_epi32(1, 1, 1, 2, 3, 2, 3, 3);
    Flpsign = _mm256_set_ps(0.f, -0.f, 0.f, -0.f, -0.f, 0.f, -0.f, 0.f);

    /* i i i i i j j k */
    /* n n n o p o p p */
    /* m m m m m n n o */
    /* j j j k l k l l */
    /* e e e e e f f g */
    /* f f f g h g h h */
    Y5 = _mm256_permutevar_ps(Y2, Yi1);
    Y6 = _mm256_permutevar_ps(Y3, Yi2);
    Y7 = _mm256_permutevar_ps(Y3, Yi1);
    Y8 = _mm256_permutevar_ps(Y2, Yi2);
    Y2 = _mm256_permutevar_ps(Y13, Yi1);
    Y3 = _mm256_permutevar_ps(Y13, Yi2);

    Yi1 = _mm256_set_epi32(2, 1, 0, 0, 2, 1, 0, 0);
    Yi2 = _mm256_set_epi32(2, 1, 1, 0, 2, 1, 1, 0);
    Yi3 = _mm256_set_epi32(3, 3, 2, 0, 3, 3, 2, 0);

    /*
     t0[0] = k * p - o * l;    t1[0] = g * p - o * h;    t2[0] = g * l - k * h;
     t0[1] = j * p - n * l;    t1[1] = f * p - n * h;    t2[1] = f * l - j * h;
     t0[2] = j * o - n * k;    t1[2] = f * o - n * g;    t2[2] = f * k - j * g;
     t0[3] = i * p - m * l;    t1[3] = e * p - m * h;    t2[3] = e * l - i * h;
     t0[4] = i * o - m * k;    t1[4] = e * o - m * g;    t2[4] = e * k - i * g;
     t0[5] = i * n - m * j;    t1[5] = e * n - m * f;    t2[5] = e * j - i * f;
     */
    Yt0 = _mm256_sub_ps(_mm256_mul_ps(Y5, Y6), _mm256_mul_ps(Y7, Y8));
    Yt1 = _mm256_sub_ps(_mm256_mul_ps(Y2, Y6), _mm256_mul_ps(Y7, Y3));
    Yt2 = _mm256_sub_ps(_mm256_mul_ps(Y2, Y8), _mm256_mul_ps(Y5, Y3));

    /* t3 t2 t1 t0 t3 t2 t1 t0 */
    /* t5 t5 t5 t4 t5 t5 t5 t4 */
    Y9 = _mm256_permute2f128_ps(Yt0, Yt0, 0x00);
    Y10 = _mm256_permute2f128_ps(Yt0, Yt0, 0x11);

    /* t2 t1 t0 t0 t2 t1 t0 t0 */
    T0 = _mm256_permutevar_ps(Y9, Yi1);

    /* t4 t3 t3 t1 t4 t3 t3 t1 */
    Y11 = _mm256_shuffle_ps(Y9, Y10, 0x4D);
    Y12 = _mm256_permutevar_ps(Y11, Yi2);
    T1 = _mm256_permute2f128_ps(Y12, Y9, 0x00);

    /* t5 t5 t4 t2 t5 t5 t4 t2 */
    Y11 = _mm256_shuffle_ps(Y9, Y10, 0x4A);
    Y12 = _mm256_permutevar_ps(Y11, Yi3);
    T2 = _mm256_permute2f128_ps(Y12, Y12, 0x00);

    /* a a a b e e e f */
    /* b b c c f f g g */
    /* c d d d g h h h */
    Y9 = _mm256_permute_ps(Y4, 0x01);
    Y10 = _mm256_permute_ps(Y4, 0x5A);
    Y11 = _mm256_permute_ps(Y4, 0xBF);

    /*
     dest[0][0] =  f * t[0] - g * t[1] + h * t[2];
     dest[1][0] =-(e * t[0] - g * t[3] + h * t[4]);
     dest[2][0] =  e * t[1] - f * t[3] + h * t[5];
     dest[3][0] =-(e * t[2] - f * t[4] + g * t[5]);
     dest[0][1] =-(b * t[0] - c * t[1] + d * t[2]);
     dest[1][1] =  a * t[0] - c * t[3] + d * t[4];
     dest[2][1] =-(a * t[1] - b * t[3] + d * t[5]);
     dest[3][1] =  a * t[2] - b * t[4] + c * t[5];
     */
    R1 = _mm256_xor_ps(_mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(Y9, T0),
        _mm256_mul_ps(Y10, T1)),
        _mm256_mul_ps(Y11, T2)),
        Flpsign);

    /* d c b a d c b a */
    Y2 = _mm256_permute2f128_ps(Y0, Y0, 0x0);

    /* a a a b a a a b */
    /* b b c c b b c c */
    /* c d d d c d d d */
    Y3 = _mm256_permutevar_ps(Y2, _mm256_set_epi32(0, 0, 0, 1, 0, 0, 0, 1));
    Y4 = _mm256_permutevar_ps(Y2, _mm256_set_epi32(1, 1, 2, 2, 1, 1, 2, 2));
    Y5 = _mm256_permutevar_ps(Y2, _mm256_set_epi32(2, 3, 3, 3, 2, 3, 3, 3));

    /* t2[3] t2[2] t2[1] t2[0] t1[3] t1[2] t1[1] t1[0] */
    /* t2[5] t2[5] t2[5] t2[4] t1[5] t1[5] t1[5] t1[4] */
    Y6 = _mm256_permute2f128_ps(Yt1, Yt2, 0x20);
    Y7 = _mm256_permute2f128_ps(Yt1, Yt2, 0x31);

    /* t2[2] t2[1] t2[0] t2[0] t1[2] t1[1] t1[0] t1[0] */
    T0 = _mm256_permutevar_ps(Y6, Yi1);

    /* t1[4] t1[3] t1[3] t1[1] t1[4] t1[3] t1[3] t1[1] */

    /* t1[4] t1[3] t1[3] t1[1] t1[4] t1[3] t1[3] t1[1] */
    Y11 = _mm256_shuffle_ps(Y6, Y7, 0x4D);
    T1 = _mm256_permutevar_ps(Y11, Yi2);


    /* t2[5] t2[5] t2[4] t2[2] t1[5] t1[5] t1[4] t1[2] */
    Y11 = _mm256_shuffle_ps(Y6, Y7, 0x4A);
    T2 = _mm256_permutevar_ps(Y11, Yi3);

    /*
     dest[0][2] =  b * t1[0] - c * t1[1] + d * t1[2];
     dest[1][2] =-(a * t1[0] - c * t1[3] + d * t1[4]);
     dest[2][2] =  a * t1[1] - b * t1[3] + d * t1[5];
     dest[3][2] =-(a * t1[2] - b * t1[4] + c * t1[5]);
     dest[0][3] =-(b * t2[0] - c * t2[1] + d * t2[2]);
     dest[1][3] =  a * t2[0] - c * t2[3] + d * t2[4];
     dest[2][3] =-(a * t2[1] - b * t2[3] + d * t2[5]);
     dest[3][3] =  a * t2[2] - b * t2[4] + c * t2[5];
     */
    R2 = _mm256_xor_ps(_mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(Y3, T0),
        _mm256_mul_ps(Y4, T1)),
        _mm256_mul_ps(Y5, T2)),
        Flpsign);

    /* determinant */

    Y4 = _mm256_dp_ps(Y0, R1, 0xff);
    Y4 = _mm256_permute2f128_ps(Y4, Y4, 0x00);

    Y5 = _mm256_div_ps(_mm256_set1_ps(1.0f), Y4);
    R1 = _mm256_mul_ps(R1, Y5);
    R2 = _mm256_mul_ps(R2, Y5);

    /* transpose */

    /* d c b a h g f e */
    /* l k j i p o n m */
    Y0 = _mm256_permute2f128_ps(R1, R1, 0x03);
    Y1 = _mm256_permute2f128_ps(R2, R2, 0x03);

    /* b a f e f e b a */
    /* j i n m n m j i */
    /* i m a e m i e a */
    /* j n b f n j f b */
    /* n j f b m i e a */
    Y2 = _mm256_shuffle_ps(R1, Y0, 0x44);
    Y3 = _mm256_shuffle_ps(R2, Y1, 0x44);
    Y4 = _mm256_shuffle_ps(Y2, Y3, 0x88);
    Y5 = _mm256_shuffle_ps(Y2, Y3, 0xDD);
    Y6 = _mm256_permute2f128_ps(Y4, Y5, 0x20);

    /* d c h g h g d c */
    /* l k p o p o l k */
    /* k o c g o k g c */
    /* l p d h p l h d */
    /* p l h d o k g c */
    Y2 = _mm256_shuffle_ps(R1, Y0, 0xEE);
    Y3 = _mm256_shuffle_ps(R2, Y1, 0xEE);
    Y4 = _mm256_shuffle_ps(Y2, Y3, 0x88);
    Y5 = _mm256_shuffle_ps(Y2, Y3, 0xDD);
    Y7 = _mm256_permute2f128_ps(Y4, Y5, 0x20);

    FMatrix mResult;
    mResult.Row256[0] = Y6;
    mResult.Row256[1] = Y7;
    return mResult;
#elif SSE
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
#if SSE
    // Identity이므로 M[3][3] = 1
    translationMatrix.Row[3] = _mm_set_ps(1.0f, position.z, position.y, position.x);
#else
    translationMatrix.M[3][0] = position.x;
    translationMatrix.M[3][1] = position.y;
    translationMatrix.M[3][2] = position.z;
#endif
    return translationMatrix;
}

#if SSE
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
#if AVX //@TODO: TransformVector(FVector) - SSE와 차이가 없음. 개선 방안
    __m256 V0 = _mm256_set_m128(_mm_set1_ps(Vec.y), _mm_set1_ps(Vec.x));
    __m256 V1 = _mm256_set_m128(_mm_setzero_ps(), _mm_set1_ps(Vec.z));

    __m256 R = _mm256_add_ps(_mm256_mul_ps(V0, Mat.Row256[0]), _mm256_mul_ps(V1, Mat.Row256[1]));

    _mm_storeu_ps(reinterpret_cast<float*>(&Result), _mm_add_ps(_mm256_castps256_ps128(R), _mm256_extractf128_ps(R, 1)));
#elif SSE
    __m128 VX = _mm_set_ps1(Vec.x);
    __m128 VY = _mm_set_ps1(Vec.y);
    __m128 VZ = _mm_set_ps1(Vec.z);

    __m128 R = _mm_mul_ps(VX, Mat.Row[0]);
    R = _mm_add_ps(R, _mm_mul_ps(VY, Mat.Row[1]));
    R = _mm_add_ps(R, _mm_mul_ps(VZ, Mat.Row[2]));

    //@Note: Vector는 float[3]이므로 reinterpret_cast를 통해 __m128을 변환 시 메모리 침식이 일어남
    //_mm_storeu_ps(reinterpret_cast<float*>(&Result), R);
    _mm_store_ss(&Result.x, R);
    R = _mm_shuffle_ps(R, R, _MM_SHUFFLE(0, 3, 2, 1));
    _mm_store_ss(&Result.y, R);
    R = _mm_shuffle_ps(R, R, _MM_SHUFFLE(0, 3, 2, 1));
    _mm_store_ss(&Result.z, R);
#else
    // 4x4 행렬을 사용하여 벡터 변환 (W = 0으로 가정, 방향 벡터)
    Result.x = Vec.x * Mat.M[0][0] + Vec.y * Mat.M[1][0] + Vec.z * Mat.M[2][0] + 0.0f * Mat.M[3][0];
    Result.y = Vec.x * Mat.M[0][1] + Vec.y * Mat.M[1][1] + Vec.z * Mat.M[2][1] + 0.0f * Mat.M[3][1];
    Result.z = Vec.x * Mat.M[0][2] + Vec.y * Mat.M[1][2] + Vec.z * Mat.M[2][2] + 0.0f * Mat.M[3][2];
#endif

    return Result;
}

// FVector4를 변환하는 함수
FVector4 FMatrix::TransformVector(const FVector4& Vec, const FMatrix& Mat)
{
    FVector4 Result;
#if AVX //@TODO: TransformVector(FVector4) - SSE와 차이가 없음. 개선 방안
    __m256 V0 = _mm256_set_m128(_mm_set1_ps(Vec.y), _mm_set1_ps(Vec.x));
    __m256 V1 = _mm256_set_m128(_mm_set1_ps(Vec.a), _mm_set1_ps(Vec.z));

    __m256 R = _mm256_add_ps(_mm256_mul_ps(V0, Mat.Row256[0]), _mm256_mul_ps(V1, Mat.Row256[1]));

    _mm_storeu_ps(reinterpret_cast<float*>(&Result), _mm_add_ps(_mm256_castps256_ps128(R), _mm256_extractf128_ps(R, 1)));
#elif SSE
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
    Result.x = Vec.x * Mat.M[0][0] + Vec.y * Mat.M[1][0] + Vec.z * Mat.M[2][0] + Vec.a * Mat.M[3][0];
    Result.y = Vec.x * Mat.M[0][1] + Vec.y * Mat.M[1][1] + Vec.z * Mat.M[2][1] + Vec.a * Mat.M[3][1];
    Result.z = Vec.x * Mat.M[0][2] + Vec.y * Mat.M[1][2] + Vec.z * Mat.M[2][2] + Vec.a * Mat.M[3][2];
    Result.a = Vec.x * Mat.M[0][3] + Vec.y * Mat.M[1][3] + Vec.z * Mat.M[2][3] + Vec.a * Mat.M[3][3];
#endif
    return Result;
}

FVector FMatrix::TransformPosition(const FVector& Vec) const
{
#if SSE or AVX //@TODO: TransformPosition(FVector) - SSE와 차이가 없음. 미구현
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

    //@Note: Vector는 float[3]이므로 reinterpret_cast를 통해 __m128을 변환 시 메모리 침식이 일어남
    //_mm_storeu_ps(reinterpret_cast<float*>(&Result), R);
    _mm_store_ss(&Result.x, R);
    R = _mm_shuffle_ps(R, R, _MM_SHUFFLE(0, 3, 2, 1));
    _mm_store_ss(&Result.y, R);
    R = _mm_shuffle_ps(R, R, _MM_SHUFFLE(0, 3, 2, 1));
    _mm_store_ss(&Result.z, R);
    return Result;
#else
	float x = M[0][0] * Vec.x + M[1][0] * Vec.y + M[2][0] * Vec.z + M[3][0];
	float y = M[0][1] * Vec.x + M[1][1] * Vec.y + M[2][1] * Vec.z + M[3][1];
	float z = M[0][2] * Vec.x + M[1][2] * Vec.y + M[2][2] * Vec.z + M[3][2];
	float w = M[0][3] * Vec.x + M[1][3] * Vec.y + M[2][3] * Vec.z + M[3][3];
	return w != 0.0f ? FVector{ x / w, y / w, z / w } : FVector{ x, y, z };
#endif
}
