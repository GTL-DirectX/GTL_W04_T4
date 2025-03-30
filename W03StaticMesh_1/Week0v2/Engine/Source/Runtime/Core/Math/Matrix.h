#pragma once

#include <DirectXMath.h>

// 4x4 행렬 연산
union FMatrix
{
	float M[4][4];
    __m128 Row[4];
    __m256 Row256[2];   // TODO: M의 주소를 'reinterpret_cast<const float*>(Mat.M) + 8' 의 형태로 사용하면 필요 없음, __m128 Row[4] 또한 마찬가지 (하지만 변환에 드는 비용은???)

	static const FMatrix Identity;
	// 기본 연산자 오버로딩
	FMatrix operator+(const FMatrix& Other) const;
	FMatrix operator-(const FMatrix& Other) const;
	FMatrix operator*(const FMatrix& Other) const;
#ifdef SSE
    __m128 MulVecMat(const __m128& Vector, const FMatrix& Matrix) const;
#endif
	FMatrix operator*(float Scalar) const;
	FMatrix operator/(float Scalar) const;
	float* operator[](int row);
	const float* operator[](int row) const;
	
	// 유틸리티 함수
	static FMatrix Transpose(const FMatrix& Mat);
	static float Determinant(const FMatrix& Mat);
	static FMatrix Inverse(const FMatrix& Mat);
	static FMatrix CreateRotation(float roll, float pitch, float yaw);
	static FMatrix CreateScale(float scaleX, float scaleY, float scaleZ);
	static FVector TransformVector(const FVector& v, const FMatrix& m);
	static FVector4 TransformVector(const FVector4& v, const FMatrix& m);
	static FMatrix CreateTranslationMatrix(const FVector& position);

    // 2x2 행렬
    static float Det2x2(__m128 Mat);
    static __m128 Inverse2x2(__m128 Mat);
    static __m128 InverseDet2x2(__m128 Mat, float Det);
    static __m128 Mul2x2(__m128 LMat, __m128 RMat);
    static __m128 Sub2x2(__m128 LMat, __m128 RMat);
    static void Split4x4To2x2(const FMatrix& Mat, __m128& A, __m128& B, __m128& C, __m128& D);

	DirectX::XMMATRIX ToXMMATRIX() const
	{
		return DirectX::XMMatrixSet(
			M[0][0], M[1][0], M[2][0], M[3][0], // 첫 번째 열
			M[0][1], M[1][1], M[2][1], M[3][1], // 두 번째 열
			M[0][2], M[1][2], M[2][2], M[3][2], // 세 번째 열
			M[0][3], M[1][3], M[2][3], M[3][3]  // 네 번째 열
		);
	}
	FVector4 TransformFVector4(const FVector4& vector)
	{
		return FVector4(
			M[0][0] * vector.x + M[1][0] * vector.y + M[2][0] * vector.z + M[3][0] * vector.a,
			M[0][1] * vector.x + M[1][1] * vector.y + M[2][1] * vector.z + M[3][1] * vector.a,
			M[0][2] * vector.x + M[1][2] * vector.y + M[2][2] * vector.z + M[3][2] * vector.a,
			M[0][3] * vector.x + M[1][3] * vector.y + M[2][3] * vector.z + M[3][3] * vector.a
		);
	}
    FVector TransformPosition(const FVector& vector) const;
};