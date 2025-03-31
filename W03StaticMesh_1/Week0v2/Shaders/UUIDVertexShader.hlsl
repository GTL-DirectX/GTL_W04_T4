// MatrixBuffer: 변환 행렬 관리
cbuffer MatrixConstants : register(b0)
{
    row_major float4x4 MVP;
    float4 UUID;
};

struct VS_INPUT
{
    float4 position : POSITION; // 버텍스 위치
};

struct PS_INPUT
{
    float4 position : SV_POSITION; // 변환된 화면 좌표
};

PS_INPUT mainVS(VS_INPUT input)
{
    PS_INPUT output;
    
    // 위치 변환
    output.position = mul(input.position, MVP);
    
    return output;
}