cbuffer MatrixConstants : register(b0)
{
    row_major float4x4 MVP;
    float4 UUID;
};

struct PS_INPUT
{
    float4 position : SV_POSITION; // 변환된 화면 좌표
};

struct PS_OUTPUT
{
    float4 UUID : SV_Target0;   // a.k.a. color
};

PS_OUTPUT mainPS(PS_INPUT input)
{
    PS_OUTPUT output;
    
    output.UUID = UUID;
            
    return output;
}