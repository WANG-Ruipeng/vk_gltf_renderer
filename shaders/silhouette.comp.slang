// shader.slang

struct PushConstantSilhouette
{
  float3 color;
};

[[vk::binding(0)]]
Texture2D<float> u_R8Buffer;
[[vk::binding(1)]]
RWTexture2D<float4> u_RgbaImage;
[[vk::push_constant]]
ConstantBuffer<PushConstantSilhouette> pushConst;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  uint2 texSize;
  u_RgbaImage.GetDimensions(texSize.x, texSize.y); // DispatchRaysDimensions();

  const uint2 pixelCoord = uint2(dispatchThreadID.xy);

  // Define the Sobel kernel in the x and y directions
  const float kx0 = -1.0, kx1 = 0.0, kx2 = 1.0;
  const float kx3 = -2.0, kx4 = 0.0, kx5 = 2.0;
  const float kx6 = -1.0, kx7 = 0.0, kx8 = 1.0;

  const float ky0 = -1.0, ky1 = -2.0, ky2 = -1.0;
  const float ky3 = 0.0, ky4 = 0.0, ky5 = 0.0;
  const float ky6 =  1.0, ky7 =  2.0, ky8 =  1.0;

  float sumX = 0.0;
  float sumY = 0.0;

  // Check boundaries
  if (pixelCoord.x > 0 && pixelCoord.x < texSize.x - 1 && pixelCoord.y > 0 &&
      pixelCoord.y < texSize.y - 1)
  {
    [unroll]
    for (int i = -1; i <= 1; i++)
    {
      [unroll]
      for (int j = -1; j <= 1; j++)
      {
        const int index = (i + 1) * 3 + (j + 1);
        const uint2 offset = uint2(i, j);
        const float val = u_R8Buffer[pixelCoord + offset].r;
        float kx = 0.0, ky = 0.0;
        switch (index)
        {
        case 0: kx = kx0; ky = ky0; break;
        case 1: kx = kx1; ky = ky1; break;
        case 2: kx = kx2; ky = ky2; break;
        case 3: kx = kx3; ky = ky3; break;
        case 4: kx = kx4; ky = ky4; break;
        case 5: kx = kx5; ky = ky5; break;
        case 6: kx = kx6; ky = ky6; break;
        case 7: kx = kx7; ky = ky7; break;
        case 8: kx = kx8; ky = ky8; break;
        }
        sumX += val * kx;
        sumY += val * ky;
      }
    }
  }

  // Compute the magnitude of the gradient
  const float magnitude = (sumX * sumX + sumY * sumY);

  // If the magnitude is above a certain threshold, we consider it an edge
  const float min_threshold = 2.0;
  const float max_threshold = 20.0;

  // Write the result to the output image
  u_RgbaImage[pixelCoord].xyz =
      lerp(u_RgbaImage[pixelCoord].xyz, pushConst.color,
           smoothstep(min_threshold, max_threshold, magnitude));
}
