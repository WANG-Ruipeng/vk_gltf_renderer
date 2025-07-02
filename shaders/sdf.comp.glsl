#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// 包含项目中已有的、用于获取相机矩阵等信息的头文件
#include "device_host.h" 

// 定义Compute Shader的本地工作组大小。
// 这个值需要和C++代码中vkCmdDispatch的groupSize匹配，以获得最佳性能。
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// 绑定我们在C++中定义的描述符集
// Set 0 是我们 RendererSdf 私有的描述符集
layout(set = 0, binding = 0) uniform _SdfFrameInfo { SceneFrameInfo frameInfo; };
layout(set = 0, binding = 1) uniform sampler3D sdfTexture;
layout(set = 0, binding = 2, rgba32f) uniform image2D outputImage;

// SDF采样函数
float sampleSdf(vec3 pos) {
    // 检查分母是否为零，防止除以零的错误
    if (frameInfo.sdf_bbox_ext.x < 0.0001 || frameInfo.sdf_bbox_ext.y < 0.0001 || frameInfo.sdf_bbox_ext.z < 0.0001) {
        return 1000.0; // 返回一个很大的距离值
    }
    // 将世界坐标转换为SDF纹理的UVW坐标 (0-1范围)
    vec3 uvw = (pos - frameInfo.sdf_bbox_min) / frameInfo.sdf_bbox_ext;
    // 采样并返回距离值。乘以包围盒的最大维度，将归一化的距离转换回世界空间单位
    return texture(sdfTexture, uvw).r * max(max(frameInfo.sdf_bbox_ext.x, frameInfo.sdf_bbox_ext.y), frameInfo.sdf_bbox_ext.z);
}

// 计算SDF表面的法线 (通过梯度)
vec3 calcNormal(vec3 pos) {
    // 使用一个很小的偏移量来近似计算梯度
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        sampleSdf(pos + e.xyy) - sampleSdf(pos - e.xyy),
        sampleSdf(pos + e.yxy) - sampleSdf(pos - e.yxy),
        sampleSdf(pos + e.yyx) - sampleSdf(pos - e.yyx)
    ));
}

void main()
{
    // 获取当前着色器实例正在处理的像素坐标
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(outputImage);

    // 防止越界（当图像尺寸不是工作组大小的整数倍时）
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y) {
        return;
    }

    // 1. 生成每个像素的相机光线
    // 将像素坐标转换为NDC坐标(-1到1)
    vec2 ndc = (vec2(pixelCoord) + 0.5) / vec2(imageSize) * 2.0 - 1.0;
    
    // 使用相机逆矩阵计算世界空间中的光线方向
    vec4 target = frameInfo.projMatrixI * vec4(ndc.x, ndc.y, 1.0, 1.0);
    vec3 rayDir = normalize(vec3(frameInfo.viewMatrixI * vec4(normalize(target.xyz / target.w), 0.0)));
    vec3 rayOrigin = frameInfo.camPos;

    // 2. 光线步进 (Ray Marching)
    float t = 0.0;      // 从相机开始的距离
    vec3 hitPos = vec3(0); // 命中点
    bool hit = false;
    for (int i = 0; i < 128; ++i) { // 最多步进128次
        vec3 p = rayOrigin + t * rayDir;
        float dist = sampleSdf(p);
        
        // 如果距离非常小，我们认为已经命中了表面
        if (dist < 0.001) { 
            hit = true;
            hitPos = p;
            break;
        }
        
        // 步进的距离就是SDF返回的距离，这是球体追踪的核心
        t += dist;
        
        // 如果距离太远，就停止追踪
        if (t > 2000.0) break;
    }

    // 3. 计算颜色并写入输出纹理
    vec3 finalColor = vec3(0.1, 0.1, 0.15); // 背景色
    if (hit) {
        // 如果命中，根据法线来可视化
        vec3 normal = calcNormal(hitPos);
        // 一个简单的lambert光照
        vec3 lightDir = normalize(vec3(0.5, 0.8, -0.3));
        float diffuse = max(dot(normal, lightDir), 0.0);
        finalColor = vec3(diffuse) + vec3(0.1); // 添加一点环境光
    }
    
    // 将最终颜色写入到输出图像的对应像素位置
    imageStore(outputImage, pixelCoord, vec4(finalColor, 1.0));
}