#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable

// 包含项目中已有的、用于获取相机矩阵等信息的头文件
// 注意：这里的 #include 只用于 main 函数内部的结构体定义，不用于 layout
#include "device_host.h" 

// 定义Compute Shader的本地工作组大小。
// 这个值需要和C++代码中vkCmdDispatch的groupSize匹配，以获得最佳性能。
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// 绑定我们在C++中定义的描述符集
// 使用字面量整数 0, 1, 2 而不是枚举名
layout(set = 0, binding = 0) uniform _SdfFrameInfo { SceneFrameInfo frameInfo; };
layout(set = 0, binding = 1) uniform sampler3D sdfTexture;
layout(set = 0, binding = 2, rgba32f) uniform image2D outputImage;

layout(push_constant) uniform _PushConstantSdf {
    float sdf_slice_depth;
    int visualization_mode; // 0 = Ray Marching, 1 = Slice View
};

vec3 sdfToColor(float dist) {
    // 内部 (dist < 0): 显示为蓝色
    float epsilon = 0.01;
    if (dist < -epsilon) {
        return vec3(0.0, 0.0, 1.0);
    }
    // 表面附近 (dist ≈ 0): 显示为绿色
    else if (abs(dist) <= epsilon) {
        return vec3(0.0, 1.0, 0.0);
    }
    // 外部 (dist > 0): 根据距离远近显示不同亮度的红色
    else {
        // 使用 smoothstep 让颜色过渡更平滑
        float intensity = smoothstep(0.0, max(frameInfo.sdf_bbox_ext.x, frameInfo.sdf_bbox_ext.y) * 0.1, dist);
        return vec3(intensity, 0.0, 0.0);
    }
}

bool is_inside_box(vec3 p) {
    vec3 uvw = (p - frameInfo.sdf_bbox_min) / frameInfo.sdf_bbox_ext;
    // all() 函数会检查所有分量是否都为true
    return all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)));
}

// 这是用于可视化的函数
void visualizeSdfSlice(ivec2 pixelCoord, ivec2 imageSize) {
    // 1. 定义你想要可视化的SDF纹理切片深度
    // z_slice 的范围是 0.0 到 1.0，代表SDF包围盒从最小Z到最大Z的位置
    float z_slice = sdf_slice_depth; //  <-- 修改这个值来查看不同的Z轴切片！

    // 2. 将像素坐标转换为SDF的UV坐标
    vec2 uv = vec2(pixelCoord) / vec2(imageSize);

    // 3. 构建3D纹理坐标 (u, v, w)，其中w是我们的切片深度
    vec3 uvw = vec3(uv.x, uv.y, z_slice);

    // 4. 从SDF纹理中采样原始距离值（这个值是归一化的）
    // 注意：我们直接用 texture 函数，而不是 sampleSdf，因为我们想看原始数据
    float raw_dist = texture(sdfTexture, uvw).r;

    // 5. 将SDF距离值转换为颜色
    vec3 color = sdfToColor(raw_dist);

    // 6. 将最终颜色写入输出图像
    imageStore(outputImage, pixelCoord, vec4(color, 1.0));
}

float sdBox( vec3 p, vec3 c, vec3 b )
{
    p = p - c; // 将坐标系移到以包围盒中心为原点
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sampleSdf(vec3 pos) {
    if (frameInfo.sdf_bbox_ext.x < 0.0001) {
        return 1000.0;
    }

    // 首先，检查点是否在包围盒内部
    if (is_inside_box(pos)) {
        // 如果在内部，执行我们原来的逻辑：采样SDF纹理
        vec3 uvw = (pos - frameInfo.sdf_bbox_min) / frameInfo.sdf_bbox_ext;
        return texture(sdfTexture, uvw).r * max(max(frameInfo.sdf_bbox_ext.x, frameInfo.sdf_bbox_ext.y), frameInfo.sdf_bbox_ext.z);
    } else {
        // 如果在外部，计算到包围盒的精确距离
        vec3 box_center = frameInfo.sdf_bbox_min + frameInfo.sdf_bbox_ext / 2.0;
        vec3 box_half_extents = frameInfo.sdf_bbox_ext / 2.0;
        return sdBox(pos, box_center, box_half_extents);
    }
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
    // 获取像素坐标
    ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(outputImage);
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y) {
        return;
    }

    // 模式切换
    if (visualization_mode == 1)
    {
        visualizeSdfSlice(pixelCoord, imageSize);
    }
    else // 3D光线步进模式 (新的 sampleSdf 可视化逻辑)
    {
        // 1. 生成相机光线
        vec2 ndc = (vec2(pixelCoord) + 0.5) / vec2(imageSize) * 2.0 - 1.0;
        vec4 target = frameInfo.projMatrixI * vec4(ndc.x, ndc.y, 1.0, 1.0);
        vec3 rayDir = normalize(vec3(frameInfo.viewMatrixI * vec4(normalize(target.xyz / target.w), 0.0)));
        vec3 rayOrigin = frameInfo.camPos;

        // ====================================================================
        //  核心调试逻辑: 可视化 sampleSdf 的返回值
        // ====================================================================

        // 我们选择一个固定的测试点，例如相机前方 10.0 个单位的点
        // 你可以调整这个距离来观察不同位置的SDF值
        float test_distance = 100.0;
        vec3 test_point = rayOrigin + rayDir * test_distance;

        // 直接调用 sampleSdf 获取该点的距离值
        float sdf_dist = sampleSdf(test_point);

        // 将这个距离值映射为颜色
        // 正距离 (外部): 红色 (越远越亮)
        // 负距离 (内部): 蓝色
        // 零距离 (表面): 绿色
        vec3 finalColor = sdfToColor(sdf_dist);

        // 如果点在包围盒外部，我们用紫色叠加，以示区分
        if (!is_inside_box(test_point)) {
            finalColor += vec3(0.5, 0.0, 0.5); // 叠加紫色
        }

        imageStore(outputImage, pixelCoord, vec4(finalColor, 1.0));
    }
}