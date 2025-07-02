#version 450 core

// 定义一个输出变量，它将作为片元的颜色。
// layout(location = 0) 表示这个颜色将输出到第一个颜色附件。
layout(location = 0) out vec4 outColor;

void main()
{
    // 输出一个固定的颜色。
    // 这里我们输出不透明的洋红色 (R=1, G=0, B=1, A=1)。
    // 这是一个常用的调试颜色，因为它很显眼。
    outColor = vec4(1.0, 0.0, 1.0, 1.0);
}