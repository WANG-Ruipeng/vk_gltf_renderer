// src/renderer_sdf.cpp

#include "renderer.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvkhl/shaders/dh_tonemap.h"

#include "nvvk/error_vk.hpp"

#include "_autogen/sdf.comp.glsl.h"

namespace gltfr {

class RendererSdf : public Renderer
{
public:
    RendererSdf() = default;
    ~RendererSdf() override { deinit(); }

    bool init(Resources& res, Scene& scene) override;
    void deinit(Resources& res) override { deinit(); }
    void render(VkCommandBuffer cmd, Resources& res, Scene& scene, Settings& settings, nvvk::ProfilerVK& profiler) override;
    void handleChange(Resources& res, Scene& scene) override;
    bool onUI() override { return false; }
    VkDescriptorImageInfo getOutputImage() const override;
    bool reloadShaders(Resources& res, Scene& scene) override;

private:
    void createSDFPipeline(Resources& res, Scene& scene);
    void createGBuffer(Resources& res);
    void updateSdfDescriptorSet(Scene& scene);
    void deinit();

    nvvk::DescriptorSetContainer m_descSetContainer;
    std::unique_ptr<nvvkhl::GBuffer> m_gBuffer;

    VkPipelineLayout m_pipelineLayout{};
    VkPipeline       m_pipeline{};
    VkDevice         m_device{VK_NULL_HANDLE};
};


std::unique_ptr<Renderer> makeRendererSdf()
{
    return std::make_unique<RendererSdf>();
}

bool RendererSdf::init(Resources& res, Scene& scene)
{
    m_device = res.ctx.device;
    m_gBuffer = std::make_unique<nvvkhl::GBuffer>(m_device, res.m_allocator.get());
    createGBuffer(res);
    m_descSetContainer.init(m_device);
    createSDFPipeline(res, scene);
    updateSdfDescriptorSet(scene);
    return true;
}

void RendererSdf::deinit()
{
    if (m_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    m_descSetContainer.deinit();
    if (m_gBuffer)
    {
        m_gBuffer->destroy();
    }
}

void RendererSdf::render(VkCommandBuffer cmd, Resources& res, Scene& scene, Settings& settings, nvvk::ProfilerVK& profiler)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

    std::vector<VkDescriptorSet> descSets = {
        m_descSetContainer.getSet(),
        scene.m_sceneDescriptorSet
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
                            static_cast<uint32_t>(descSets.size()), descSets.data(), 0, nullptr);

    auto extent = res.m_finalImage->getSize();
    // 这里的 8 需要与 sdf.comp.glsl 中的 local_size_x/y 匹配
    uint32_t groupSizeX = 8; 
    uint32_t groupSizeY = 8;
    
    vkCmdDispatch(cmd, (extent.width + groupSizeX - 1) / groupSizeX, 
                       (extent.height + groupSizeY - 1) / groupSizeY, 1);
}

void RendererSdf::createSDFPipeline(Resources& res, Scene& scene)
{
    m_descSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_descSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_descSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    m_descSetContainer.initLayout();
    m_descSetContainer.initPool(1);

    std::vector<VkDescriptorSetLayout> layouts = {
        m_descSetContainer.getLayout(),
        scene.m_sceneDescriptorSetLayout
    };
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = static_cast<uint32_t>(layouts.size());
    layout_info.pSetLayouts    = layouts.data();
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &layout_info, nullptr, &m_pipelineLayout));

    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = nvvk::createShaderModule(m_device, sdf_comp_glsl, sizeof(sdf_comp_glsl));
    stage_info.pName  = "main";

    VkComputePipelineCreateInfo compute_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compute_info.layout = m_pipelineLayout;
    compute_info.stage  = stage_info;
    NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compute_info, nullptr, &m_pipeline));

    vkDestroyShaderModule(m_device, stage_info.module, nullptr);
}

void RendererSdf::handleChange(Resources& res, Scene& scene)
{
    if (res.hasGBuffersChanged()) {
        createGBuffer(res);
        updateSdfDescriptorSet(scene);
    }
}

bool RendererSdf::reloadShaders(Resources& res, Scene& scene)
{
    return true; 
}

void RendererSdf::createGBuffer(Resources& res) 
{
    if(m_gBuffer) m_gBuffer->destroy();
    std::vector<VkFormat> color_format = { VK_FORMAT_R32G32B32A32_SFLOAT };
    m_gBuffer->create(res.m_finalImage->getSize(), color_format, VK_FORMAT_UNDEFINED);
}

void RendererSdf::updateSdfDescriptorSet(Scene& scene)
{
    VkDescriptorBufferInfo sceneFrameInfoBinfo = {scene.m_sceneFrameInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo sdfTextureInfo = scene.m_sdfTexture.descriptor;
    VkDescriptorImageInfo outputImageInfo = m_gBuffer->getDescriptorImageInfo();
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkWriteDescriptorSet> writes;
    writes.push_back(m_descSetContainer.makeWrite(0, 0, &sceneFrameInfoBinfo));
    writes.push_back(m_descSetContainer.makeWrite(0, 1, &sdfTextureInfo));
    writes.push_back(m_descSetContainer.makeWrite(0, 2, &outputImageInfo));

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

VkDescriptorImageInfo RendererSdf::getOutputImage() const
{
    // 最终的显示结果现在由 m_gBuffer 管理
    // 但我们需要确保它被 Tonemapper 处理过，或者直接连接到最终输出
    // 这里我们先返回 G-Buffer， tonemapper会处理它
    return m_gBuffer ? m_gBuffer->getDescriptorImageInfo() : VkDescriptorImageInfo{};
}

}