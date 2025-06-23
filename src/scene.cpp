/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scene.hpp"

#include "fileformats/tiny_converter.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/timesampler.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "settings.hpp"
#include "shaders/dh_bindings.h"
#include "tiny_obj_loader.h"
#include "create_tangent.hpp"
#include "nvvkhl/shaders/dh_tonemap.h"
#include "collapsing_header_manager.h"
#include "nvh/parallel_work.hpp"
#include <random>

extern std::shared_ptr<nvvkhl::ElementCamera> g_elemCamera;  // Is accessed elsewhere in the App
namespace PE = ImGuiH::PropertyEditor;

constexpr uint32_t MAXTEXTURES = 1000;  // Maximum textures allowed in the application

std::vector<glm::vec3> generateUniformSphereSamples(int numSamples)
{
    std::vector<glm::vec3> samples;
    samples.reserve(numSamples);

    std::mt19937 generator(0); // 使用固定的种子以保证每次结果一致
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (int i = 0; i < numSamples; ++i)
    {
        float theta = 2.0f * glm::pi<float>() * distribution(generator);
        float phi = std::acos(1.0f - 2.0f * distribution(generator));

        glm::vec3 sample(
            std::sin(phi) * std::cos(theta),
            std::sin(phi) * std::sin(theta),
            std::cos(phi)
        );
        samples.push_back(sample);
    }
    return samples;
}

// 1. 自定义上下文结构体，现在它将直接持有 RTCGeometry 句柄
class GltfEmbreeContext : public RTCPointQueryContext
{
  public:
  RTCGeometry geometry_handle;
  unsigned int numTriangles;
};

// 2. 一个标准的点到三角形最近点函数
glm::vec3 closestPointOnTriangle(const glm::vec3 &p, const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c)
{
  glm::vec3 ab = b - a, ac = c - a, ap = p - a;
  float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
  if (d1 <= 0 && d2 <= 0)
    return a;
  glm::vec3 bp = p - b;
  float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
  if (d3 >= 0 && d4 <= d3)
    return b;
  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0)
  {
    float v = d1 / (d1 - d3);
    return a + v * ab;
  }
  glm::vec3 cp = p - c;
  float d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
  if (d6 >= 0 && d5 <= d6)
    return c;
  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0)
  {
    float w = d2 / (d2 - d6);
    return a + w * ac;
  }
  float va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
  {
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + w * (c - b);
  }
  float denom = 1.0f / (va + vb + vc);
  float v = vb * denom, w = vc * denom;
  return a + ab * v + ac * w;
}

// 完全替换你现有的 rtcPointQueryCallback
bool rtcPointQueryCallback(RTCPointQueryFunctionArguments *args)
{
  // 1. 从 userPtr 获取我们的自定义数据
  PointQueryUserData *userData = static_cast<PointQueryUserData *>(args->userPtr);
  float *pClosestDistanceSq = userData->closestDistanceSq;

  // 2. 使用 args->geomID 从场景中动态获取当前命中的几何体
  RTCGeometry geometry = rtcGetGeometry(userData->scene, args->geomID);

  // 3. 后续代码几乎不变
  const float *vertexBuffer = static_cast<const float *>(rtcGetGeometryBufferData(geometry, RTC_BUFFER_TYPE_VERTEX, 0));
  const uint32_t *indexBuffer = static_cast<const uint32_t *>(rtcGetGeometryBufferData(geometry, RTC_BUFFER_TYPE_INDEX, 0));

  if (!vertexBuffer || !indexBuffer)
    return false;

  // primID 是在当前这个 geometry 内的索引
  const uint32_t i0 = indexBuffer[args->primID * 3 + 0];
  const uint32_t i1 = indexBuffer[args->primID * 3 + 1];
  const uint32_t i2 = indexBuffer[args->primID * 3 + 2];

  const glm::vec3 v0(vertexBuffer[i0 * 3], vertexBuffer[i0 * 3 + 1], vertexBuffer[i0 * 3 + 2]);
  const glm::vec3 v1(vertexBuffer[i1 * 3], vertexBuffer[i1 * 3 + 1], vertexBuffer[i1 * 3 + 2]);
  const glm::vec3 v2(vertexBuffer[i2 * 3], vertexBuffer[i2 * 3 + 1], vertexBuffer[i2 * 3 + 2]);

  glm::vec3 queryPosition(args->query->x, args->query->y, args->query->z);
  glm::vec3 closestPt = closestPointOnTriangle(queryPosition, v0, v1, v2);

  glm::vec3 diff = queryPosition - closestPt;
  float distSq = glm::dot(diff, diff);

  if (distSq < *pClosestDistanceSq)
  {
    *pClosestDistanceSq = distSq;
    args->query->radius = std::sqrt(distSq);
    return true;
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
// Initialization of the scene object
// - Create the buffers for the scene frame information
// - Create the sky
// - Create the empty HDR environment
void gltfr::Scene::init(Resources& res)
{
  nvvk::ResourceAllocator* alloc = res.m_allocator.get();

  createHdr(res, "");  // Initialize the environment with nothing (constant white: for now)
  m_sky = std::make_unique<nvvkhl::PhysicalSkyDome>();  // Sun&Sky
  m_sky->setup(res.ctx.device, alloc);

  // Create the buffer of the current frame, changing at each frame
  {
    m_sceneFrameInfoBuffer =
        res.m_allocator->createBuffer(sizeof(DH::SceneFrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    nvvk::DebugUtil(res.ctx.device).DBG_NAME(m_sceneFrameInfoBuffer.buffer);
  }

  VkDevice device = res.ctx.device;
  createDescriptorPool(device);
  createDescriptorSet(device);
}

//--------------------------------------------------------------------------------------------------
// De-initialization of the scene object
// - Destroy the buffers
// - Reset the scene objects
// - Reset the HDR environment
// - Reset the sky
void gltfr::Scene::deinit(Resources& res)
{
  res.m_allocator->destroy(m_sceneFrameInfoBuffer);
  res.m_allocator->destroy(m_sdfTexture);

  destroyDescriptorSet(res.ctx.device);

  m_gltfSceneVk.reset();
  m_gltfSceneRtx.reset();
  m_gltfScene.reset();
  m_hdrEnv.reset();
  m_hdrDome.reset();
  m_sky.reset();
}

//--------------------------------------------------------------------------------------------------
// Position the camera to fit the scene
//
void gltfr::Scene::fitSceneToView() const
{
  if(m_gltfScene)
  {
    auto bbox = m_gltfScene->getSceneBounds();
    CameraManip.fit(bbox.min(), bbox.max(), false, true, CameraManip.getAspectRatio());
  }
}

//--------------------------------------------------------------------------------------------------
// Position the camera to fit the selected object
//
void gltfr::Scene::fitObjectToView() const
{
  if(m_selectedRenderNode >= 0)
  {
    nvh::Bbox worldBbox = getRenderNodeBbox(m_selectedRenderNode);
    CameraManip.fit(worldBbox.min(), worldBbox.max(), false, true, CameraManip.getAspectRatio());
  }
}

//--------------------------------------------------------------------------------------------------
// Select a render node
// - tells the scene graph to select the node
void gltfr::Scene::selectRenderNode(int renderNodeIndex)
{
  m_selectedRenderNode = renderNodeIndex;
  if(m_sceneGraph && m_gltfScene && renderNodeIndex > -1)
  {
    const nvh::gltf::RenderNode& renderNode = m_gltfScene->getRenderNodes()[renderNodeIndex];
    m_sceneGraph->selectNode(renderNode.refNodeID);
  }
  else if(m_sceneGraph)
  {
    m_sceneGraph->selectNode(-1);
  }
}

//--------------------------------------------------------------------------------------------------
// Return the filename of the scene
//
std::string gltfr::Scene::getFilename() const
{
  if(m_gltfScene != nullptr)
    return m_gltfScene->getFilename();
  return "empty";
}

//--------------------------------------------------------------------------------------------------
// Recreating the tangents of the scene
void gltfr::Scene::recreateTangents(bool mikktspace)
{
  if(m_gltfScene && m_gltfScene->valid())
  {
    {
      nvh::ScopedTimer st(std::string("\n") + __FUNCTION__);
      recomputeTangents(m_gltfScene->getModel(), true, mikktspace);
    }
    m_dirtyFlags.set(eVulkanAttributes);
    resetFrameCount();
  }
}

//--------------------------------------------------------------------------------------------------
// Load a scene or HDR environment
//
bool gltfr::Scene::load(Resources& resources, const std::string& filename)
{
  const std::string extension   = std::filesystem::path(filename).extension().string();
  bool              sceneloaded = false;

  if(extension == ".gltf" || extension == ".glb")
  {
    m_gltfScene = std::make_unique<nvh::gltf::Scene>();
    m_sceneGraph.reset();
    m_selectedRenderNode = -1;

    // Loading the scene
    if(m_gltfScene->load(filename))
    {
      sceneloaded  = true;
      m_sceneGraph = std::make_unique<GltfModelUI>(m_gltfScene->getModel(), m_gltfScene->getSceneBounds());
      createVulkanScene(resources);
    }
    else
    {
      m_gltfScene.reset();
      return false;
    }
  }
  else if(extension == ".obj")
  {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = std::filesystem::path(filename).parent_path().string();
    tinyobj::ObjReader reader;

    bool        result = reader.ParseFromFile(filename, reader_config);
    std::string warn   = reader.Warning();
    std::string error  = reader.Error();

    if(result)
    {
      sceneloaded = true;
      TinyConverter   converter;
      tinygltf::Model model;
      converter.convert(model, reader);
      m_gltfScene = std::make_unique<nvh::gltf::Scene>();
      m_gltfScene->takeModel(std::move(model));
      m_sceneGraph = std::make_unique<GltfModelUI>(m_gltfScene->getModel(), m_gltfScene->getSceneBounds());
      createVulkanScene(resources);
    }
    else
    {
      m_gltfScene.reset();
      LOGE("Error loading OBJ: %s\n", error.c_str());
      LOGW("Warning: %s\n", warn.c_str());
      return false;
    }
  }
  else if(extension == ".hdr")
  {
    createHdr(resources, filename);
    sceneloaded = false;
  }

  if(sceneloaded)
  {
    postSceneCreateProcess(resources, filename);
  }

  resetFrameCount();
  return true;
}

//--------------------------------------------------------------------------------------------------
// After the scene is loaded, we need to create the descriptor set and write the information
// - This is done after the scene is loaded, and the camera is fitted
//
void gltfr::Scene::postSceneCreateProcess(Resources& resources, const std::string& filename)
{
  if(filename.empty())
    return;

  setDirtyFlag(Scene::eNewScene, true);

  writeDescriptorSet(resources);

  // Scene camera fitting
  nvh::Bbox                                   bbox    = m_gltfScene->getSceneBounds();
  const std::vector<nvh::gltf::RenderCamera>& cameras = m_gltfScene->getRenderCameras();
  nvvkhl::setCamera(filename, cameras, bbox);   // Camera auto-scene-fitting
  g_elemCamera->setSceneRadius(bbox.radius());  // Navigation help
}

//--------------------------------------------------------------------------------------------------
// Save the scene
//
bool gltfr::Scene::save(const std::string& filename) const
{
  if(m_gltfScene && m_gltfScene->valid() && !filename.empty())
  {
    // First, copy the camera
    nvh::gltf::RenderCamera camera;
    CameraManip.getLookat(camera.eye, camera.center, camera.up);
    camera.yfov  = glm::radians(CameraManip.getFov());
    camera.znear = CameraManip.getClipPlanes().x;
    camera.zfar  = CameraManip.getClipPlanes().y;
    m_gltfScene->setSceneCamera(camera);
    // Saving the scene
    return m_gltfScene->save(filename);
  }
  return false;
}

void gltfr::Scene::createDescriptorPool(VkDevice device)
{
  const std::vector<VkDescriptorPoolSize> poolSizes{
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAXTEXTURES},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
  };

  const VkDescriptorPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |  //  allows descriptor sets to be updated after they have been bound to a command buffer
               VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  // individual descriptor sets can be freed from the descriptor pool
      .maxSets       = MAXTEXTURES,  // Allowing to create many sets (ImGui uses this for textures)
      .poolSizeCount = uint32_t(poolSizes.size()),
      .pPoolSizes    = poolSizes.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool));
  // nvvk::DebugUtil(device).DBG_NAME(m_descriptorPool));
}

//--------------------------------------------------------------------------------------------------
// Create the descriptor set for the scene
//
void gltfr::Scene::createDescriptorSet(VkDevice device)
{
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
  layoutBindings.push_back({.binding         = SceneBindings::eFrameInfo,
                            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags      = VK_SHADER_STAGE_ALL});
  layoutBindings.push_back({.binding         = SceneBindings::eSceneDesc,
                            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags      = VK_SHADER_STAGE_ALL});
  layoutBindings.push_back({.binding         = SceneBindings::eTextures,
                            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .descriptorCount = MAXTEXTURES,  // Not all will be filled - but pipeline will be cached
                            .stageFlags      = VK_SHADER_STAGE_ALL});
  layoutBindings.push_back({.binding = SceneBindings::eSdfTexture,
                            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .descriptorCount = 1,
                            .stageFlags = VK_SHADER_STAGE_ALL});

  const VkDescriptorBindingFlags flags[] = {
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, // Flags for binding 0 (uniform buffer)
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, // Flags for binding 1 (storage buffer)
      // Flags for binding 2 (texture array):
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |               // Can update while in use
          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT | // Can update unused entries
          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,              // Not all array elements need to be valid (0,2,3 vs 0,1,2,3)

      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
    };
  const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount  = uint32_t(layoutBindings.size()),  // matches our number of bindings
      .pBindingFlags = flags,                            // the flags for each binding
  };

  const VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &bindingFlags,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,  // Allows to update the descriptor set after it has been bound
      .bindingCount = uint32_t(layoutBindings.size()),
      .pBindings    = layoutBindings.data(),
  };
  NVVK_CHECK(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, nullptr, &m_sceneDescriptorSetLayout));
  nvvk::DebugUtil(device).DBG_NAME(m_sceneDescriptorSetLayout);

  // Allocate the descriptor set, needed only for larger descriptor sets
  const VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_sceneDescriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &m_sceneDescriptorSet));
  nvvk::DebugUtil(device).DBG_NAME(m_sceneDescriptorSet);
}

//--------------------------------------------------------------------------------------------------
// Write the descriptor set for the scene
//
void gltfr::Scene::writeDescriptorSet(Resources& resources) const
{
  if(!m_gltfScene->valid())
  {
    return;
  }

  // Write to descriptors
  const VkDescriptorBufferInfo frameBufferInfo{m_sceneFrameInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo sceneBufferInfo{m_gltfSceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.push_back({.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet          = m_sceneDescriptorSet,
                                 .dstBinding      = SceneBindings::eFrameInfo,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                 .pBufferInfo     = &frameBufferInfo});
  writeDescriptorSets.push_back({.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet          = m_sceneDescriptorSet,
                                 .dstBinding      = SceneBindings::eSceneDesc,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 .pBufferInfo     = &sceneBufferInfo});

  std::vector<VkDescriptorImageInfo> descImageInfos;
  descImageInfos.reserve(m_gltfSceneVk->nbTextures());
  for(const nvvk::Texture& texture : m_gltfSceneVk->textures())  // All texture samplers
  {
    descImageInfos.emplace_back(texture.descriptor);
  }
  writeDescriptorSets.push_back({.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet          = m_sceneDescriptorSet,
                                 .dstBinding      = SceneBindings::eTextures,
                                 .dstArrayElement = 0,
                                 .descriptorCount = m_gltfSceneVk->nbTextures(),
                                 .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 .pImageInfo      = descImageInfos.data()});

  if (m_sdfTexture.image != VK_NULL_HANDLE)
  {
    VkWriteDescriptorSet sdfWrite = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = m_sceneDescriptorSet,
        .dstBinding = SceneBindings::eSdfTexture,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &m_sdfTexture.descriptor};
    writeDescriptorSets.push_back(sdfWrite);
  }

  vkUpdateDescriptorSets(resources.ctx.device, static_cast<uint32_t>(writeDescriptorSets.size()),
                         writeDescriptorSets.data(), 0, nullptr);
}

void gltfr::Scene::destroyDescriptorSet(VkDevice device)
{
  if(m_descriptorPool)
  {
    vkFreeDescriptorSets(device, m_descriptorPool, 1, &m_sceneDescriptorSet);
    vkDestroyDescriptorSetLayout(device, m_sceneDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Processing the frame is something call at each frame
// If something has changed, we need to update one of the following:
// - Update the animation
// - Update the camera
// - Update the environment
// - Update the Vulkan scene
// - Update the RTX scene
//
bool gltfr::Scene::processFrame(VkCommandBuffer cmd, Settings& settings)
{
  // Dealing with animation
  if(m_gltfScene->hasAnimation() && m_animControl.doAnimation())
  {
    float                     deltaTime = m_animControl.deltaTime();
    nvh::gltf::AnimationInfo& animInfo  = m_gltfScene->getAnimationInfo(m_animControl.currentAnimation);
    if(m_animControl.isReset())
    {
      animInfo.reset();
    }
    else
    {
      animInfo.incrementTime(deltaTime);
    }

    m_gltfScene->updateAnimation(m_animControl.currentAnimation);
    m_gltfScene->updateRenderNodes();

    m_animControl.clearStates();

    {
      m_dirtyFlags.set(eVulkanScene);
      m_dirtyFlags.set(eRtxScene);
      resetFrameCount();
    }
  }

  // Increase the frame count and return if we reached the maximum
  if(!updateFrameCount(settings))
    return false;

  // Check for scene changes
  if(m_dirtyFlags.test(eVulkanScene))
  {
    m_gltfSceneVk->updateRenderNodesBuffer(cmd, *m_gltfScene);       // Animation, changing nodes transform
    m_gltfSceneVk->updateRenderPrimitivesBuffer(cmd, *m_gltfScene);  // Animation
    m_gltfSceneVk->updateRenderLightsBuffer(cmd, *m_gltfScene);      // changing lights data
    m_dirtyFlags.reset(eVulkanScene);
  }
  if(m_dirtyFlags.test(eVulkanMaterial))
  {
    m_gltfSceneVk->updateMaterialBuffer(cmd, *m_gltfScene);
    m_dirtyFlags.reset(eVulkanMaterial);
  }
  if(m_dirtyFlags.test(eVulkanAttributes))
  {
    m_gltfSceneVk->updateVertexBuffers(cmd, *m_gltfScene);
    m_dirtyFlags.reset(eVulkanAttributes);
  }
  if(m_dirtyFlags.test(eRtxScene))
  {
    m_gltfSceneRtx->updateTopLevelAS(cmd, *m_gltfScene);
    m_gltfSceneRtx->updateBottomLevelAS(cmd, *m_gltfScene);

    m_dirtyFlags.reset(eRtxScene);
  }

  // Update the camera
  const glm::vec2& clip       = CameraManip.getClipPlanes();
  m_sceneFrameInfo.viewMatrix = CameraManip.getMatrix();
  m_sceneFrameInfo.projMatrix =
      glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), CameraManip.getAspectRatio(), clip.x, clip.y);
  m_sceneFrameInfo.projMatrix[1][1] *= -1;
  m_sceneFrameInfo.projMatrixI = glm::inverse(m_sceneFrameInfo.projMatrix);
  m_sceneFrameInfo.viewMatrixI = glm::inverse(m_sceneFrameInfo.viewMatrix);
  m_sceneFrameInfo.camPos      = CameraManip.getEye();

  // Update the environment
  m_sceneFrameInfo.envIntensity = glm::vec4(settings.hdrEnvIntensity, settings.hdrEnvIntensity, settings.hdrEnvIntensity, 1.0F);
  m_sceneFrameInfo.envRotation = settings.hdrEnvRotation;
  m_sceneFrameInfo.envBlur     = settings.hdrBlur;
  m_sceneFrameInfo.flags       = 0;
  if(settings.envSystem == Settings::eSky)
  {
    SET_FLAG(m_sceneFrameInfo.flags, USE_SKY_FLAG);
    m_sceneFrameInfo.nbLights = 1;  //static_cast<int>(settings.lights.size());
    m_sceneFrameInfo.light[0] = m_sky->getSun();
  }
  else
  {
    SET_FLAG(m_sceneFrameInfo.flags, USE_HDR_FLAG);
    m_sceneFrameInfo.nbLights = 0;
  }
  if(settings.useSolidBackground)
  {
    SET_FLAG(m_sceneFrameInfo.flags, USE_SOLID_BACKGROUND_FLAG);
    m_sceneFrameInfo.backgroundColor = nvvkhl_shaders::toLinear(settings.solidBackgroundColor);
  }


  vkCmdUpdateBuffer(cmd, m_sceneFrameInfoBuffer.buffer, 0, sizeof(DH::SceneFrameInfo), &m_sceneFrameInfo);

  // Barrier to ensure the buffer is updated before rendering
  std::array<VkBufferMemoryBarrier2, 1> bufferBarriers = {
      {{.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,  // vkCmdUpdateBuffer uses transfer
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .buffer        = m_sceneFrameInfoBuffer.buffer,
        .offset        = 0,
        .size          = sizeof(DH::SceneFrameInfo)}},
  };

  VkDependencyInfo dependencyInfo = {.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                     .bufferMemoryBarrierCount = uint32_t(bufferBarriers.size()),
                                     .pBufferMemoryBarriers    = bufferBarriers.data()};
  vkCmdPipelineBarrier2(cmd, &dependencyInfo);


  // Update the sky
  m_sky->skyParams().yIsUp = CameraManip.getUp().y > CameraManip.getUp().z;
  m_sky->updateParameterBuffer(cmd);

  return true;
}

void embreeErrorFunc(void *userPtr, RTCError code, const char *str)
{
  std::cerr << "Embree Error: " << code << ", " << str << std::endl;
}

nvvk::Buffer gltfr::Scene::generateSdf(Resources &res, VkCommandBuffer cmd)
{
  nvh::ScopedTimer st("SDF Generation from glTF");

  // 1. 初始化 Embree
  RTCDevice embreeDevice = rtcNewDevice(nullptr);
  RTCScene embreeScene = rtcNewScene(embreeDevice);
  rtcSetDeviceErrorFunction(embreeDevice, embreeErrorFunc, nullptr);

  // ==================================================================
  // 从 glTF 动态加载几何体
  // ==================================================================
  std::cout << "\n--- Loading geometry from glTF scene into Embree ---" << std::endl;

  const auto &model = m_gltfScene->getModel();

  // 我们遍历所有“渲染节点”，因为它们包含了计算好的世界变换矩阵
  const auto &renderNodes = m_gltfScene->getRenderNodes();

  for (size_t i = 0; i < renderNodes.size(); ++i)
  {
    const auto &node = renderNodes[i];
    // ==================== 最终修正 ====================
    // 1. 从 RenderNode 获取 RenderPrimitive 对象
    const auto &renderPrim = m_gltfScene->getRenderPrimitive(node.renderPrimID);

    // 2. 直接从 RenderPrimitive 解引用指针，得到原始的 primitive 对象
    const auto &primitive = *renderPrim.pPrimitive;
    // ===============================================

    // --- a. 提取顶点位置数据 (本地空间) ---
    if (primitive.attributes.find("POSITION") == primitive.attributes.end())
      continue;

    const auto &posAccessor = model.accessors[primitive.attributes.at("POSITION")];
    const auto &posBufferView = model.bufferViews[posAccessor.bufferView];
    const auto &posBuffer = model.buffers[posBufferView.buffer];
    const float *vertexBuffer = reinterpret_cast<const float *>(
        &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);
    const int numVertices = static_cast<int>(posAccessor.count);
    const int vertexStride = posAccessor.ByteStride(posBufferView) / sizeof(float); // 通常是 3

    // --- b. 提取索引数据 ---
    const auto &indexAccessor = model.accessors[primitive.indices];
    const auto &indexBufferView = model.bufferViews[indexAccessor.bufferView];
    const auto &indexBuffer = model.buffers[indexBufferView.buffer];
    const void *indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
    const int numIndices = static_cast<int>(indexAccessor.count);

    // --- c. 转换索引为 32-bit (Embree需要) ---
    // GLTF 索引可能是 8-bit, 16-bit, 或 32-bit，我们统一转成 32-bit
    std::vector<uint32_t> convertedIndices(numIndices);
    switch (indexAccessor.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      for (int k = 0; k < numIndices; ++k)
        convertedIndices[k] = reinterpret_cast<const uint8_t *>(indexData)[k];
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      for (int k = 0; k < numIndices; ++k)
        convertedIndices[k] = reinterpret_cast<const uint16_t *>(indexData)[k];
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      memcpy(convertedIndices.data(), indexData, numIndices * sizeof(uint32_t));
      break;
    default:
      continue; // 不支持的索引类型
    }

    // --- d. 转换顶点到世界空间 ---
    // 获取节点的变换矩阵
    const glm::mat4 worldMatrix = node.worldMatrix;
    std::vector<glm::vec3> worldSpaceVertices(numVertices);
    for (int v = 0; v < numVertices; ++v)
    {
      // 从原始缓冲区读取本地坐标
      glm::vec3 localPos(vertexBuffer[v * vertexStride], vertexBuffer[v * vertexStride + 1], vertexBuffer[v * vertexStride + 2]);
      // 应用变换
      worldSpaceVertices[v] = glm::vec3(worldMatrix * glm::vec4(localPos, 1.0f));
    }

    // --- e. 创建并填充 Embree 几何体 ---
    RTCGeometry embreeGeom = rtcNewGeometry(embreeDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

    // 使用 rtcSetNewGeometryBuffer，因为我们创建了临时的新缓冲区
    // Embree 会自己管理这份数据的内存
    float *embreeVerts = (float *)rtcSetNewGeometryBuffer(embreeGeom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(glm::vec3), numVertices);
    memcpy(embreeVerts, worldSpaceVertices.data(), numVertices * sizeof(glm::vec3));

    unsigned int *embreeIndices = (unsigned int *)rtcSetNewGeometryBuffer(embreeGeom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(uint32_t) * 3, numIndices / 3);
    memcpy(embreeIndices, convertedIndices.data(), numIndices * sizeof(uint32_t));

    rtcCommitGeometry(embreeGeom);
    // 使用 renderNode 的索引 'i' 作为几何体 ID
    rtcAttachGeometryByID(embreeScene, embreeGeom, static_cast<unsigned int>(i));
    rtcReleaseGeometry(embreeGeom); // 添加后可以立即释放
  }

  rtcCommitScene(embreeScene);
  std::cout << "--- glTF scene committed to Embree ---" << std::endl;

  const int numSignSamples = 64;
  const std::vector<glm::vec3> sampleDirections = generateUniformSphereSamples(numSignSamples);

  nvh::Bbox sceneBbox = m_gltfScene->getSceneBounds();
  const int resolution = 64; // 可以用一个更高的分辨率
  std::vector<float> sdfData(resolution * resolution * resolution);
  glm::vec3 step = sceneBbox.extents() / float(resolution - 1);

  const uint64_t totalVoxels = static_cast<uint64_t>(resolution) * resolution * resolution;

  // 遍历体素 (并行版本)
  std::cout << "--- Starting SDF Generation with Sign Calculation (Parallel) ---" << std::endl;

  // 使用正确的 API：nvh::parallel_batches
  // BATCHSIZE 是一个性能调优参数，默认的 512 对于这种计算密集型任务是个不错的选择。
  // 你也可以像这样指定它: nvh::parallel_batches<256>(...)
  nvh::parallel_batches(totalVoxels, [&](uint64_t linear_index)
                        {
        // 在任务内部，根据一维索引反算三维坐标
        const int z = static_cast<int>(linear_index / (resolution * resolution));
        const int y = static_cast<int>((linear_index / resolution) % resolution);
        const int x = static_cast<int>(linear_index % resolution);
  
        glm::vec3 queryPos = sceneBbox.min() + glm::vec3(x, y, z) * step;
  
        // --- 阶段1: 查询无符号距离 ---
        float closestDistanceSq = std::numeric_limits<float>::max();
        PointQueryUserData userData = { embreeScene, &closestDistanceSq };
        RTCPointQueryContext context;
        rtcInitPointQueryContext(&context);
        RTCPointQuery query;
        query.x = queryPos.x;
        query.y = queryPos.y;
        query.z = queryPos.z;
        query.radius = std::numeric_limits<float>::max();
        query.time = 0.f;
        rtcPointQuery(embreeScene, &query, &context, rtcPointQueryCallback, &userData);
  
        if (closestDistanceSq >= std::numeric_limits<float>::max())
        {
            sdfData[linear_index] = sceneBbox.extents().x;
            return; 
        }
  
        float unsigned_distance = std::sqrt(closestDistanceSq);
        float final_sdf_distance = unsigned_distance;
  
        const float epsilon = 1e-5f;
  
        // --- 阶段2: 通过光线投射判断符号 ---
        if (unsigned_distance > epsilon)
        {
            int hitCount = 0;
            int hitBackCount = 0;
  
            for (const auto& dir : sampleDirections)
            {
                RTCRayHit rayhit;
                rayhit.ray.org_x = queryPos.x;
                rayhit.ray.org_y = queryPos.y;
                rayhit.ray.org_z = queryPos.z;
                rayhit.ray.dir_x = dir.x;
                rayhit.ray.dir_y = dir.y;
                rayhit.ray.dir_z = dir.z;
                rayhit.ray.tnear = 0.0f;
                rayhit.ray.tfar = std::numeric_limits<float>::max();
                rayhit.ray.mask = -1;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
  
                rtcIntersect1(embreeScene, &rayhit);
  
                if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
                {
                    hitCount++;
                    glm::vec3 hitNormal = glm::normalize(glm::vec3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z));
                    if (glm::dot(dir, hitNormal) > 0)
                    {
                        hitBackCount++;
                    }
                }
            }
  
            if (hitCount > 0 && (static_cast<float>(hitBackCount) / hitCount) > 0.5f)
            {
                final_sdf_distance *= -1.0f;
            }
        }
        else
        {
            final_sdf_distance = 0.0f;
        }
  
        // 将最终计算好的值存入数组
        sdfData[linear_index] = final_sdf_distance; });

  std::cout << "--- SDF Generation Finished ---" << std::endl;

  // ==================================================================
  // 1. 创建目标3D纹理 (GPU-Only)
  // ==================================================================
  VkDevice device = res.ctx.device;
  nvvk::ResourceAllocator *alloc = res.m_allocator.get();

  // 如果旧纹理存在，先销毁
  if (m_sdfTexture.image != VK_NULL_HANDLE)
  {
    alloc->destroy(m_sdfTexture);
  }

  VkExtent3D extent = {(uint32_t)resolution, (uint32_t)resolution, (uint32_t)resolution};
  VkFormat format = VK_FORMAT_R32_SFLOAT; // 单通道32位浮点格式

  VkImageCreateInfo imageCI = nvvk::makeImage3DCreateInfo(extent, format,
                                                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);

  // 使用默认的 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT 创建图像
  nvvk::Image image = alloc->createImage(imageCI);
  VkImageViewCreateInfo viewCI = nvvk::makeImageViewCreateInfo(image.image, imageCI);
  VkSamplerCreateInfo samplerCI = nvvk::makeSamplerCreateInfo();

  m_sdfTexture = alloc->createTexture(image, viewCI, samplerCI);
  m_sdfTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  // 为新创建的纹理命名，便于调试
  nvvk::DebugUtil(device).DBG_NAME(m_sdfTexture.image);

  // ==================================================================
  // 2. 创建暂存缓冲区 (Staging Buffer) 并上传数据
  // ==================================================================
  VkDeviceSize bufferSize = sdfData.size() * sizeof(float);

  // 使用简单的 createBuffer 签名创建一个CPU可见的缓冲区
  nvvk::Buffer stagingBuffer = alloc->createBuffer(bufferSize,
                                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  nvvk::DebugUtil(device).DBG_NAME(stagingBuffer.buffer, "SdfStagingBuffer"); // 给它一个有意义的名字

  // 将SDF数据拷贝到暂存缓冲区
  void *mappedData = alloc->map(stagingBuffer);
  memcpy(mappedData, sdfData.data(), bufferSize);
  alloc->unmap(stagingBuffer);

  // ==================================================================
  // 3. 记录并执行拷贝命令
  // ==================================================================
  // 转换图像布局，准备接收数据
  nvvk::cmdBarrierImageLayout(cmd, m_sdfTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  VkBufferImageCopy region = {};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = extent;
  vkCmdCopyBufferToImage(cmd, stagingBuffer.buffer, m_sdfTexture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  nvvk::cmdBarrierImageLayout(cmd, m_sdfTexture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  std::cout << "--- SDF data uploaded to GPU as a 3D texture ---" << std::endl;

  // 清理Embree资源
  rtcReleaseScene(embreeScene);
  rtcReleaseDevice(embreeDevice);

  return stagingBuffer;
}

//--------------------------------------------------------------------------------------------------
// Create the Vulkan scene representation
// This means that the glTF scene is converted into buffers and acceleration structures
// The sceneVk is the Vulkan representation of the scene
// - Materials
// - Textures
// - RenderNodes and RenderPrimitives
// The sceneRtx is the Vulkan representation of the scene for ray tracing
// - Bottom-level acceleration structures
// - Top-level acceleration structure
void gltfr::Scene::createVulkanScene(Resources& res)
{
  nvh::ScopedTimer st(std::string("\n") + __FUNCTION__);

  nvvk::ResourceAllocator* alloc = res.m_allocator.get();

  m_gltfSceneVk = std::make_unique<nvvkhl::SceneVk>(res.ctx.device, res.ctx.physicalDevice, alloc);
  m_gltfSceneRtx = std::make_unique<nvvkhl::SceneRtx>(res.ctx.device, res.ctx.physicalDevice, alloc, res.ctx.compute.familyIndex);

  if(m_gltfScene->valid())
  {
    // Create the Vulkan side of the scene
    // Since we load and display simultaneously, we need to use a second GTC queue
    nvvk::CommandPool cmd_pool(res.ctx.device, res.ctx.compute.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                               res.ctx.compute.queue);
    VkCommandBuffer   cmd;
    {  // Creating the scene in Vulkan buffers
      cmd = cmd_pool.createCommandBuffer();
      m_gltfSceneVk->create(cmd, *m_gltfScene, false);

      nvvk::Buffer sdfStagingBuffer = generateSdf(res, cmd);

      // This method is simpler, but it is not as efficient as the while-loop below
      // m_sceneRtx->create(cmd, *m_scene, *m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
      cmd_pool.submitAndWait(cmd);
      alloc->destroy(sdfStagingBuffer);
      res.m_allocator->finalizeAndReleaseStaging();  // Make sure there are no pending staging buffers and clear them up
    }

    // Create the acceleration structure, and compact the BLAS
    VkBuildAccelerationStructureFlagsKHR blasBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                          | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR
                                                          | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    m_gltfSceneRtx->createBottomLevelAccelerationStructure(*m_gltfScene, *m_gltfSceneVk, blasBuildFlags);
    bool finished = false;
    do
    {
      {  // Building the BLAS
        cmd      = cmd_pool.createCommandBuffer();
        finished = m_gltfSceneRtx->cmdBuildBottomLevelAccelerationStructure(cmd, 512'000'000);
        cmd_pool.submitAndWait(cmd);
      }
      {  // Compacting the BLAS
        cmd = cmd_pool.createCommandBuffer();
        m_gltfSceneRtx->cmdCompactBlas(cmd);
        cmd_pool.submitAndWait(cmd);
      }
      m_gltfSceneRtx->destroyNonCompactedBlas();
    } while(!finished);

    {  // Creating the top-level acceleration structure
      cmd = cmd_pool.createCommandBuffer();
      m_gltfSceneRtx->cmdCreateBuildTopLevelAccelerationStructure(cmd, *m_gltfScene);
      cmd_pool.submitAndWait(cmd);
    }
    if(!m_gltfScene->hasAnimation())
    {
      m_gltfSceneRtx->destroyScratchBuffers();
    }
  }
  else
  {
    m_gltfSceneRtx.reset();
    m_gltfSceneVk.reset();
  }
}

//--------------------------------------------------------------------------------------------------
// Create the HDR environment
//
void gltfr::Scene::createHdr(Resources& res, const std::string& filename)
{
  nvh::ScopedTimer st(std::string("\n") + __FUNCTION__);

  nvvk::ResourceAllocator* alloc          = res.m_allocator.get();
  const uint32_t           c_family_queue = res.ctx.compute.familyIndex;

  m_hdrEnv  = std::make_unique<nvvkhl::HdrEnv>(res.ctx.device, res.ctx.physicalDevice, alloc, c_family_queue);
  m_hdrDome = std::make_unique<nvvkhl::HdrEnvDome>(res.ctx.device, res.ctx.physicalDevice, alloc, c_family_queue);
  m_hdrEnv->loadEnvironment(filename, true);
  m_hdrDome->create(m_hdrEnv->getDescriptorSet(), m_hdrEnv->getDescriptorSetLayout());
  alloc->finalizeAndReleaseStaging();

  m_hdrFilename = std::filesystem::path(filename).filename().string();
  setDirtyFlag(Scene::eHdrEnv, true);
}

void gltfr::Scene::generateHdrMipmap(VkCommandBuffer cmd, Resources& res)
{
  vkQueueWaitIdle(res.ctx.GCT0.queue);
  nvvk::cmdGenerateMipmaps(cmd, m_hdrEnv->getHdrTexture().image, VK_FORMAT_R32G32B32A32_SFLOAT,
                           m_hdrEnv->getHdrImageSize(), nvvk::mipLevels(m_hdrEnv->getHdrImageSize()));
}

//--------------------------------------------------------------------------------------------------
// Update the frame counter only if the camera has NOT changed
// otherwise, reset the frame counter
//
bool gltfr::Scene::updateFrameCount(Settings& settings)
{
  static glm::mat4 ref_cam_matrix;
  static float     ref_fov{CameraManip.getFov()};

  const glm::mat4& m   = CameraManip.getMatrix();
  const float      fov = CameraManip.getFov();

  if(ref_cam_matrix != m || ref_fov != fov)
  {
    resetFrameCount();
    ref_cam_matrix = m;
    ref_fov        = fov;
  }

  if(m_sceneFrameInfo.frameCount >= settings.maxFrames)
  {
    return false;
  }
  m_sceneFrameInfo.frameCount++;
  return true;
}

//--------------------------------------------------------------------------------------------------
// Reset the frame counter
void gltfr::Scene::resetFrameCount()
{
  m_sceneFrameInfo.frameCount = -1;
}

nvh::Bbox gltfr::Scene::getRenderNodeBbox(int nodeID) const
{
  nvh::Bbox worldBbox({-1, -1, -1}, {1, 1, 1});
  if(nodeID < 0)
    return worldBbox;

  const nvh::gltf::RenderNode&      renderNode      = m_gltfScene->getRenderNodes()[nodeID];
  const nvh::gltf::RenderPrimitive& renderPrimitive = m_gltfScene->getRenderPrimitive(renderNode.renderPrimID);
  const tinygltf::Model&            model           = m_gltfScene->getModel();
  const tinygltf::Accessor&         accessor = model.accessors[renderPrimitive.pPrimitive->attributes.at("POSITION")];

  glm::vec3 minValues = {-1.f, -1.f, -1.f};
  glm::vec3 maxValues = {1.f, 1.f, 1.f};
  if(!accessor.minValues.empty())
    minValues = glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]);
  if(!accessor.maxValues.empty())
    maxValues = glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]);
  nvh::Bbox objBbox(minValues, maxValues);
  worldBbox = objBbox.transform(renderNode.worldMatrix);

  return worldBbox;
}

//--------------------------------------------------------------------------------------------------
// Rendering the UI of the scene
// - Environment
//  - Sky
//  - HDR
// - Scene
//   - Multiple Scenes
//   - Variants
//   - Animation
//   - Scene Graph
//   - Statistics
//
bool gltfr::Scene::onUI(Resources& resources, Settings& settings, GLFWwindow* winHandle)
{
  auto& headerManager = CollapsingHeaderManager::getInstance();

  bool reset = false;

  if(headerManager.beginHeader("Environment"))
  {
    const bool          skyOnly          = !(m_hdrEnv && m_hdrEnv->isValid());
    Settings::EnvSystem cache_env_system = settings.envSystem;
    reset |= ImGui::RadioButton("Sky", reinterpret_cast<int*>(&settings.envSystem), Settings::eSky);
    ImGui::SameLine();
    ImGui::BeginDisabled(skyOnly);
    reset |= ImGui::RadioButton("Hdr", reinterpret_cast<int*>(&settings.envSystem), Settings::eHdr);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if(ImGui::SmallButton("Load##env"))
    {
      std::string filename = NVPSystem::windowOpenFileDialog(winHandle, "Load HDR", "HDR(.hdr)|*.hdr");
      if(!filename.empty())
      {
        vkDeviceWaitIdle(resources.ctx.device);
        createHdr(resources, filename);
        settings.envSystem = Settings::eHdr;
        reset              = true;
      }
    }

    // When switching the environment, reset Firefly max luminance
    if(cache_env_system != settings.envSystem && m_hdrEnv)
    {
      settings.setDefaultLuminance(m_hdrEnv->getIntegral());
    }

    PE::begin();
    if(settings.envSystem == Settings::eSky)
    {
      reset |= m_sky->onUI();
    }
    else  // HDR
    {
      PE::Text("HDR File", m_hdrFilename);
      reset |= PE::SliderFloat("Intensity", &settings.hdrEnvIntensity, 0, 100, "%.3f", ImGuiSliderFlags_Logarithmic, "HDR intensity");
      reset |= PE::SliderAngle("Rotation", &settings.hdrEnvRotation, -360, 360, "%.0f deg", 0, "Rotating the environment");
      reset |= PE::SliderFloat("Blur", &settings.hdrBlur, 0, 1, "%.3f", 0, "Blur the environment");
    }
    PE::end();
    PE::begin();
    reset |= PE::Checkbox("Use Solid Background", &settings.useSolidBackground);
    if(settings.useSolidBackground)
    {
      reset |= PE::ColorEdit3("Background Color", glm::value_ptr(settings.solidBackgroundColor),
                              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_Float);
    }
    PE::end();
  }

  if(m_gltfScene && m_gltfScene->valid())
  {
    // Multiple scenes
    if(m_gltfScene->getModel().scenes.size() > 1)
    {
      if(headerManager.beginHeader("Multiple Scenes"))
      {
        ImGui::PushID("Scenes");
        for(size_t i = 0; i < m_gltfScene->getModel().scenes.size(); i++)
        {
          if(ImGui::RadioButton(m_gltfScene->getModel().scenes[i].name.c_str(), m_gltfScene->getCurrentScene() == i))
          {
            m_gltfScene->setCurrentScene(int(i));
            vkDeviceWaitIdle(resources.ctx.device);
            createVulkanScene(resources);
            postSceneCreateProcess(resources, m_gltfScene->getFilename());
            reset = true;
            setDirtyFlag(Scene::eNewScene, true);
          }
        }
        ImGui::PopID();
      }
    }

    // Variant selection
    if(m_gltfScene->getVariants().size() > 0)
    {
      if(headerManager.beginHeader("Variants"))
      {
        ImGui::PushID("Variants");
        for(size_t i = 0; i < m_gltfScene->getVariants().size(); i++)
        {
          if(ImGui::Selectable(m_gltfScene->getVariants()[i].c_str(), m_gltfScene->getCurrentVariant() == i))
          {
            m_gltfScene->setCurrentVariant(int(i));
            m_dirtyFlags.set(eVulkanScene);
            reset = true;
          }
        }
        ImGui::PopID();
      }
    }

    // Animation
    if(m_gltfScene->hasAnimation())
    {
      if(headerManager.beginHeader("Animation"))
      {
        m_animControl.onUI(m_gltfScene.get());
      }
    }


    if(m_sceneGraph && headerManager.beginHeader("Scene Graph"))
    {
      int selectedNode = m_sceneGraph->selectedNode();
      m_sceneGraph->render();

      // Find the `render node` corresponding to the selected node
      // The `render node` is the node that is rendered, and different from the `scene node`
      if(m_sceneGraph->selectedNode() > -1 && selectedNode != m_selectedRenderNode)
      {
        selectedNode      = m_sceneGraph->selectedNode();
        auto& renderNodes = m_gltfScene->getRenderNodes();
        for(size_t i = 0; i < renderNodes.size(); i++)
        {
          if(renderNodes[i].refNodeID == selectedNode)
          {
            m_selectedRenderNode = int(i);
            break;
          }
        }
      }
      else if(selectedNode == -1)
      {
        m_selectedRenderNode = -1;  // No node selected
      }

      // Check for scene graph changes
      bool transformChanged  = m_sceneGraph->hasTransformChanged();
      bool lightChanged      = m_sceneGraph->hasLightChanged();
      bool visibilityChanged = m_sceneGraph->hasVisibilityChanged();
      bool materialChanged   = m_sceneGraph->hasMaterialChanged();

      if(m_sceneGraph->hasMaterialFlagChanges())
      {
        m_dirtyFlags.set(eRtxScene);
        reset = true;
      }

      if(transformChanged || lightChanged || visibilityChanged)
      {
        m_dirtyFlags.set(eVulkanScene);
        m_dirtyFlags.set(eRtxScene);

        if(visibilityChanged)
          m_dirtyFlags.set(eNodeVisibility);

        m_gltfScene->updateRenderNodes();
        reset = true;
      }

      if(materialChanged)
      {
        m_dirtyFlags.set(eVulkanMaterial);
        reset = true;
      }

      m_sceneGraph->resetChanges();
    }


    if(headerManager.beginHeader("Statistics"))
    {
      const tinygltf::Model& tiny = m_gltfScene->getModel();
      PE::begin("Stat_Val");
      PE::Text("Nodes", std::to_string(tiny.nodes.size()));
      PE::Text("Render Nodes", std::to_string(m_gltfScene->getRenderNodes().size()));
      PE::Text("Render Primitives", std::to_string(m_gltfScene->getNumRenderPrimitives()));
      PE::Text("Materials", std::to_string(tiny.materials.size()));
      PE::Text("Triangles", std::to_string(m_gltfScene->getNumTriangles()));
      PE::Text("Lights", std::to_string(tiny.lights.size()));
      PE::Text("Textures", std::to_string(tiny.textures.size()));
      PE::Text("Images", std::to_string(tiny.images.size()));
      PE::end();
    }


    if(reset)
    {
      resetFrameCount();
    }
  }

  return reset;
}
