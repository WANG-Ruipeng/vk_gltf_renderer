/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*

This is the settings of the application.
It allow to control which renderer to use, the environment system, and the intensity of the environment.

*/

// Device/host structures for the scene
#include "nvvkhl/shaders/dh_lighting.h"
namespace DH {
#include "shaders/device_host.h"  // Include the device/host structures
}

namespace gltfr {

// Settings for the renderer
struct Settings
{
  enum EnvSystem
  {
    eSky,
    eHdr,
  };

  // Here: add all renderers and their name
  enum RenderSystem
  {
    ePathtracer,
    eRaster,
    eSdf,
  };
  static constexpr const char* rendererNames[] = {"Pathtracer", "Raster", "SDF"};


  int          maxFrames            = 200000;       // Maximum number of frames to render (used by pathtracer)
  bool         showAxis             = true;         // Show the axis (bottom left)
  EnvSystem    envSystem            = eSky;         // Environment system: Sky or HDR
  RenderSystem renderSystem         = ePathtracer;  // Renderer to use
  float        hdrEnvIntensity      = 1.0f;         // Intensity of the environment (HDR)
  float        hdrEnvRotation       = 0.0f;         // Rotation of the environment (HDR)
  float        hdrBlur              = 0.0f;         // Blur of the environment (HDR)
  float        maxLuminance         = 10.0f;        // For firefly
  glm::vec3    silhouetteColor      = {0.6f, 0.4f, 0.0f};
  bool         useSolidBackground   = false;
  glm::vec3    solidBackgroundColor = {0.0f, 0.0f, 0.0f};

  void onUI();
  void setDefaultLuminance(float hdrEnvIntensity);
};


enum RenderMode
{
  eRTX,
  eIndirect,
};

struct PathtraceSettings
{
  int              maxDepth{50};
  int              maxSamples{1};
  DH::EDebugMethod dbgMethod = DH::eDbgMethod_none;
  RenderMode       renderMode{eIndirect};  // RTX / Indirect
  float            aperture{0.0f};
  float            focalDistance{10.0f};
  bool             autoFocus{true};
};


}  // namespace gltfr