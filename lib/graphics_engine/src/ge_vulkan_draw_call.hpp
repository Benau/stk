#ifndef HEADER_GE_VULKAN_DRAW_CALL_HPP
#define HEADER_GE_VULKAN_DRAW_CALL_HPP

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vulkan_wrapper.h"

#include "matrix4.h"
#include "vector3d.h"
#include "ESceneNodeTypes.h"
#include "SColor.h"
#include "SMaterial.h"

#include "LinearMath/btQuaternion.h"

namespace irr
{
    namespace scene
    {
        class ISceneNode; class IBillboardSceneNode; struct SParticle;
        class IMesh; class ILightSceneNode;
    }
}

namespace GE
{
class GECullingTool;
struct GEVulkanBloomRects;
class GESPMBuffer;
class GEVulkanCameraSceneNode;
class GEVulkanDriver;
class GEVulkanDynamicBuffer;
class GEVulkanLightHandler;
class GEVulkanShadowFBO;
class GEVulkanSkyBoxRenderer;
class GEVulkanTextureDescriptor;

typedef std::pair<std::vector<VkVertexInputBindingDescription>,
    std::vector<VkVertexInputAttributeDescription> > VertexDescription;

struct ObjectData
{
    float m_translation_x;
    float m_translation_y;
    float m_translation_z;
    float m_hue_change;
    float m_rotation[4];
    float m_scale_x;
    float m_scale_y;
    float m_scale_z;
    irr::video::SColor m_custom_vertex_color;
    int m_skinning_offset;
    int m_material_id;
    float m_texture_trans[2];
    // ------------------------------------------------------------------------
    void init(irr::scene::ISceneNode* node, int material_id,
              int skinning_offset, irr::video::SMaterial& m);
    // ------------------------------------------------------------------------
    void init(irr::scene::IBillboardSceneNode* node, int material_id,
              const btQuaternion& rotation, bool mirror);
    // ------------------------------------------------------------------------
    void init(const irr::scene::SParticle& particle, int material_id,
              const btQuaternion& rotation,
              const irr::core::vector3df& view_position, bool flips,
              bool sky_particle, bool backface_culling);
};

enum GEVulkanPipelineType : unsigned
{
    GVPT_DEPTH = 1,
    GVPT_SOLID,
    GVPT_DEFERRED_LIGHTING,
    GVPT_DEFERRED_CONVERT_COLOR,
    GVPT_GHOST_DEPTH,
    GVPT_TRANSPARENT,
    GVPT_SKYBOX,
    GVPT_DISPLACE_MASK,
    GVPT_DISPLACE_COLOR,
    // Meshes with glow outline color, drawn to the render pass of
    // GEVulkanGlowOutline (same vertex shaders as the g-buffer)
    GVPT_GLOW_OUTLINE,
    // Fullscreen draw in the lighting pass which adds the blurred glow
    GVPT_LIGHTING_COMPOSITE,
};

struct GEMaterial;

struct PipelineSettings
{
    std::string m_shader_name;
    std::shared_ptr<const GEMaterial> m_material;
    char m_drawing_priority;
    VkPipelineLayout m_custom_pl;
    VkCompareOp m_depth_op;
    VkPrimitiveTopology m_topology;
    VertexDescription m_vertex_description;
    GEVulkanPipelineType m_pipeline_type;

    PipelineSettings();
    void loadMaterial(const GEMaterial& m);
};

struct PipelineData
{
    PipelineSettings m_settings;
    std::map<GEVulkanPipelineType, std::shared_ptr<VkPipeline> > m_pipelines;
};

struct DrawCallData
{
    VkDrawIndexedIndirectCommand m_cmd;
    std::string m_shader;
    GESPMBuffer* m_mb;
    int m_material_id;
    int m_dynamic_offset;
};

struct DynamicSPMData
{
    int m_material_id;
    uint32_t m_dynamic_offset;
    uint32_t m_instance_count;
    VkDescriptorSet m_descriptor_set;
};

// A batch (instances of one mesh buffer with the same glow color) of the glow
// outline of a draw call, see GEVulkanDrawCall::renderGlowOutline
struct GlowOutlineDrawData
{
    GESPMBuffer* m_mb;
    // Pipeline name, with the skinning suffix if needed
    std::string m_shader;
    // Dynamic offset of the first object data in m_dspm_data (aligned)
    uint32_t m_dynamic_offset;
    uint32_t m_instance_count;
    // 0xRRGGBB (linear)
    uint32_t m_color;
};

class GEVulkanHiZDepth;
class GEVulkanDrawCall
{
private:
    GEVulkanShadowFBO* m_shadow_fbo;

    // ------------------------------------------------------------------------
    virtual bool isShadow() const                             { return false; }
    // ------------------------------------------------------------------------
    virtual bool skip(irr::scene::ISceneNode* node) const     { return false; }
    // ------------------------------------------------------------------------
    virtual bool useDepthClamp() const                        { return false; }
    // ------------------------------------------------------------------------
    virtual bool ignoreMaterial(irr::video::E_MATERIAL_TYPE mt,
                                const irr::video::SMaterial& m) const
                                                              { return false; }
    // ------------------------------------------------------------------------
    size_t getDynamicSPMSize() const;
    // ------------------------------------------------------------------------
    void drawCommands(VkCommandBuffer cmd, const std::string& pipeline,
                      int current_buffer_idx,VkBuffer indirect_buffer,
                      size_t indirect_offset, unsigned draw_count,
                      GEVulkanPipelineType pt);
    // ------------------------------------------------------------------------
    void createPipelineLayout(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    virtual const GEVulkanDrawCall* getMasterDrawCall() const  { return this; }

protected:
    typedef std::array<const irr::video::ITexture*,
        _IRR_MATERIAL_MAX_TEXTURES_> TexturesList;

    std::map<TexturesList, GESPMBuffer*> m_billboard_buffers;

    irr::core::vector3df m_view_position;

    btQuaternion m_billboard_rotation;

    using Nodes = std::vector<std::pair<irr::scene::ISceneNode*,
        irr::video::SMaterial&> >;
    std::map<std::pair<GESPMBuffer*, int>,
        std::unordered_map<uint32_t, Nodes> > m_visible_nodes;

    std::map<GESPMBuffer*, irr::scene::IMesh*> m_mb_map;

    std::map<std::string,
        std::map<GESPMBuffer*, std::vector<irr::scene::ISceneNode*> > >
        m_dynamic_spm_buffers;

    // Meshes with glow outline color, by color, shader (pipeline name) and
    // mesh buffer (so it's sorted by color first, which is pushed as a
    // constant, and each key is one draw). Filled by addNode if
    // m_glow_outline, and written to m_dspm_data by generateDynamicSPM which
    // makes m_glow_outline_draws, the list which GEVulkanGlowOutline renders
    using GlowOutlineKey = std::pair<uint64_t, GESPMBuffer*>;
    std::map<GlowOutlineKey, std::vector<irr::scene::ISceneNode*
         > > m_glow_outline_nodes;

    std::vector<GlowOutlineDrawData> m_glow_outline_draws;

    // True if the current FBO has glow outline, set by prepare()
    bool m_glow_outline;

    // Same, the FBO has light scattering (GEVulkanLightScatter)
    bool m_light_scatter;

    GECullingTool* m_culling_tool;

    GEVulkanLightHandler* m_light_handler;

    std::vector<DrawCallData> m_cmds;

    std::vector<ObjectData> m_visible_objects;

    GEVulkanDynamicBuffer* m_indirect_buffer;

    GEVulkanDynamicBuffer* m_sbo_data;

    GEVulkanDynamicBuffer* m_skinning_data;

    GEVulkanDynamicBuffer* m_dspm_data;

    const VkPhysicalDeviceLimits& m_limits;

    size_t m_object_data_padded_size;

    size_t m_materials_padded_size;

    size_t m_dynamic_spm_padded_size;

    unsigned m_camera_ubo_offset;

    unsigned m_light_data_offset;

    unsigned m_skinning_offset;

    std::weak_ptr<bool> m_camera_ubo_observer, m_skinning_data_observer;

    bool m_update_data_descriptor_sets;

    VkDescriptorSetLayout m_data_layout;

    VkDescriptorPool m_descriptor_pool;

    std::vector<VkDescriptorSet> m_data_descriptor_sets,
        m_dspm_descriptor_sets;

    VkDescriptorSet m_env_descriptor_set;

    std::weak_ptr<bool> m_env_observer;

    std::weak_ptr<VkDescriptorSetLayout> m_texture_descriptor_set_layout;

    VkPipelineLayout m_pipeline_layout, m_skybox_layout;

    std::vector<VkPipelineLayout> m_deferred_layouts;

    std::unordered_map<std::string, PipelineData> m_graphics_pipelines;

    std::map<std::string, std::map<GESPMBuffer*, DynamicSPMData > >
        m_rendered_dspm;

    GEVulkanSkyBoxRenderer* m_skybox_renderer;

    GEVulkanTextureDescriptor* m_texture_descriptor;

    std::unordered_map<std::string, std::pair<uint32_t, std::vector<int> > >
        m_materials_data;

    GEVulkanHiZDepth* m_hiz_depth;

    std::array<irr::video::E_MATERIAL_TYPE,
        irr::video::EMT_MATERIAL_COUNT> m_fallback_materials;

    irr::scene::IMesh* m_lightning_mesh;

    // ------------------------------------------------------------------------
    void initNonPBRFallbackMaterials();
    // ------------------------------------------------------------------------
    void createAllPipelines(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void createPipeline(GEVulkanDriver* vk, const PipelineSettings& settings,
      std::unordered_map<std::string, std::shared_ptr<VkPipeline> >& dp_cache);
    // ------------------------------------------------------------------------
    void createVulkanData();
    // ------------------------------------------------------------------------
    bool bindPipeline(VkCommandBuffer cmd, const std::string& name,
                      VkPipeline* prev_pipeline,
                      GEVulkanPipelineType pt) const;
    // ------------------------------------------------------------------------
    TexturesList getTexturesList(const irr::video::SMaterial& m)
    {
        TexturesList textures;
        for (unsigned i = 0; i < textures.size(); i++)
            textures[i] = m.TextureLayer[i].Texture;
        return textures;
    }
    // ------------------------------------------------------------------------
    void addGlowOutlineNode(irr::scene::ISceneNode* node, GESPMBuffer* buffer,
                            irr::video::SMaterial& m,
                            irr::video::E_MATERIAL_TYPE mt);
    // ------------------------------------------------------------------------
    void bindBaseVertex(GEVulkanDriver* vk, VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    void bindSingleMaterial(VkCommandBuffer cmd,
                            const std::string& cur_pipeline,
                            int material_id, GEVulkanPipelineType pt);
    // ------------------------------------------------------------------------
    void bindDataDescriptor(VkCommandBuffer cmd, int current_buffer_idx,
                            std::vector<uint32_t>& dynamic_offsets)
    {
        vkCmdBindDescriptorSets(cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout, 1, 1,
            &m_data_descriptor_sets[current_buffer_idx],
            dynamic_offsets.size(), dynamic_offsets.data());
    }
    // ------------------------------------------------------------------------
    VertexDescription getDefaultVertexDescription() const;
    // ------------------------------------------------------------------------
    std::vector<uint32_t> getDefaultDynamicOffsets() const;
    // ------------------------------------------------------------------------
    virtual VkRenderPass getRenderPassForPipelineCreation(GEVulkanDriver* vk,
                                                  GEVulkanPipelineType type);
    // ------------------------------------------------------------------------
    virtual uint32_t getSubpassForPipelineCreation(GEVulkanDriver* vk,
                                           GEVulkanPipelineType type);
    // ------------------------------------------------------------------------
    virtual void generateDynamicSPM(GEVulkanDriver* vk);

public:
    // ------------------------------------------------------------------------
    GEVulkanDrawCall();
    // ------------------------------------------------------------------------
    virtual ~GEVulkanDrawCall();
    // ------------------------------------------------------------------------
    void addNode(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    void addBillboardNode(irr::scene::ISceneNode* node,
                          irr::scene::ESCENE_NODE_TYPE node_type);
    // ------------------------------------------------------------------------
    virtual void prepare(GEVulkanCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void generate(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void uploadDynamicData(GEVulkanDriver* vk, bool& has_indirect,
                           VkCommandBuffer custom_cmd = VK_NULL_HANDLE);
    // ------------------------------------------------------------------------
    virtual bool doDepthOnlyRenderingFirst();
    // ------------------------------------------------------------------------
    void bindAllMaterials(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    void updateDataDescriptorSets(GEVulkanDriver* vk,
                                  GEVulkanCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void prepareViewport(GEVulkanDriver* vk,
                         const irr::core::rect<irr::s32>& viewp,
                         VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    // The viewport which prepareViewport sets, in pixels of the render target
    VkViewport getRenderViewport(GEVulkanDriver* vk,
                                 const irr::core::rect<irr::s32>& viewp) const;
    // ------------------------------------------------------------------------
    void renderPipeline(GEVulkanDriver* vk, VkCommandBuffer cmd,
                        GEVulkanPipelineType pt, bool& rebind_base_vertex);
    // ------------------------------------------------------------------------
    bool renderSkyBox(GEVulkanDriver* vk, VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    void renderDeferredLighting(GEVulkanDriver* vk, VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    // bloom_rects are the areas of the bloom images (see
    // GEVulkanBloom::getRects) of this draw call, NULL if there is none (only
    // used if the FBO has bloom, the tonemap shader adds it to hdr)
    void renderDeferredConvertColor(GEVulkanDriver* vk, VkCommandBuffer cmd,
                                    const GEVulkanBloomRects* bloom_rects);
    // ------------------------------------------------------------------------
    // True if renderDeferredLighting and renderDeferredConvertColor draw
    // anything of this draw call (the deferred FBO is used, and any mesh or
    // sky is visible)
    bool hasDeferredOutput() const
    {
        return !m_deferred_layouts.empty() &&
            (!m_visible_nodes.empty() || m_skybox_renderer);
    }
    // ------------------------------------------------------------------------
    void renderDisplaceColor(GEVulkanDriver* vk, VkCommandBuffer cmd,
                             VkBool32 has_displace);
    // ------------------------------------------------------------------------
    // True if there is any mesh with glow outline to draw (valid after
    // generate()), each camera of splitscreen has its own list
    bool hasGlowOutline() const         { return !m_glow_outline_draws.empty(); }
    // ------------------------------------------------------------------------
    // True if the FBO has light scattering and this camera has fog and lights
    // this frame (known after generate), see GEVulkanLightScatter
    bool hasLightScatter() const;
    // ------------------------------------------------------------------------
    // Creates a layout of the data descriptor set (camera, lights), all of
    // them are defined the same so the sets of any draw call are compatible
    // with a pipeline layout created from any of them. The caller owns it, so
    // it can outlive the draw calls (a pipeline layout can't be used after the
    // layout it was created with is destroyed, see GEVulkanLightScatter)
    static VkDescriptorSetLayout createDataLayout(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    // Binds the data descriptor set (camera, lights) to a pipeline layout
    // whose set at the index is the data layout
    void bindDataDescriptorSet(GEVulkanDriver* vk, VkCommandBuffer cmd,
                               VkPipelineBindPoint bind_point,
                               VkPipelineLayout layout, uint32_t set) const;
    // ------------------------------------------------------------------------
    // Draws the glow list in the render pass of GEVulkanGlowOutline (viewport
    // already set), one vkCmdDrawIndexed per batch
    void renderGlowOutline(GEVulkanDriver* vk, VkCommandBuffer cmd,
                           bool& rebind_base_vertex);
    // ------------------------------------------------------------------------
    // Adds the blurred glow (without the area of meshes) to hdr, in the
    // lighting pass
    // Adds the glow outline and the light scattering (the ones which were
    // rendered this frame and this draw call has) to hdr
    void renderLightingComposite(GEVulkanDriver* vk, VkCommandBuffer cmd,
                                 bool glow_outline, bool light_scatter);
    // ------------------------------------------------------------------------
    unsigned getPolyCount() const
    {
        unsigned result = 0;
        for (auto& cmd : m_cmds)
            result += (cmd.m_cmd.indexCount / 3) * cmd.m_cmd.instanceCount;
        return result;
    }
    // ------------------------------------------------------------------------
    void reset()
    {
        m_visible_nodes.clear();
        m_mb_map.clear();
        m_cmds.clear();
        m_visible_objects.clear();
        m_rendered_dspm.clear();
        m_materials_data.clear();
        m_dynamic_spm_buffers.clear();
        m_glow_outline_nodes.clear();
        m_glow_outline_draws.clear();
        m_skybox_renderer = NULL;
    }
    // ------------------------------------------------------------------------
    void addSkyBox(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    void addLightNode(irr::scene::ILightSceneNode* node);
    // ------------------------------------------------------------------------
    bool hasDisplaceMaterial() const;
    // ------------------------------------------------------------------------
    bool hasShaderForRendering(const std::string& shader) const
    {
        return m_materials_data.find(shader) != m_materials_data.end();
    }
    // ------------------------------------------------------------------------
    GEVulkanHiZDepth* getHiZDepth() const               { return m_hiz_depth; }
    // ------------------------------------------------------------------------
    virtual const VkDescriptorSet* getEnvDescriptorSet(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    GEVulkanLightHandler* getLightHandler() const   { return m_light_handler; }
    // ------------------------------------------------------------------------
    GEVulkanShadowFBO* getShadowFBO() const            { return m_shadow_fbo; }
    // ------------------------------------------------------------------------
    void setCameraUBOOffset(unsigned offset)  { m_camera_ubo_offset = offset; }
    // ------------------------------------------------------------------------
    GECullingTool* getCullingTool() const            { return m_culling_tool; }
    // ------------------------------------------------------------------------
    void setLightDataOffset(unsigned offset)  { m_light_data_offset = offset; }
    // ------------------------------------------------------------------------
    const std::map<std::string, std::map<GESPMBuffer*, DynamicSPMData > >&
                      getRenderedDynamicSPM() const { return m_rendered_dspm; }
    // ------------------------------------------------------------------------
    irr::core::matrix4* getSkinningOffset(unsigned bone_count,
                                          unsigned* offset);
    // ----------------------------------------------------------------------------
    void swapDrawCallData(GEVulkanDrawCall* other)
    {
        std::swap(m_indirect_buffer, other->m_indirect_buffer);
        std::swap(m_materials_data, other->m_materials_data);
        std::swap(m_cmds, other->m_cmds);
        std::swap(m_dynamic_spm_buffers, other->m_dynamic_spm_buffers);
        std::swap(m_rendered_dspm, other->m_rendered_dspm);
        std::swap(m_data_descriptor_sets, other->m_data_descriptor_sets);
        std::swap(m_dspm_descriptor_sets, other->m_dspm_descriptor_sets);
    }

};   // GEVulkanDrawCall

}

#endif
