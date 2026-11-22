#include "ge_vulkan_static_scene_manager.hpp"

#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_render_info.hpp"
#include "ge_spm_buffer.hpp"
#include "ge_vulkan_lod_node.hpp"
#include "ge_vulkan_scene_manager.hpp"
#include "../source/Irrlicht/CEmptySceneNode.h"

#include "IMeshSceneNode.h"

namespace GE
{
using namespace irr;
using Node = scene::ISceneNode;

// ============================================================================
// A branch keeps the original parent's world transform while its static
// children are detached. The scene manager releases the branches before STK
// removes the track objects through their original nodes.
class StaticSceneBranch : public scene::CEmptySceneNode
{
public:
    Node* m_source;
    // ------------------------------------------------------------------------
    StaticSceneBranch(Node* source) : CEmptySceneNode(NULL,
        source->getSceneManager(), -1), m_source(source)
    {
        if (m_source != SceneManager->getRootSceneNode())
            m_source->grab();
        AbsoluteTransformation = m_source->getAbsoluteTransformation();
    }
    // ------------------------------------------------------------------------
    ~StaticSceneBranch()
    {
        if (m_source != SceneManager->getRootSceneNode())
            m_source->drop();
    }
    // ------------------------------------------------------------------------
    const core::vector3df& getScale() const override
    {
        return m_source->getScale();
    }
};   // StaticSceneBranch

// ============================================================================
static bool isStaticMesh(Node* node)
{
    if (node->getType() != irr::scene::ESNT_MESH)
        return false;
    scene::IMesh* mesh =
        static_cast<irr::scene::IMeshSceneNode*>(node)->getMesh();
    if (!mesh || !mesh->getMeshBufferCount() ||
        mesh->getMeshBufferCount() != node->getMaterialCount())
        return false;
    for (unsigned index = 0; index < node->getMaterialCount(); index++)
    {
        const auto& info = node->getMaterial(index).getRenderInfo();
        auto* buffer = dynamic_cast<GESPMBuffer*>(mesh->getMeshBuffer(index));
        if (!info || info->getRayTracingType() != GRTT_STATIC ||
            !buffer || buffer->isDynamic())
            return false;
    }
    return true;
}   // isStaticMesh

// ----------------------------------------------------------------------------
static bool canExtract(Node* node)
{
    if (!node->isVisible() || !node->getAnimators().empty())
        return false;
    const auto& children = node->getChildren();
    if (auto* lod = dynamic_cast<GEVulkanLODNode*>(node))
    {
        if (!lod->hasOnlyLevels() || children.empty())
            return false;
        for (unsigned index = 0; index < children.size(); index++)
        {
            Node* level = children[index];
            if (!level->getAnimators().empty() ||
                !level->getChildren().empty() || !isStaticMesh(level))
                return false;
        }
        return true;
    }
    else if (node->getType() != scene::ESNT_EMPTY)
        return children.empty() && isStaticMesh(node);
    if (children.empty())
        return false;
    for (unsigned index = 0; index < children.size(); index++)
    {
        if (!canExtract(children[index]))
            return false;
    }
    return true;
}   // canExtract

// ----------------------------------------------------------------------------
static bool containsStatic(Node* node)
{
    for (unsigned index = 0; index < node->getMaterialCount(); index++)
    {
        const auto& info = node->getMaterial(index).getRenderInfo();
        if (info && info->getRayTracingType() == GRTT_STATIC)
            return true;
    }
    const auto& children = node->getChildren();
    for (unsigned index = 0; index < children.size(); index++)
    {
        if (containsStatic(children[index]))
            return true;
    }
    return false;
}   // containsStatic

// ============================================================================
GEVulkanStaticSceneManager::~GEVulkanStaticSceneManager()
{
    release();
}   // ~GEVulkanStaticSceneManager

// ----------------------------------------------------------------------------
void GEVulkanStaticSceneManager::splitChildren(Node* parent)
{
    if (!parent->isVisible())
        return;
    parent->updateAbsolutePosition();
    auto children = parent->getChildren();
    std::vector<Node*> destinations;
    destinations.reserve(children.size());
    StaticSceneBranch* branch = NULL;
    for (unsigned index = 0; index < children.size(); index++)
    {
        Node* child = children[index];
        if (canExtract(child))
        {
            if (!branch)
            {
                branch = new StaticSceneBranch(parent);
                m_branches.push_back(branch);
            }
            destinations.push_back(branch);
        }
        else
        {
            destinations.push_back(parent);
            if (child->getType() == scene::ESNT_EMPTY &&
                child->getAnimators().empty())
                splitChildren(child);
        }
    }
    if (!branch)
        return;
    for (unsigned index = 0; index < children.size(); index++)
        children[index]->grab();
    parent->ISceneNode::removeAll();
    for (unsigned index = 0; index < children.size(); index++)
    {
        destinations[index]->addChild(children[index]);
        children[index]->drop();
    }
}   // splitChildren

// ----------------------------------------------------------------------------
void GEVulkanStaticSceneManager::cache(Node* node)
{
    node->setNeedsUpdateAbsTrans(true);
    node->updateAbsolutePosition();
    auto* lod = dynamic_cast<GEVulkanLODNode*>(node);
    const auto& children = node->getChildren();
    if (node->getType() == scene::ESNT_EMPTY)
    {
        for (unsigned index = 0; index < children.size(); index++)
            cache(children[index]);
        return;
    }
    if (lod)
    {
        for (unsigned index = 0; index < children.size(); index++)
        {
            children[index]->setNeedsUpdateAbsTrans(true);
            children[index]->updateAbsolutePosition();
        }
    }
    m_entries.push_back({node, lod});
    Node* highest = lod ? lod->getHighestLevel() : node;
    m_rt_meshes.push_back(highest);
    static const auto grass = GEMaterialManager::getIrrMaterialType("grass");
    for (unsigned index = 0; index < highest->getMaterialCount(); index++)
    {
        if (highest->getMaterialType(index) == grass)
        {
            m_rt_grass.push_back(highest);
            break;
        }
    }
}   // cache

// ----------------------------------------------------------------------------
void GEVulkanStaticSceneManager::registerNodes(GEVulkanSceneManager* manager,
    const core::matrix4& camera_transform)
{
    for (const Entry& entry : m_entries)
    {
        Node* node = entry.m_node;
        if (entry.m_lod)
            node = entry.m_lod->getLevelForView(camera_transform);
        if (!node)
            continue;
        node->refreshTextureDescriptorID();
        manager->registerNodeForRendering(node, scene::ESNRP_SOLID);
    }
}   // registerNodes

// ----------------------------------------------------------------------------
void GEVulkanStaticSceneManager::split(GEVulkanSceneManager* manager)
{
    release();
    splitChildren(manager);
    for (StaticSceneBranch* branch : m_branches)
    {
        const auto& children = branch->getChildren();
        for (unsigned index = 0; index < children.size(); index++)
            cache(children[index]);
    }
    m_live_static = containsStatic(manager);
    m_split = true;
    logMessage(GLL_INFO, "Static scene: %zu drawables in %zu branches",
        m_entries.size(), m_branches.size());
}   // split

// ----------------------------------------------------------------------------
void GEVulkanStaticSceneManager::release()
{
    for (auto* branch : m_branches)
    {
        auto children = branch->getChildren();
        for (unsigned index = 0; index < children.size(); index++)
            children[index]->grab();
        branch->removeAll();
        for (unsigned index = 0; index < children.size(); index++)
        {
            branch->m_source->addChild(children[index]);
            children[index]->drop();
        }
        branch->drop();
    }
    m_branches.clear();
    m_entries.clear();
    m_rt_meshes.clear();
    m_rt_grass.clear();
    m_live_static = true;
    m_split = false;
}   // release

// ----------------------------------------------------------------------------
void GEVulkanStaticSceneManager::detectDeferred(
                                           GEVulkanSceneManager* manager) const
{
    for (StaticSceneBranch* branch : m_branches)
        manager->detectDeferred(branch);
}   // detectDeferred

}
