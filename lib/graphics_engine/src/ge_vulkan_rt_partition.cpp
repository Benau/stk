#include "ge_vulkan_rt_scene.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_main.hpp"
#include "mini_glm.hpp"
#include "ISceneNode.h"
#include "aabbox3d.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <queue>

namespace GE
{
namespace
{
using irr::core::vector3df;
using irr::core::aabbox3df;

struct Triangle
{
    uint32_t source, first;
    vector3df center;
    aabbox3df bounds;
};

struct Bin
{
    size_t count = 0;
    aabbox3df bounds;

    void add(const aabbox3df& box, size_t amount)
    {
        if (!amount)
            return;
        if (count == 0)
            bounds = box;
        else
            bounds.addInternalBox(box);
        count += amount;
    }

    double cost() const { return double(bounds.getArea()) * count; }
};

using TriangleIterator = std::vector<Triangle>::iterator;

TriangleIterator splitByTraversalCost(TriangleIterator first, TriangleIterator end, double minimum_gain)
{
    aabbox3df centers(first->center);
    Bin parent;
    for (auto triangle = first; triangle != end; ++triangle)
    {
        centers.addInternalPoint(triangle->center);
        parent.add(triangle->bounds, 1);
    }
    const size_t bucket_count = 8;
    const auto extent = centers.getExtent();
    const float vector3df::* axes[] = { &vector3df::X, &vector3df::Y, &vector3df::Z };
    std::array<std::array<Bin, bucket_count>, 3> bins = {};
    auto bucket = [&](const Triangle& triangle, unsigned axis)
    {
        const double offset = double(triangle.center.*axes[axis]) - centers.MinEdge.*axes[axis];
        return std::min(bucket_count - 1, size_t(bucket_count * offset / (extent.*axes[axis])));
    };
    for (auto triangle = first; triangle != end; ++triangle)
    {
        for (unsigned axis = 0; axis < bins.size(); ++axis)
            if (extent.*axes[axis] > 0.0f)
                bins[axis][bucket(*triangle, axis)].add(triangle->bounds, 1);
    }
    double best_cost = parent.cost() * (1.0 - minimum_gain);
    unsigned best_axis = 0;
    size_t boundary = 0;
    for (unsigned axis = 0; axis < bins.size(); ++axis)
    {
        const auto& row = bins[axis];
        std::array<Bin, bucket_count> right = row;
        for (size_t index = bucket_count - 1; index > 0; --index)
            right[index - 1].add(right[index].bounds, right[index].count);
        Bin left;
        for (size_t index = 1; index < bucket_count; ++index)
        {
            left.add(row[index - 1].bounds, row[index - 1].count);
            const double cost = left.cost() + right[index].cost();
            if (left.count && right[index].count && cost < best_cost)
            {
                best_cost = cost;
                best_axis = axis;
                boundary = index;
            }
        }
    }
    if (!boundary)
        return end;
    return std::partition(first, end, [&](const Triangle& triangle)
    {
        return bucket(triangle, best_axis) < boundary;
    });
}

vector3df position(const GERTVertex& vertex)
{
    return vector3df(vertex.m_position[0], vertex.m_position[1], vertex.m_position[2]);
}

vector3df toLocal(const vector3df& value, float cosine, float sine)
{
    return vector3df(cosine * value.X + sine * value.Z, value.Y,
        cosine * value.Z - sine * value.X);
}

uint32_t rotateDirection(uint32_t packed, float cosine, float sine)
{
    return (packed & 0xc0000000u) | (MiniGLM::compressVector3(
        toLocal(MiniGLM::decompressVector3(packed), cosine, sine)) & 0x3fffffffu);
}
}

bool GEVulkanRTScene::partitionStaticMeshes(std::vector<GERTVertex>& vertices,
    std::vector<uint32_t>& indices, std::vector<GERTGeometry>& infos,
    std::vector<std::pair<uint32_t, uint32_t> >& counts)
{
    std::vector<Triangle> triangles;
    triangles.reserve(indices.size() / 3);
    for (uint32_t source = 0; source < infos.size(); ++source)
    {
        const GERTGeometry& info = infos[source];
        for (uint32_t offset = 0; offset < counts[source].second; offset += 3)
        {
            const uint32_t first = info.m_index_offset + offset;
            vector3df sum;
            aabbox3df bounds;
            for (unsigned corner = 0; corner < 3; ++corner)
            {
                if (indices[first + corner] >= counts[source].first)
                    return false;
                const auto point = position(vertices[info.m_vertex_offset + indices[first + corner]]);
                if (corner == 0)
                    bounds.reset(point);
                else
                    bounds.addInternalPoint(point);
                sum += point;
            }
            triangles.push_back({ source, first, sum / 3.0f, bounds });
        }
    }
    if (triangles.empty())
        return false;

    const char* setting = std::getenv("STK_RT_BLAS_LEAF_TRIANGLES");
    const size_t target = setting ? std::max(64L, std::min(65536L, std::strtol(setting, NULL, 10))) : 2048;
    const size_t capacity = std::max(1u, m_instance_capacity / 4);
    const size_t leaf_size = std::max(target, (triangles.size() + capacity - 1) / capacity);
    const char* gain_setting = std::getenv("STK_RT_BLAS_SPLIT_GAIN");
    const double minimum_gain = gain_setting ? std::clamp(std::strtod(gain_setting, NULL), 0.0, 1.0) : 0.30;
    const uint32_t limit = GEVulkanFeatures::getAccelerationStructureMaxGeometryCount();
    using Range = std::pair<size_t, size_t>;
    auto smaller = [](const Range& left, const Range& right)
    {
        const size_t left_count = left.second - left.first;
        const size_t right_count = right.second - right.first;
        return left_count == right_count ? left.first > right.first : left_count < right_count;
    };
    std::priority_queue<Range, std::vector<Range>, decltype(smaller)> pending(smaller);
    pending.emplace(0, triangles.size());
    std::vector<Range> leaves;
    while (!pending.empty())
    {
        const Range range = pending.top();
        pending.pop();
        const size_t count = range.second - range.first;
        if (count <= leaf_size / 2 || leaves.size() + pending.size() + 1 >= capacity)
        {
            leaves.push_back(range);
            continue;
        }
        auto first = triangles.begin() + range.first;
        auto end = triangles.begin() + range.second;
        const double gain = count > leaf_size ? minimum_gain : std::min(1.0, minimum_gain * 2.0);
        const auto split = splitByTraversalCost(first, end, gain);
        if (split == end)
            leaves.push_back(range);
        else
        {
            const size_t middle = split - triangles.begin();
            pending.emplace(range.first, middle);
            pending.emplace(middle, range.second);
        }
    }
    std::sort(leaves.begin(), leaves.end());

    std::vector<GERTVertex> packed_vertices;
    std::vector<uint32_t> packed_indices, remap(vertices.size()), stamps(vertices.size(), 0);
    std::vector<GERTGeometry> packed_infos;
    std::vector<GeometrySource> sources;
    packed_vertices.reserve(vertices.size());
    packed_indices.reserve(indices.size());
    counts.clear();
    for (const Range& range : leaves)
    {
        auto first = triangles.begin() + range.first;
        auto end = triangles.begin() + range.second;
        std::sort(first, end, [](const Triangle& left, const Triangle& right)
        {
            return left.source == right.source ? left.first < right.first : left.source < right.source;
        });
        vector3df center;
        for (auto triangle = first; triangle != end; ++triangle)
            center += triangle->center;
        center /= float(end - first);
        double covariance_xx = 0.0, covariance_zz = 0.0, covariance_xz = 0.0;
        for (auto triangle = first; triangle != end; ++triangle)
        {
            vector3df delta = triangle->center - center;
            covariance_xx += double(delta.X) * delta.X;
            covariance_zz += double(delta.Z) * delta.Z;
            covariance_xz += double(delta.X) * delta.Z;
        }
        const float yaw = float(0.5 * std::atan2(2.0 * covariance_xz, covariance_xx - covariance_zz));
        float cosine = std::cos(yaw), sine = std::sin(yaw);
        aabbox3df world_bounds(vector3df(0)), rotated_bounds(vector3df(0));
        for (auto triangle = first; triangle != end; ++triangle)
        {
            uint32_t base = infos[triangle->source].m_vertex_offset;
            for (unsigned corner = 0; corner < 3; ++corner)
            {
                vector3df delta = position(vertices[base + indices[triangle->first + corner]]) - center;
                world_bounds.addInternalPoint(delta);
                rotated_bounds.addInternalPoint(toLocal(delta, cosine, sine));
            }
        }
        if (rotated_bounds.getArea() >= world_bounds.getArea() * 0.95f)
        {
            cosine = 1.0f;
            sine = 0.0f;
        }
        StaticPart part = {};
        part.m_transform = {{{cosine, 0, -sine, center.X}, {0, 1, 0, center.Y},
            {sine, 0, cosine, center.Z}}};
        part.m_first = uint32_t(packed_infos.size());
        for (auto triangle = first; triangle != end; )
        {
            uint32_t source = triangle->source;
            GERTGeometry info = infos[source];
            info.m_vertex_offset = uint32_t(packed_vertices.size());
            info.m_index_offset = uint32_t(packed_indices.size());
            uint32_t stamp = uint32_t(packed_infos.size()) + 1;
            do
            {
                for (unsigned corner = 0; corner < 3; ++corner)
                {
                    uint32_t id = infos[source].m_vertex_offset + indices[triangle->first + corner];
                    if (stamps[id] != stamp)
                    {
                        stamps[id] = stamp;
                        remap[id] = uint32_t(packed_vertices.size()) - info.m_vertex_offset;
                        GERTVertex vertex = vertices[id];
                        vector3df local = toLocal(position(vertex) - center, cosine, sine);
                        vertex.m_position[0] = local.X;
                        vertex.m_position[1] = local.Y;
                        vertex.m_position[2] = local.Z;
                        if (sine != 0.0f)
                        {
                            vertex.m_normal = rotateDirection(vertex.m_normal, cosine, sine);
                            vertex.m_tangent = rotateDirection(vertex.m_tangent, cosine, sine);
                        }
                        packed_vertices.push_back(vertex);
                    }
                    packed_indices.push_back(remap[id]);
                }
                ++triangle;
            } while (triangle != end && triangle->source == source);
            counts.emplace_back(uint32_t(packed_vertices.size()) - info.m_vertex_offset,
                uint32_t(packed_indices.size()) - info.m_index_offset);
            packed_infos.push_back(info);
            sources.push_back(m_geometry_sources[source]);
            ++part.m_count;
            if (triangle == end || part.m_count == limit)
            {
                m_static_parts.push_back(part);
                part.m_first += part.m_count;
                part.m_count = 0;
            }
        }
    }
    if (m_static_parts.size() > m_instance_capacity || packed_infos.size() + MAX_DYNAMIC_GEOMETRIES > 0xffffffu)
        return false;
    for (const GeometrySource& source : sources) source.m_node->grab();
    for (const GeometrySource& source : m_geometry_sources) source.m_node->drop();
    m_geometry_sources.swap(sources);
    vertices.swap(packed_vertices);
    indices.swap(packed_indices);
    infos.swap(packed_infos);
    return true;
}
}
