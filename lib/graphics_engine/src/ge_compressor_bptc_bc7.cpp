#include "ge_compressor_bptc_bc7.hpp"
#include "ge_main.hpp"

#ifdef BASISU_TC
#include <basisu_transcoder.h>
#endif
#include <algorithm>
#include <cassert>

namespace GE
{
// ============================================================================
void GECompressorBPTCBC7::init()
{
#ifdef BASISU_TC
    basist::basisu_transcoder_init();
#endif
}   // init

// ----------------------------------------------------------------------------
GECompressorBPTCBC7::GECompressorBPTCBC7(uint8_t* texture, unsigned channels,
                                         const irr::core::dimension2d<irr::u32>& size,
                                         bool normal_map)
                   : GEMipmapGenerator(texture, channels, size, normal_map)
{
    m_compressed_data = NULL;
#ifdef BASISU_TC
    assert(channels == 4);
    size_t total_size = 0;
    m_mipmap_sizes = 0;
    for (unsigned i = 0; i < m_levels.size(); i++)
    {
        GEImageLevel& level = m_levels[i];
        unsigned cur_size = get4x4CompressedTextureSize(level.m_dim.Width,
            level.m_dim.Height);
        total_size += cur_size;
        if (i > 0)
            m_mipmap_sizes += cur_size;
    }

    // Fully analytical BC7F mode: fast, without iterative refinement.
    const uint32_t bc7f_flags = basist::bc7f::cPackBC7FlagDefault;
    std::vector<GEImageLevel> compressed_levels;
    m_compressed_data = new uint8_t[total_size];
    uint8_t* cur_offset = m_compressed_data;

    for (GEImageLevel& level : m_levels)
    {
        uint8_t* out = cur_offset;
        const unsigned width = level.m_dim.Width;
        const unsigned height = level.m_dim.Height;
        const size_t pitch = size_t(width) * 4;
        const uint8_t* rgba = static_cast<const uint8_t*>(level.m_data);
        for (unsigned y = 0; y < height; y += 4)
        {
            for (unsigned x = 0; x < width; x += 4)
            {
                basist::color_rgba source_rgba[16];
                // BC7F takes 16 RGBA pixels in row-major order.
                // Clamp partial edge blocks to the last valid texel.
                for (unsigned py = 0; py < 4; py++)
                {
                    for (unsigned px = 0; px < 4; px++)
                    {
                        const unsigned sx = std::min(x + px, width - 1);
                        const unsigned sy = std::min(y + py, height - 1);
                        const uint8_t* src =
                            rgba + size_t(sy) * pitch + size_t(sx) * 4;
                        basist::color_rgba& dst = source_rgba[py * 4 + px];
                        dst.r = src[0];
                        dst.g = src[1];
                        dst.b = src[2];
                        dst.a = src[3];
                    }
                }
                basist::bc7f::fast_pack_bc7_auto_rgba(
                    out, source_rgba, bc7f_flags);
                out += 16;
            }
        }
        const unsigned cur_size = get4x4CompressedTextureSize(width, height);
        compressed_levels.push_back({ level.m_dim, cur_size, cur_offset });
        cur_offset += cur_size;
    }
    freeMipmapCascade();
    std::swap(compressed_levels, m_levels);
#endif
}   // GECompressorBPTCBC7

}
