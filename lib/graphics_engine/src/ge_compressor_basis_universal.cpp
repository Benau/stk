#include "ge_compressor_basis_universal.hpp"
#include "ge_main.hpp"

#ifdef BASISU_TC
#include <basisu_transcoder.h>
#endif
#include <algorithm>

namespace GE
{
// ============================================================================
void GECompressorBasisUniversal::init()
{
#ifdef BASISU_TC
    basist::basisu_transcoder_init();
#endif
}   // init

// ----------------------------------------------------------------------------
void GECompressorBasisUniversal::handleMipmap()
{
    GEMipmapGenerator::handleMipmap();
#ifdef BASISU_TC
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
                // Build a 4x4 block of RGBA pixels for the BasisU encoder.
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
                if (!pack(out, source_rgba))
                {
                    GE::logMessage(
                        GE::GLL_ERROR, "BasisU compression failed for %s!",
                        typeid(*this).name());
                }
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
}   // handleMipmap

}
