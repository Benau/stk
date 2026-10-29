#ifndef HEADER_GE_COMPRESSOR_BASIS_UNIVERSAL_HPP
#define HEADER_GE_COMPRESSOR_BASIS_UNIVERSAL_HPP

#include "ge_mipmap_generator.hpp"

#include <cassert>

namespace basist
{
    class color_rgba;
}

namespace GE
{
class GECompressorBasisUniversal : public GEMipmapGenerator
{
private:
    uint8_t* m_compressed_data;
    // ------------------------------------------------------------------------
    virtual bool pack(uint8_t* out, const basist::color_rgba* pixels) = 0;
public:
    // ------------------------------------------------------------------------
    static void init();
    // ------------------------------------------------------------------------
    GECompressorBasisUniversal(uint8_t* texture, unsigned channels,
                               const irr::core::dimension2d<irr::u32>& size,
                               bool normal_map)
        : GEMipmapGenerator(texture, channels, size, normal_map)
    {
        assert(channels == 4);
        m_compressed_data = NULL;
    }
    // ------------------------------------------------------------------------
    ~GECompressorBasisUniversal()              { delete [] m_compressed_data; }
    // ------------------------------------------------------------------------
    virtual void handleMipmap();
};   // GECompressorBasisUniversal

}

#endif
