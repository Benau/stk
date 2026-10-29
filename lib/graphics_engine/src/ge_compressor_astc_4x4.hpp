#ifndef HEADER_GE_ASTC_COMPRESSOR_HPP
#define HEADER_GE_ASTC_COMPRESSOR_HPP

#include "ge_compressor_basis_universal.hpp"

namespace GE
{
class GECompressorASTC4x4 : public GECompressorBasisUniversal
{
private:
    // ------------------------------------------------------------------------
    virtual bool pack(uint8_t* out, const basist::color_rgba* pixels);
public:
    // ------------------------------------------------------------------------
    GECompressorASTC4x4(uint8_t* texture, unsigned channels,
                        const irr::core::dimension2d<irr::u32>& size,
                        bool normal_map)
           : GECompressorBasisUniversal(texture, channels, size, normal_map) {}
};   // GECompressorASTC4x4

}

#endif
