#ifndef HEADER_GE_COMPRESSOR_BPTC_BC7_HPP
#define HEADER_GE_COMPRESSOR_BPTC_BC7_HPP

#include "ge_compressor_basis_universal.hpp"

namespace GE
{
class GECompressorBPTCBC7 : public GECompressorBasisUniversal
{
private:
    // ------------------------------------------------------------------------
    virtual bool pack(uint8_t* out, const basist::color_rgba* pixels);
public:
    // ------------------------------------------------------------------------
    GECompressorBPTCBC7(uint8_t* texture, unsigned channels,
                        const irr::core::dimension2d<irr::u32>& size,
                        bool normal_map)
           : GECompressorBasisUniversal(texture, channels, size, normal_map) {}
};   // GECompressorBPTCBC7

}

#endif
