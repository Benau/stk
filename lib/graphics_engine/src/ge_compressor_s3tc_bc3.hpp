#ifndef HEADER_GE_COMPRESSOR_S3TC_BC3_HPP
#define HEADER_GE_COMPRESSOR_S3TC_BC3_HPP

#include "ge_compressor_basis_universal.hpp"

namespace GE
{
class GECompressorS3TCBC3 : public GECompressorBasisUniversal
{
private:
    // ------------------------------------------------------------------------
    virtual bool pack(uint8_t* out, const basist::color_rgba* pixels);
public:
    // ------------------------------------------------------------------------
    GECompressorS3TCBC3(uint8_t* texture, unsigned channels,
                        const irr::core::dimension2d<irr::u32>& size,
                        bool normal_map)
           : GECompressorBasisUniversal(texture, channels, size, normal_map) {}
};   // GECompressorS3TCBC3

}

#endif
