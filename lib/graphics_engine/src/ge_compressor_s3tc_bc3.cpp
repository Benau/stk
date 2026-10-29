#include "ge_compressor_s3tc_bc3.hpp"

#ifdef BASISU_TC
#include <basisu_transcoder.h>
#endif

namespace GE
{
// ----------------------------------------------------------------------------
bool GECompressorS3TCBC3::pack(uint8_t* out, const basist::color_rgba* pixels)
{
#ifdef BASISU_TC
    const uint8_t* rgba =
        reinterpret_cast<const uint8_t*>(pixels);
    // BC3: 8-byte alpha block, followed by an 8-byte RGB block.
    basist::encode_bc4(out, rgba + 3, 4);
    basist::encode_bc1(out + 8, rgba, basist::cEncodeBC1HigherQuality);
    return true;
#else
    return false;
#endif
}   // pack

}
