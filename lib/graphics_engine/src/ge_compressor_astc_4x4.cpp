#include "ge_compressor_astc_4x4.hpp"

#ifdef BASISU_TC
#include <basisu_transcoder.h>
#endif

namespace GE
{
// ----------------------------------------------------------------------------
bool GECompressorASTC4x4::pack(uint8_t* out, const basist::color_rgba* pixels)
{
#ifdef BASISU_TC
    return basist::bc7f::fast_pack_astc(out, pixels, 0);
#else
    return false;
#endif
}   // pack

}
