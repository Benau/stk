#include "ge_compressor_bptc_bc7.hpp"

#ifdef BASISU_TC
#include <basisu_transcoder.h>
#endif

namespace GE
{
// ----------------------------------------------------------------------------
bool GECompressorBPTCBC7::pack(uint8_t* out, const basist::color_rgba* pixels)
{
#ifdef BASISU_TC
    // Fully analytical BC7F mode: fast, without iterative refinement.
    basist::bc7f::fast_pack_bc7_auto_rgba(out, pixels,
        basist::bc7f::cPackBC7FlagDefault);
    return true;
#else
    return false;
#endif
}   // pack

}
