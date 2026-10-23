#import <Metal/Metal.h>
// ============================================================================
int g_enable_ray_query = 0;
extern "C" int enableRayQuery()
{
    return g_enable_ray_query;
}   // enableRayQuery

// ============================================================================
extern "C" void detectRayQuery()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device != nil)
    {
        BOOL supportsRayQueries = NO;

        if (@available(macOS 13.0, tvOS 16.0, iOS 16.0, *))
        {
            supportsRayQueries =
                device.supportsRaytracing &&
                device.supportsRaytracingFromRender &&
                device.argumentBuffersSupport == MTLArgumentBuffersTier2;
        }
        g_enable_ray_query = supportsRayQueries ? 1 : 0;
#if !__has_feature(objc_arc)
        [device release];
#endif
    }
}   // detectRayQuery
