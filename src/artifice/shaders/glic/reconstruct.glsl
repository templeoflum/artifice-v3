#version 430

// GLIC Reconstruction
// Reconstructs image: prediction + residual
// With quantized residuals, this creates the characteristic GLIC glitch look

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba32f, binding = 0) readonly uniform image2D prediction_image;
layout(rgba32f, binding = 1) readonly uniform image2D residual_image;
layout(rgba32f, binding = 2) writeonly uniform image2D output_image;

uniform int clamp_method;  // Must match residual's clamp method
uniform float scale;       // Inverse of residual scale

// Clamp methods
const int CLAMP_NONE = 0;
const int CLAMP_MOD256 = 1;
const int CLAMP_ABS = 2;
const int CLAMP_CLIP = 3;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(prediction_image);

    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }

    vec4 prediction = imageLoad(prediction_image, pixel);
    vec4 residual_stored = imageLoad(residual_image, pixel);

    // Decode residual based on clamp method
    vec3 residual;
    switch (clamp_method) {
        case CLAMP_NONE:
            // Decode from 0.5-centered storage
            residual = (residual_stored.rgb - 0.5) * 2.0;
            break;

        case CLAMP_MOD256:
            // Decode modulo-wrapped residual
            residual = residual_stored.rgb - 0.5;
            break;

        case CLAMP_ABS:
            // Absolute residual (loses sign - creates interesting effects)
            residual = residual_stored.rgb;
            break;

        case CLAMP_CLIP:
        default:
            residual = residual_stored.rgb - 0.5;
            break;
    }

    // Apply inverse scale
    residual /= max(scale, 0.001);

    // Reconstruct
    vec3 result = prediction.rgb + residual;

    // Final clamp to valid range
    result = clamp(result, 0.0, 1.0);

    imageStore(output_image, pixel, vec4(result, prediction.a));
}
