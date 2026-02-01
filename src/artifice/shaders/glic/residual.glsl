#version 430

// GLIC Residual Calculator
// Computes residuals: original - prediction
// This is where glitch effects emerge from the encoding errors

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba32f, binding = 0) readonly uniform image2D original_image;
layout(rgba32f, binding = 1) readonly uniform image2D prediction_image;
layout(rgba32f, binding = 2) writeonly uniform image2D output_image;

uniform int clamp_method;  // How to clamp residuals
uniform float scale;       // Residual scale/amplification

// Clamp methods (from GLIC)
const int CLAMP_NONE = 0;      // No clamping, can go negative (stored as 0.5 + r*0.5)
const int CLAMP_MOD256 = 1;    // Modulo 256 wrapping (glitchy!)
const int CLAMP_ABS = 2;       // Absolute value
const int CLAMP_CLIP = 3;      // Clip to 0-1

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(original_image);

    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }

    vec4 original = imageLoad(original_image, pixel);
    vec4 prediction = imageLoad(prediction_image, pixel);

    // Calculate residual
    vec3 residual = original.rgb - prediction.rgb;

    // Apply scale (amplifies the residual for artistic effect)
    residual *= scale;

    // Apply clamping method
    vec3 result;
    switch (clamp_method) {
        case CLAMP_NONE:
            // Store as 0.5 + residual * 0.5 to preserve negative values
            result = residual * 0.5 + 0.5;
            break;

        case CLAMP_MOD256:
            // Modulo wrapping - creates glitchy color banding
            result = fract(residual + 1.0);
            break;

        case CLAMP_ABS:
            // Absolute value - shows magnitude of error
            result = abs(residual);
            break;

        case CLAMP_CLIP:
        default:
            // Simple clip to valid range
            result = clamp(residual + 0.5, 0.0, 1.0);
            break;
    }

    imageStore(output_image, pixel, vec4(result, original.a));
}
