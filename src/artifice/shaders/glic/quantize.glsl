#version 430

// GLIC Quantization
// Reduces precision of values - key to GLIC's characteristic banding effects

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform float quant_value;   // Quantization divisor (higher = more quantization)
uniform bool forward;        // true = quantize (divide), false = dequantize (multiply)
uniform int quant_mode;      // Quantization mode

// Quantization modes
const int QUANT_UNIFORM = 0;      // Uniform quantization (divide by value)
const int QUANT_LEVELS = 1;       // Reduce to N levels
const int QUANT_POSTERIZE = 2;    // Posterize effect
const int QUANT_BITCRUSH = 3;     // Bit depth reduction

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }

    vec4 color = imageLoad(input_image, pixel);
    vec3 result;

    float q = max(quant_value, 1.0);

    switch (quant_mode) {
        case QUANT_UNIFORM:
            // GLIC-style: divide then multiply rounds to quantization steps
            if (forward) {
                result = floor(color.rgb * 255.0 / q) * q / 255.0;
            } else {
                result = color.rgb; // Already quantized
            }
            break;

        case QUANT_LEVELS:
            {
                // Reduce to N discrete levels
                float levels = q;
                result = floor(color.rgb * levels) / (levels - 1.0);
            }
            break;

        case QUANT_POSTERIZE:
            {
                // Posterize with smooth steps
                float levels = max(q, 2.0);
                result = floor(color.rgb * levels + 0.5) / levels;
            }
            break;

        case QUANT_BITCRUSH:
            {
                // Reduce bit depth
                float bits = clamp(q, 1.0, 8.0);
                float steps = pow(2.0, bits);
                result = floor(color.rgb * steps) / (steps - 1.0);
            }
            break;

        default:
            result = color.rgb;
    }

    imageStore(output_image, pixel, vec4(result, color.a));
}
