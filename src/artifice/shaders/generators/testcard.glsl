#version 430

// Test Card Generator
// Procedural calibration patterns for testing effects

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba32f, binding = 0) writeonly uniform image2D output_image;

uniform int pattern;      // 0=ColorBars, 1=GradH, 2=GradV, 3=Checker, 4=ZonePlate, 5=RGBSweep
uniform float scale;      // Pattern scale multiplier
uniform ivec2 resolution; // Output size

const float PI = 3.14159265359;

// SMPTE color bars (8 colors)
vec3 colorBar(int index) {
    switch(index) {
        case 0: return vec3(1.0, 1.0, 1.0); // White
        case 1: return vec3(1.0, 1.0, 0.0); // Yellow
        case 2: return vec3(0.0, 1.0, 1.0); // Cyan
        case 3: return vec3(0.0, 1.0, 0.0); // Green
        case 4: return vec3(1.0, 0.0, 1.0); // Magenta
        case 5: return vec3(1.0, 0.0, 0.0); // Red
        case 6: return vec3(0.0, 0.0, 1.0); // Blue
        default: return vec3(0.0, 0.0, 0.0); // Black
    }
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    if (pixel.x >= resolution.x || pixel.y >= resolution.y) {
        return;
    }

    vec2 uv = vec2(pixel) / vec2(resolution);
    vec3 color = vec3(0.0);

    switch(pattern) {
        case 0: // Color Bars
        {
            int barIndex = int(uv.x * 8.0);
            color = colorBar(barIndex);
            break;
        }

        case 1: // Horizontal Gradient
        {
            color = vec3(uv.x);
            break;
        }

        case 2: // Vertical Gradient
        {
            color = vec3(uv.y);
            break;
        }

        case 3: // Checkerboard
        {
            float checkSize = 32.0 / scale;
            vec2 check = floor(vec2(pixel) / checkSize);
            float c = mod(check.x + check.y, 2.0);
            color = vec3(c);
            break;
        }

        case 4: // Zone Plate (frequency test)
        {
            vec2 center = uv - 0.5;
            float r = length(center) * scale * 20.0;
            float v = 0.5 + 0.5 * sin(r * r);
            color = vec3(v);
            break;
        }

        case 5: // RGB Sweep
        {
            color = vec3(
                uv.x,                           // Red: left to right
                uv.y,                           // Green: bottom to top
                1.0 - (uv.x + uv.y) * 0.5       // Blue: diagonal
            );
            break;
        }
    }

    imageStore(output_image, pixel, vec4(color, 1.0));
}
