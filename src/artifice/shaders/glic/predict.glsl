#version 430

// GLIC Prediction
// Predicts pixel values based on neighbors - the core of GLIC glitch effects

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int prediction_method;  // 0-15 prediction type
uniform int block_size;         // Block size for prediction (2-256)

// Prediction method constants (from GLIC)
const int PRED_NONE = 0;
const int PRED_CORNER = 1;      // Top-left corner
const int PRED_H = 2;           // Horizontal (left neighbor)
const int PRED_V = 3;           // Vertical (top neighbor)
const int PRED_DC = 4;          // DC/Mean of neighbors
const int PRED_DCMEDIAN = 5;    // DC median
const int PRED_MEDIAN = 6;      // Median of 3 neighbors
const int PRED_AVG = 7;         // Average of H and V
const int PRED_TRUEMOTION = 8;  // TrueMotion (H + V - corner)
const int PRED_PAETH = 9;       // PNG Paeth predictor
const int PRED_LDIAG = 10;      // Left diagonal
const int PRED_HV = 11;         // H or V based on position
const int PRED_JPEGLS = 12;     // JPEG-LS predictor
const int PRED_DIFF = 13;       // Difference predictor
const int PRED_GRADIENT = 14;   // Gradient predictor
const int PRED_SELECT = 15;     // Select best of H/V

// Safe image load with border handling
vec3 safeLoad(ivec2 pos, ivec2 size) {
    // Clamp to image bounds, use edge pixels for out-of-bounds
    pos = clamp(pos, ivec2(0), size - 1);
    return imageLoad(input_image, pos).rgb;
}

// Get median of 3 values (per channel)
vec3 median3(vec3 a, vec3 b, vec3 c) {
    return max(min(a, b), min(max(a, b), c));
}

// Paeth predictor (PNG algorithm)
vec3 paeth(vec3 a, vec3 b, vec3 c) {
    vec3 p = a + b - c;
    vec3 pa = abs(p - a);
    vec3 pb = abs(p - b);
    vec3 pc = abs(p - c);

    vec3 result;
    for (int i = 0; i < 3; i++) {
        if (pa[i] <= pb[i] && pa[i] <= pc[i]) result[i] = a[i];
        else if (pb[i] <= pc[i]) result[i] = b[i];
        else result[i] = c[i];
    }
    return result;
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }

    // Get neighbors
    vec3 left = safeLoad(pixel + ivec2(-1, 0), size);      // A (horizontal)
    vec3 top = safeLoad(pixel + ivec2(0, -1), size);       // B (vertical)
    vec3 corner = safeLoad(pixel + ivec2(-1, -1), size);   // C (top-left)
    vec3 topRight = safeLoad(pixel + ivec2(1, -1), size);  // D (top-right)

    // For DC prediction - average of left column and top row
    vec3 dc = vec3(0.0);
    if (prediction_method == PRED_DC || prediction_method == PRED_DCMEDIAN) {
        int count = 0;
        for (int i = 0; i < block_size && pixel.x - 1 >= 0; i++) {
            dc += safeLoad(pixel + ivec2(-1, i), size);
            count++;
        }
        for (int i = 0; i < block_size && pixel.y - 1 >= 0; i++) {
            dc += safeLoad(pixel + ivec2(i, -1), size);
            count++;
        }
        dc += corner;
        count++;
        dc /= float(max(count, 1));
    }

    vec3 prediction;

    switch (prediction_method) {
        case PRED_NONE:
            prediction = vec3(0.5); // Middle gray
            break;

        case PRED_CORNER:
            prediction = corner;
            break;

        case PRED_H:
            prediction = left;
            break;

        case PRED_V:
            prediction = top;
            break;

        case PRED_DC:
            prediction = dc;
            break;

        case PRED_DCMEDIAN:
            prediction = median3(dc, left, top);
            break;

        case PRED_MEDIAN:
            prediction = median3(corner, left, top);
            break;

        case PRED_AVG:
            prediction = (left + top) * 0.5;
            break;

        case PRED_TRUEMOTION:
            prediction = clamp(left + top - corner, 0.0, 1.0);
            break;

        case PRED_PAETH:
            prediction = paeth(left, top, corner);
            break;

        case PRED_LDIAG:
            {
                // Diagonal prediction weighted by position
                int x = pixel.x % block_size;
                int y = pixel.y % block_size;
                float wx = float(x + 1);
                float wy = float(y + 1);
                prediction = (wx * top + wy * left) / (wx + wy);
            }
            break;

        case PRED_HV:
            {
                // Use H or V based on diagonal position
                int x = pixel.x % block_size;
                int y = pixel.y % block_size;
                if (x > y) prediction = top;
                else if (y > x) prediction = left;
                else prediction = (top + left) * 0.5;
            }
            break;

        case PRED_JPEGLS:
            {
                // JPEG-LS predictor
                vec3 minAB = min(left, top);
                vec3 maxAB = max(left, top);
                for (int i = 0; i < 3; i++) {
                    if (corner[i] >= maxAB[i]) prediction[i] = minAB[i];
                    else if (corner[i] <= minAB[i]) prediction[i] = maxAB[i];
                    else prediction[i] = left[i] + top[i] - corner[i];
                }
            }
            break;

        case PRED_DIFF:
            {
                // Second-order difference predictor
                vec3 left2 = safeLoad(pixel + ivec2(-2, 0), size);
                vec3 top2 = safeLoad(pixel + ivec2(0, -2), size);
                prediction = clamp((2.0 * left - left2 + 2.0 * top - top2) * 0.5, 0.0, 1.0);
            }
            break;

        case PRED_GRADIENT:
            {
                // Gradient predictor
                vec3 gradH = left - corner;
                vec3 gradV = top - corner;
                prediction = clamp(corner + gradH + gradV, 0.0, 1.0);
            }
            break;

        case PRED_SELECT:
            {
                // Select H or V based on gradient
                vec3 gradH = abs(corner - left);
                vec3 gradV = abs(corner - top);
                for (int i = 0; i < 3; i++) {
                    prediction[i] = (gradH[i] < gradV[i]) ? top[i] : left[i];
                }
            }
            break;

        default:
            prediction = vec3(0.5);
    }

    vec4 original = imageLoad(input_image, pixel);
    imageStore(output_image, pixel, vec4(prediction, original.a));
}
