#version 430

// GLIC Color Space Conversion
// All 16 color spaces from the original GLIC

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(rgba32f, binding = 0) readonly uniform image2D input_image;
layout(rgba32f, binding = 1) writeonly uniform image2D output_image;

uniform int colorspace;  // Target color space
uniform bool forward;    // true = RGB->CS, false = CS->RGB

const float PI = 3.14159265359;

// Color space constants
const int CS_RGB = 0;
const int CS_OHTA = 1;
const int CS_CMY = 2;
const int CS_HSB = 3;
const int CS_XYZ = 4;
const int CS_YXY = 5;
const int CS_HCL = 6;
const int CS_LUV = 7;
const int CS_LAB = 8;
const int CS_HWB = 9;
const int CS_RGGBG = 10;
const int CS_YPbPr = 11;
const int CS_YCbCr = 12;
const int CS_YDbDr = 13;
const int CS_GS = 14;
const int CS_YUV = 15;

// Helper: get luma
float getLuma(vec3 c) {
    return clamp(0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b, 0.0, 1.0);
}

// ============ OHTA ============
vec3 toOHTA(vec3 c) {
    float I1 = 0.33333 * c.r + 0.33334 * c.g + 0.33333 * c.b;
    float I2 = 0.5 * (c.r - c.b) + 0.5;
    float I3 = (-0.25 * c.r + 0.5 * c.g - 0.25 * c.b) + 0.5;
    return vec3(I1, I2, I3);
}

vec3 fromOHTA(vec3 c) {
    float I1 = c.r;
    float I2 = (c.g - 0.5) * 2.0;
    float I3 = (c.b - 0.5) * 2.0;
    float R = I1 + 1.0 * I2 - 0.66668 * I3;
    float G = I1 + 1.33333 * I3;
    float B = I1 - 1.0 * I2 - 0.66668 * I3;
    return clamp(vec3(R, G, B), 0.0, 1.0);
}

// ============ CMY ============
vec3 toCMY(vec3 c) {
    return 1.0 - c;
}

vec3 fromCMY(vec3 c) {
    return 1.0 - c;
}

// ============ HSB ============
vec3 toHSB(vec3 c) {
    float minC = min(c.r, min(c.g, c.b));
    float maxC = max(c.r, max(c.g, c.b));
    float delta = maxC - minC;

    float h = 0.0;
    float s = (maxC > 0.0) ? delta / maxC : 0.0;
    float b = maxC;

    if (delta > 0.0) {
        if (c.r == maxC) h = (c.g - c.b) / delta;
        else if (c.g == maxC) h = 2.0 + (c.b - c.r) / delta;
        else h = 4.0 + (c.r - c.g) / delta;
        h /= 6.0;
        if (h < 0.0) h += 1.0;
    }

    return vec3(h, s, b);
}

vec3 fromHSB(vec3 c) {
    if (c.g == 0.0) return vec3(c.b);

    float h = c.r * 6.0;
    float f = fract(h);
    float p = c.b * (1.0 - c.g);
    float q = c.b * (1.0 - c.g * f);
    float t = c.b * (1.0 - c.g * (1.0 - f));

    int i = int(h) % 6;
    if (i == 0) return vec3(c.b, t, p);
    if (i == 1) return vec3(q, c.b, p);
    if (i == 2) return vec3(p, c.b, t);
    if (i == 3) return vec3(p, q, c.b);
    if (i == 4) return vec3(t, p, c.b);
    return vec3(c.b, p, q);
}

// ============ HWB ============
vec3 toHWB(vec3 c) {
    float w = min(c.r, min(c.g, c.b));
    float v = max(c.r, max(c.g, c.b));

    float h;
    if (v == w) {
        h = 1.0;
    } else {
        float f = (c.r == w) ? c.g - c.b : ((c.g == w) ? c.b - c.r : c.r - c.g);
        float p = (c.r == w) ? 3.0 : ((c.g == w) ? 5.0 : 1.0);
        h = (p - f / (v - w)) / 6.0;
    }

    return vec3(h, w, 1.0 - v);
}

vec3 fromHWB(vec3 c) {
    float h = c.r;
    float w = c.g;
    float b = 1.0 - c.b;

    if (h >= 1.0) return vec3(b);

    float hue = h * 6.0;
    int i = int(hue);
    float f = fract(hue);
    if ((i & 1) != 0) f = 1.0 - f;
    float n = w + f * (b - w);

    if (i == 0) return vec3(b, n, w);
    if (i == 1) return vec3(n, b, w);
    if (i == 2) return vec3(w, b, n);
    if (i == 3) return vec3(w, n, b);
    if (i == 4) return vec3(n, w, b);
    return vec3(b, w, n);
}

// ============ YCbCr ============
vec3 toYCbCr(vec3 c) {
    float Y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    float Cb = -0.168736 * c.r - 0.331264 * c.g + 0.5 * c.b + 0.5;
    float Cr = 0.5 * c.r - 0.418688 * c.g - 0.081312 * c.b + 0.5;
    return vec3(Y, Cb, Cr);
}

vec3 fromYCbCr(vec3 c) {
    float Y = c.r;
    float Cb = c.g - 0.5;
    float Cr = c.b - 0.5;
    float R = Y + 1.402 * Cr;
    float G = Y - 0.344136 * Cb - 0.714136 * Cr;
    float B = Y + 1.772 * Cb;
    return clamp(vec3(R, G, B), 0.0, 1.0);
}

// ============ YUV ============
vec3 toYUV(vec3 c) {
    float Y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    float U = -0.14713 * c.r - 0.28886 * c.g + 0.436 * c.b;
    float V = 0.615 * c.r - 0.51499 * c.g - 0.10001 * c.b;
    return vec3(Y, U * 0.5 + 0.5, V * 0.5 + 0.5);
}

vec3 fromYUV(vec3 c) {
    float Y = c.r;
    float U = (c.g - 0.5) * 2.0;
    float V = (c.b - 0.5) * 2.0;
    float R = Y + 1.13983 * V;
    float G = Y - 0.39465 * U - 0.58060 * V;
    float B = Y + 2.03211 * U;
    return clamp(vec3(R, G, B), 0.0, 1.0);
}

// ============ YPbPr ============
vec3 toYPbPr(vec3 c) {
    float Y = getLuma(c);
    float Pb = c.b - Y;
    float Pr = c.r - Y;
    return vec3(Y, Pb * 0.5 + 0.5, Pr * 0.5 + 0.5);
}

vec3 fromYPbPr(vec3 c) {
    float Y = c.r;
    float Pb = (c.g - 0.5) * 2.0;
    float Pr = (c.b - 0.5) * 2.0;
    float B = Pb + Y;
    float R = Pr + Y;
    float G = (Y - 0.2126 * R - 0.0722 * B) / 0.7152;
    return clamp(vec3(R, G, B), 0.0, 1.0);
}

// ============ YDbDr ============
vec3 toYDbDr(vec3 c) {
    float Y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    float Db = (-0.450 * c.r - 0.883 * c.g + 1.333 * c.b) / 2.666 + 0.5;
    float Dr = (-1.333 * c.r + 1.116 * c.g + 0.217 * c.b) / 2.666 + 0.5;
    return vec3(Y, Db, Dr);
}

vec3 fromYDbDr(vec3 c) {
    float Y = c.r;
    float Db = (c.g - 0.5) * 2.666;
    float Dr = (c.b - 0.5) * 2.666;
    float R = Y - 0.526 * Dr;
    float G = Y - 0.129 * Db + 0.268 * Dr;
    float B = Y + 0.665 * Db;
    return clamp(vec3(R, G, B), 0.0, 1.0);
}

// ============ R-G,G,B-G ============
vec3 toRGGBG(vec3 c) {
    float rg = c.r - c.g;
    float bg = c.b - c.g;
    return vec3(rg * 0.5 + 0.5, c.g, bg * 0.5 + 0.5);
}

vec3 fromRGGBG(vec3 c) {
    float rg = (c.r - 0.5) * 2.0;
    float bg = (c.b - 0.5) * 2.0;
    return clamp(vec3(rg + c.g, c.g, bg + c.g), 0.0, 1.0);
}

// ============ Grayscale ============
vec3 toGS(vec3 c) {
    float l = getLuma(c);
    return vec3(l, l, l);
}

vec3 fromGS(vec3 c) {
    return c; // Already grayscale
}

// ============ XYZ (simplified) ============
vec3 toXYZ(vec3 c) {
    // Gamma correction
    vec3 lin = pow(c, vec3(2.2));
    float X = 0.4124 * lin.r + 0.3576 * lin.g + 0.1805 * lin.b;
    float Y = 0.2126 * lin.r + 0.7152 * lin.g + 0.0722 * lin.b;
    float Z = 0.0193 * lin.r + 0.1192 * lin.g + 0.9505 * lin.b;
    return vec3(X, Y, Z);
}

vec3 fromXYZ(vec3 c) {
    float R = 3.2406 * c.r - 1.5372 * c.g - 0.4986 * c.b;
    float G = -0.9689 * c.r + 1.8758 * c.g + 0.0415 * c.b;
    float B = 0.0557 * c.r - 0.2040 * c.g + 1.0570 * c.b;
    return pow(clamp(vec3(R, G, B), 0.0, 1.0), vec3(1.0/2.2));
}

// ============ HCL ============
vec3 toHCL(vec3 c) {
    float maxC = max(c.r, max(c.g, c.b));
    float minC = min(c.r, min(c.g, c.b));
    float chr = maxC - minC;

    float h = 0.0;
    if (chr > 0.0) {
        if (c.r == maxC) h = mod((c.g - c.b) / chr + 6.0, 6.0);
        else if (c.g == maxC) h = (c.b - c.r) / chr + 2.0;
        else h = (c.r - c.g) / chr + 4.0;
    }

    float l = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
    return vec3(h / 6.0, chr, l);
}

vec3 fromHCL(vec3 c) {
    float h = c.r * 6.0;
    float chr = c.g;
    float l = c.b;

    float x = chr * (1.0 - abs(mod(h, 2.0) - 1.0));
    vec3 rgb;

    if (h < 1.0) rgb = vec3(chr, x, 0.0);
    else if (h < 2.0) rgb = vec3(x, chr, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, chr, x);
    else if (h < 4.0) rgb = vec3(0.0, x, chr);
    else if (h < 5.0) rgb = vec3(x, 0.0, chr);
    else rgb = vec3(chr, 0.0, x);

    float m = l - (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b);
    return clamp(rgb + m, 0.0, 1.0);
}

// ============ LAB (simplified) ============
vec3 toLAB(vec3 c) {
    vec3 xyz = toXYZ(c);
    // D65 reference
    xyz /= vec3(0.95047, 1.0, 1.08883);

    vec3 f;
    for (int i = 0; i < 3; i++) {
        float v = xyz[i];
        f[i] = (v > 0.008856) ? pow(v, 1.0/3.0) : (7.787 * v + 16.0/116.0);
    }

    float L = 1.16 * f.y - 0.16;
    float a = 0.5 * (f.x - f.y) + 0.5;
    float b = 0.5 * (f.y - f.z) + 0.5;
    return vec3(L, a, b);
}

vec3 fromLAB(vec3 c) {
    float L = c.r;
    float a = (c.g - 0.5) * 2.0;
    float b = (c.b - 0.5) * 2.0;

    float y = (L + 0.16) / 1.16;
    float x = a + y;
    float z = y - b;

    vec3 xyz;
    xyz.x = (x > 0.206893) ? x * x * x : (x - 16.0/116.0) / 7.787;
    xyz.y = (y > 0.206893) ? y * y * y : (y - 16.0/116.0) / 7.787;
    xyz.z = (z > 0.206893) ? z * z * z : (z - 16.0/116.0) / 7.787;

    xyz *= vec3(0.95047, 1.0, 1.08883);
    return fromXYZ(xyz);
}

// ============ LUV (simplified) ============
vec3 toLUV(vec3 c) {
    vec3 xyz = toXYZ(c);
    float d = xyz.x + 15.0 * xyz.y + 3.0 * xyz.z;

    float L = (xyz.y > 0.008856) ? 1.16 * pow(xyz.y, 1.0/3.0) - 0.16 : 9.033 * xyz.y;

    float u = (d > 0.0) ? 4.0 * xyz.x / d : 0.0;
    float v = (d > 0.0) ? 9.0 * xyz.y / d : 0.0;

    // Reference white D65
    float un = 0.2009;
    float vn = 0.4610;

    float U = 13.0 * L * (u - un);
    float V = 13.0 * L * (v - vn);

    return vec3(L, U * 0.1 + 0.5, V * 0.1 + 0.5);
}

vec3 fromLUV(vec3 c) {
    float L = c.r;
    float U = (c.g - 0.5) * 10.0;
    float V = (c.b - 0.5) * 10.0;

    float un = 0.2009;
    float vn = 0.4610;

    float u = U / (13.0 * L + 0.0001) + un;
    float v = V / (13.0 * L + 0.0001) + vn;

    float Y = (L > 0.08) ? pow((L + 0.16) / 1.16, 3.0) : L / 9.033;
    float X = Y * 9.0 * u / (4.0 * v + 0.0001);
    float Z = Y * (12.0 - 3.0 * u - 20.0 * v) / (4.0 * v + 0.0001);

    return fromXYZ(vec3(X, Y, Z));
}

// ============ YXY ============
vec3 toYXY(vec3 c) {
    vec3 xyz = toXYZ(c);
    float sum = xyz.x + xyz.y + xyz.z + 0.0001;
    return vec3(xyz.y, xyz.x / sum, xyz.y / sum);
}

vec3 fromYXY(vec3 c) {
    float Y = c.r;
    float x = c.g;
    float y = c.b + 0.0001;

    float X = x * Y / y;
    float Z = (1.0 - x - y) * Y / y;

    return fromXYZ(vec3(X, Y, Z));
}

// Main conversion dispatcher
vec3 toColorspace(vec3 rgb, int cs) {
    switch(cs) {
        case CS_OHTA: return toOHTA(rgb);
        case CS_CMY: return toCMY(rgb);
        case CS_HSB: return toHSB(rgb);
        case CS_XYZ: return toXYZ(rgb);
        case CS_YXY: return toYXY(rgb);
        case CS_HCL: return toHCL(rgb);
        case CS_LUV: return toLUV(rgb);
        case CS_LAB: return toLAB(rgb);
        case CS_HWB: return toHWB(rgb);
        case CS_RGGBG: return toRGGBG(rgb);
        case CS_YPbPr: return toYPbPr(rgb);
        case CS_YCbCr: return toYCbCr(rgb);
        case CS_YDbDr: return toYDbDr(rgb);
        case CS_GS: return toGS(rgb);
        case CS_YUV: return toYUV(rgb);
        default: return rgb;
    }
}

vec3 fromColorspace(vec3 cs_color, int cs) {
    switch(cs) {
        case CS_OHTA: return fromOHTA(cs_color);
        case CS_CMY: return fromCMY(cs_color);
        case CS_HSB: return fromHSB(cs_color);
        case CS_XYZ: return fromXYZ(cs_color);
        case CS_YXY: return fromYXY(cs_color);
        case CS_HCL: return fromHCL(cs_color);
        case CS_LUV: return fromLUV(cs_color);
        case CS_LAB: return fromLAB(cs_color);
        case CS_HWB: return fromHWB(cs_color);
        case CS_RGGBG: return fromRGGBG(cs_color);
        case CS_YPbPr: return fromYPbPr(cs_color);
        case CS_YCbCr: return fromYCbCr(cs_color);
        case CS_YDbDr: return fromYDbDr(cs_color);
        case CS_GS: return fromGS(cs_color);
        case CS_YUV: return fromYUV(cs_color);
        default: return cs_color;
    }
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(input_image);

    if (pixel.x >= size.x || pixel.y >= size.y) {
        return;
    }

    vec4 color = imageLoad(input_image, pixel);
    vec3 rgb = color.rgb;

    vec3 result;
    if (forward) {
        result = toColorspace(rgb, colorspace);
    } else {
        result = fromColorspace(rgb, colorspace);
    }

    imageStore(output_image, pixel, vec4(result, color.a));
}
