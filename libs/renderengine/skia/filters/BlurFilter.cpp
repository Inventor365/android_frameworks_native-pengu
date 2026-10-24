/*
 * Copyright 2021 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ATRACE_TAG ATRACE_TAG_GRAPHICS
#include "BlurFilter.h"
#include <SkBlendMode.h>
#include <SkM44.h>
#include <SkCanvas.h>
#include <SkPaint.h>
#include <SkRRect.h>
#include <SkRuntimeEffect.h>
#include <SkSize.h>
#include <SkString.h>
#include <SkSurface.h>
#include <SkTileMode.h>
#include <common/ThreadStateCrashLogger.h>
#include <common/trace.h>
#include <log/log.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>

#include "RuntimeEffectManager.h"

namespace android {
namespace renderengine {
namespace skia {

const SkString kEffectSource_BlurFilter_MixEffect(R"(
    uniform shader blurredInput;
    uniform shader originalInput;
    uniform float mixFactor;

    half4 main(float2 xy) {
        return half4(mix(originalInput.eval(xy), blurredInput.eval(xy), mixFactor)).rgb1;
    }
)");

// PenguinOS Liquid Glass & iOS Frosted Blur.
// In Liquid Glass mode (style 2): a rounded blur region is drawn as a curved optical lens slab
// with smooth rim refraction, subtle chromatic dispersion, vivid backdrop saturation, and a
// directional specular rim highlight.
// In iOS Frosted Blur mode (style 1): Apple UIVisualEffectView vibrancy with lush color
// saturation, deep dark-mode contrast preservation, and a soft satin frost sheen.
const SkString kEffectSource_LiquidGlassEffect(R"(
    uniform shader blurredInput;
    uniform shader originalInput;
    uniform float clarity;
    uniform float4 bounds;
    uniform float radius;
    uniform float band;
    uniform float refraction;
    uniform float dispersion;
    uniform float rim;
    uniform float saturation;
    uniform float gain;
    uniform float lift;
    uniform float edgeClear;
    uniform float curve;

    float roundRect(float2 p, float2 c, float2 h, float r) {
        float2 q = abs(p - c) - (h - r);
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    half4 main(float2 xy) {
        float2 c = (bounds.xy + bounds.zw) * 0.5;
        float2 h = max((bounds.zw - bounds.xy) * 0.5, float2(1.0));
        float r = min(radius, min(h.x, h.y));
        float d = roundRect(xy, c, h, r);

        // Smooth C1-continuous outward normal across the entire refraction band without
        // diagonal corner creases.
        float2 rel = xy - c;
        float rNorm = min(max(r, band), min(h.x, h.y));
        float2 qNorm = max(abs(rel) - (h - rNorm), 0.0);
        float qLen = length(qNorm);
        float2 n = qLen > 1e-4
                ? (qNorm / qLen)
                : (abs(rel.x) * h.y > abs(rel.y) * h.x ? float2(1.0, 0.0) : float2(0.0, 1.0));
        n *= float2(rel.x < 0.0 ? -1.0 : 1.0, rel.y < 0.0 ? -1.0 : 1.0);

        // 0 across the flat middle, 1 at the very edge.
        float t = clamp(1.0 + d / max(band, 1.0), 0.0, 1.0);
        // Smooth lens profile: dome curve steepest right at the edge where glass curves away.
        float dome = 1.0 - sqrt(max(1.0 - t * t, 0.0));
        float bend = refraction * mix(t * t * t, dome, curve);
        float2 offset = n * bend;
        float2 src = xy - offset;

        // Frosted across the middle so content stays legible, clearer towards the curved rim.
        float clear = clamp(clarity + (1.0 - clarity) * edgeClear * t * t, 0.0, 1.0);

        half3 rgb;
        if (dispersion > 0.001 && bend > 0.05) {
            float2 disp = offset * dispersion;
            half rCh = blurredInput.eval(src + disp).r;
            half gCh = blurredInput.eval(src).g;
            half bCh = blurredInput.eval(src - disp).b;
            rgb = half3(rCh, gCh, bCh);
            if (clear > 0.002) {
                half rOrig = originalInput.eval(src + disp).r;
                half gOrig = originalInput.eval(src).g;
                half bOrig = originalInput.eval(src - disp).b;
                rgb = mix(rgb, half3(rOrig, gOrig, bOrig), half(clear));
            }
        } else {
            rgb = blurredInput.eval(src).rgb;
            if (clear > 0.002) {
                rgb = mix(rgb, originalInput.eval(src).rgb, half(clear));
            }
        }

        half l = dot(rgb, half3(0.2126, 0.7152, 0.0722));
        // Vibrancy saturation boost: keep deep shadows clean while boosting midtone/highlight chroma
        half satWeight = half(saturation);
        rgb = mix(half3(l), rgb, satWeight);
        rgb = rgb * half(gain) + half(lift);

        if (rim > 0.001) {
            float facing = dot(n, normalize(float2(-0.6, -1.0)));
            float edge = pow(t, 5.0);
            float innerGlow = pow(t, 2.2) * (1.0 - edge) * 0.18;
            rgb += half(rim * (edge * (0.58 + 0.42 * facing) + innerGlow * max(facing, 0.0)));
            rgb += half(rim * 0.38 * edge * max(-facing, 0.0));
        }
        return half4(clamp(rgb, 0.0, 1.0), 1.0);
    }
)");

static constexpr const char* kLiquidGlassProperty = "persist.sys.penguin.liquid_glass";
static constexpr const char* kBlurStyleProperty = "persist.sys.penguin.blur_style";
// The names of the screens running a DeX desktop, separated by '|', which have Liquid Glass even
// with the system switch off.
static constexpr const char* kDexGlassProperty = "sys.penguin.liquid_glass.dex";

// A system property's value, re-read only when it changes.
class CachedProperty {
public:
    explicit CachedProperty(const char* name) : mName(name) {}

    const std::string& get() {
        if (mInfo == nullptr) {
            const uint32_t area = __system_property_area_serial();
            if (area == mAreaSerial) return mValue;
            mAreaSerial = area;
            mInfo = __system_property_find(mName);
            if (mInfo == nullptr) return mValue;
        }
        const uint32_t serial = __system_property_serial(mInfo);
        if (serial != mSerial) {
            mSerial = serial;
            __system_property_read_callback(
                    mInfo,
                    [](void* cookie, const char*, const char* value, uint32_t) {
                        *static_cast<std::string*>(cookie) = value;
                    },
                    &mValue);
        }
        return mValue;
    }

private:
    const char* mName;
    const prop_info* mInfo = nullptr;
    uint32_t mAreaSerial = UINT32_MAX;
    uint32_t mSerial = UINT32_MAX;
    std::string mValue;
};

// Returns active blur style for the display:
// 0 = Standard Blur, 1 = iOS Frosted Blur, 2 = iOS Liquid Glass
static int blurStyleFor(const std::string& namePlusId) {
    static CachedProperty styleProp(kBlurStyleProperty);
    static CachedProperty global(kLiquidGlassProperty);
    static CachedProperty dex(kDexGlassProperty);

    const std::string& styleStr = styleProp.get();
    const std::string& on = global.get();
    const bool lgOn = (on == "1" || on == "true");

    if (styleStr == "2") return 2;
    if (styleStr == "1") return lgOn ? 2 : 1;
    if (lgOn) return 2;
    if (styleStr == "0") return 0;

    const std::string& names = dex.get();
    size_t start = 0;
    while (start < names.size()) {
        size_t end = names.find('|', start);
        if (end == std::string::npos) end = names.size();
        if (end > start) {
            // Compositing names the display alone; screenshots put "ScreenCapture, " first.
            const std::string name = names.substr(start, end - start) + " (";
            if (namePlusId.compare(0, name.size(), name) == 0 ||
                namePlusId.find(", " + name) != std::string::npos) {
                return 2;
            }
        }
        start = end + 1;
    }
    // Default to iOS Frosted Blur (1) when no explicit override disables it, or 0 if standard
    return 0;
}

// A Liquid Glass value overridden live from debug.lg.sf.<name>, for matching the material by eye.
static float tune(const char* name, float fallback) {
    static std::mutex lock;
    static std::map<std::string, std::unique_ptr<CachedProperty>> props;
    std::lock_guard<std::mutex> guard(lock);
    auto& prop = props[name];
    if (prop == nullptr) {
        prop = std::make_unique<CachedProperty>(
                strdup((std::string("debug.lg.sf.") + name).c_str()));
    }
    const std::string& value = prop->get();
    if (value.empty()) return fallback;
    char* end = nullptr;
    const float parsed = strtof(value.c_str(), &end);
    return end != value.c_str() ? parsed : fallback;
}

void BlurFilter::setDisplay(const std::string& namePlusId) {
    mBlurStyle = mLiquidGlassEffect != nullptr ? blurStyleFor(namePlusId) : 0;
}

static SkMatrix getShaderTransform(const SkCanvas* canvas, const SkRect& blurRect,
                                   const float scale, const float zoomScale) {
    // 1. Apply the blur shader matrix, which scales up the blurred surface to its real size
    auto matrix = SkMatrix::Scale(scale, scale);
    // 2. Since the blurred surface has the size of the layer, we align it with the
    // top left corner of the layer position.
    matrix.postConcat(SkMatrix::Translate(blurRect.fLeft, blurRect.fTop));
    // 3. Apply the "zoom" effect as an extra scale + translate around the center of the blur.
    if (zoomScale != 1.0f) {
        matrix.postScale(zoomScale, zoomScale);
        matrix.postTranslate(
                blurRect.width() * (1 - zoomScale) / 2.0f,
                blurRect.height() * (1 - zoomScale) / 2.0f);
    }
    // 4. Finally, apply the inverse canvas matrix. The snapshot made in the BlurFilter is in the
    // original surface orientation. The inverse matrix has to be applied to align the blur
    // surface with the current orientation/position of the canvas.
    SkMatrix drawInverse;
    if (canvas != nullptr && canvas->getTotalMatrix().invert(&drawInverse)) {
        matrix.postConcat(drawInverse);
    }
    return matrix;
}

BlurFilter::BlurFilter(RuntimeEffectManager& effectManager, const float maxCrossFadeRadius)
      : mMaxCrossFadeRadius(maxCrossFadeRadius),
        mMixEffect(effectManager.mKnownEffects[kBlurFilter_MixEffect]),
        mLiquidGlassEffect(effectManager.mKnownEffects[kLiquidGlassEffect]) {}

float BlurFilter::getMaxCrossFadeRadius() const {
    return mMaxCrossFadeRadius;
}

void BlurFilter::drawBlurRegion(SkCanvas* canvas, const SkRRect& effectRegion,
                                const uint32_t blurRadius, const float zoomScale,
                                const float blurAlpha,
                                const SkRect& blurRect, sk_sp<SkImage> blurredImage,
                                sk_sp<SkImage> input) {
    SFTRACE_CALL();

    SkPaint paint;
    paint.setAlphaf(blurAlpha);

    auto blurMatrix = getShaderTransform(canvas, blurRect, kInverseInputScale, zoomScale);

    SkSamplingOptions linearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);
    const auto blurShader = blurredImage->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                                     linearSampling, &blurMatrix);

    // Liquid Glass (style 2) & iOS Frosted Blur (style 1):
    // Rounded regions become glass/frosted slabs; whole-window blurs, such as the shade's,
    // keep the colours behind them vivid instead of greying them out, as iOS's materials do.
    // Not while a blur is still fading in, which needs the crossfade below.
    if (mBlurStyle > 0 && (!effectRegion.isRect() || blurRadius >= mMaxCrossFadeRadius)) {
        const bool isLiquidGlass = (mBlurStyle == 2);
        const bool slab = !effectRegion.isRect();
        const SkRect& rect = effectRegion.rect();
        const float radius = slab ? effectRegion.radii(SkRRect::kUpperLeft_Corner).fX : 0.0f;
        const float shortSide = std::min(rect.width(), rect.height());
        // The curved rim is as deep as the corner, but never past the middle of the glass.
        const float band =
                slab ? std::clamp(radius * tune("band", 0.65f), 6.0f, shortSide * 0.5f) : 1.0f;
        SkRuntimeShaderBuilder glass(mLiquidGlassEffect);
        glass.child("blurredInput") = blurShader;
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        glass.child("originalInput") =
                input != nullptr
                        ? input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                            linearSampling, inputMatrix)
                        : blurShader;
        glass.uniform("clarity") = isLiquidGlass ? tune("clarity", 0.0f) : 0.0f;
        glass.uniform("bounds") = SkV4{rect.fLeft, rect.fTop, rect.fRight, rect.fBottom};
        glass.uniform("radius") = radius;
        glass.uniform("band") = band;
        glass.uniform("refraction") =
                (slab && isLiquidGlass) ? band * tune("refraction", 0.55f) : 0.0f;
        glass.uniform("dispersion") =
                (slab && isLiquidGlass) ? tune("dispersion", 0.06f) : 0.0f;
        glass.uniform("rim") =
                slab ? tune("rim", isLiquidGlass ? 0.16f : 0.07f) : 0.0f;
        // Both slab and whole-window backdrops get iOS's vivid vibrancy saturation boost
        // without lifting black levels so Dark Mode notifications stay high-contrast and crisp.
        const float defaultSlabSat = isLiquidGlass ? 1.24f : 1.35f;
        const float defaultWinSat = isLiquidGlass ? 1.22f : 1.30f;
        glass.uniform("saturation") =
                slab ? tune("slab_sat", defaultSlabSat) : tune("window_sat", defaultWinSat);
        glass.uniform("gain") =
                slab ? tune("slab_gain", isLiquidGlass ? 0.96f : 0.98f)
                     : tune("window_gain", 0.96f);
        glass.uniform("lift") =
                slab ? tune("slab_lift", 0.0f) : tune("window_lift", 0.0f);
        glass.uniform("edgeClear") = isLiquidGlass ? tune("edge_clear", 0.10f) : 0.0f;
        glass.uniform("curve") = tune("curve", 1.0f);
        paint.setShader(glass.makeShader());
        if (slab) {
            paint.setAntiAlias(true);
            canvas->drawRRect(effectRegion, paint);
        } else {
            if (blurAlpha == 1.0f) {
                paint.setBlendMode(SkBlendMode::kSrc);
            }
            canvas->drawRect(rect, paint);
        }
        return;
    }

    if (blurRadius < mMaxCrossFadeRadius) {
        LOG_THREAD_STATE_AND_CRASH_IF(!input);

        // For sampling Skia's API expects the inverse of what logically seems appropriate. In this
        // case you might expect the matrix to simply be the canvas matrix.
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        if (zoomScale != 1.0f) {
            inputMatrix.preTranslate(
                    blurRect.width() * (1 - zoomScale) / 2.0f,
                    blurRect.height() * (1 - zoomScale) / 2.0f);
            inputMatrix.preScale(zoomScale, zoomScale);
        }

        SkRuntimeShaderBuilder blurBuilder(mMixEffect);
        blurBuilder.child("blurredInput") = blurShader;
        blurBuilder.child("originalInput") =
                input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                                  inputMatrix);

        blurBuilder.uniform("mixFactor") = blurRadius / mMaxCrossFadeRadius;

        paint.setShader(blurBuilder.makeShader());
    } else {
        paint.setShader(blurShader);
    }

    if (effectRegion.isRect()) {
        if (blurAlpha == 1.0f) {
            paint.setBlendMode(SkBlendMode::kSrc);
        }
        canvas->drawRect(effectRegion.rect(), paint);
    } else {
        paint.setAntiAlias(true);
        canvas->drawRRect(effectRegion, paint);
    }
}

} // namespace skia
} // namespace renderengine
} // namespace android
