package com.appforcross.core.metrics

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.cbrt
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.math.pow

/**
 * Оценка «banding» — доля пикселей, где плавный градиент стал плоским («лестницей»)
 * после квантования: окно 7×7 в L* (или яркости), правило:
 *  σ_orig ≥ 2.0  &&  σ_quant ≤ 0.6  &&  |μ_orig - μ_quant| ≤ 3
 *
 * Возвращает долю [0..1].
 **/
object BandingMeter {

    fun bandingScore(orig: Bitmap, quant: Bitmap, win: Int = 7): Float {
        val o = if (orig.width == quant.width && orig.height == quant.height) orig
        else Downscale.toWidth(quant, orig.width).let { _ -> orig }
        val q = if (quant.width == orig.width && quant.height == orig.height) quant
        else Downscale.toWidth(quant, orig.width)

        val w = o.width; val h = o.height
        if (w < win || h < win) return 0f

        val lo = luma(o) // 0..100 (L*)
        val lq = luma(q)
        val intO  = integral(lo, w, h)
        val intO2 = integralSquared(lo, w, h)
        val intQ  = integral(lq, w, h)
        val intQ2 = integralSquared(lq, w, h)

        val r = win / 2
        var flagged = 0
        var total = 0

        var y = r
        while (y < h - r) {
            var x = r
            while (x < w - r) {
                val x0 = x - r; val y0 = y - r; val x1 = x + r; val y1 = y + r
                val area = win * win
                val sumO  = rectSum(intO,  w, x0, y0, x1, y1); val muO = sumO / area
                val sumQ  = rectSum(intQ,  w, x0, y0, x1, y1); val muQ = sumQ / area
                val varO = (rectSum(intO2, w, x0, y0, x1, y1) / area) - muO * muO
                val varQ = (rectSum(intQ2, w, x0, y0, x1, y1) / area) - muQ * muQ
                val sigmaO = sqrt(max(0f, varO))
                val sigmaQ = sqrt(max(0f, varQ))

                val cond = (sigmaO >= 2.0f) && (sigmaQ <= 0.6f) && (abs(muO - muQ) <= 3.0f)
                if (cond) flagged++
                total++
                x += 1
            }
            y += 1
        }
        if (total == 0) return 0f
        return flagged.toFloat() / total
    }

    // ---------- internals ----------

    /** L* яркость (около 0..100). Для стабильности берём CIELab L*. */
    private fun luma(src: Bitmap): FloatArray {
        val w = src.width; val h = src.height
        val px = IntArray(w * h)
        src.getPixels(px, 0, w, 0, 0, w, h)
        val out = FloatArray(px.size)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            // sRGB -> linear
            fun srgbToLinear(u: Int): Double {
                val s = u / 255.0
                return if (s <= 0.04045) s / 12.92 else ((s + 0.055) / 1.055).pow(2.4)
            }
            val rl = srgbToLinear(r); val gl = srgbToLinear(g); val bl = srgbToLinear(b)
            // linear RGB -> Y (XYZ)
            val Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
            // Y -> L*
            val e = 216.0 / 24389.0
            val k = 24389.0 / 27.0
            val fy = if (Y > e) cbrt(Y) else (k * Y + 16.0) / 116.0
            val L = (116.0 * fy - 16.0).toFloat()
            out[i] = L
            i++
        }
        return out
    }

    private fun integral(src: FloatArray, w: Int, h: Int): FloatArray {
        val out = FloatArray((w + 1) * (h + 1))
        var y = 1
        var idx = 0
        while (y <= h) {
            var row = 0f
            var x = 1
            while (x <= w) {
                row += src[idx++]
                out[y * (w + 1) + x] = out[(y - 1) * (w + 1) + x] + row
                x++
            }
            y++
        }
        return out
    }

    private fun integralSquared(src: FloatArray, w: Int, h: Int): FloatArray {
        val out = FloatArray((w + 1) * (h + 1))
        var y = 1
        var idx = 0
        while (y <= h) {
            var row = 0f
            var x = 1
            while (x <= w) {
                val v = src[idx++]
                row += v * v
                out[y * (w + 1) + x] = out[(y - 1) * (w + 1) + x] + row
                x++
            }
            y++
        }
        return out
    }

    private fun rectSum(intImg: FloatArray, w: Int, x0: Int, y0: Int, x1: Int, y1: Int): Float {
        val W = w + 1
        val a = intImg[y0 * W + x0]
        val b = intImg[y0 * W + (x1 + 1)]
        val c = intImg[(y1 + 1) * W + x0]
        val d = intImg[(y1 + 1) * W + (x1 + 1)]
        return d - b - c + a
    }
}
