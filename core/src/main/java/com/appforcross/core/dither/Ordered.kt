package com.appforcross.core.dither

import com.appforcross.core.color.Metric
import com.appforcross.core.color.argbToOkLab
import kotlin.math.max
import kotlin.math.min
import com.appforcross.core.image.Raster

/**
 * Упорядоченный (Bayer 8x8) цветовой дизеринг в OKLab:
 * — порог добавляется как «делта» к L* (амплитуда amp в 0..~0.5)
 * — затем берём ближайший цвет из allowedLab/allowedArgb.
 */
fun ditherOrderedBayer8(
    input: Raster,
    allowedLab: FloatArray,
    allowedArgb: IntArray,
    amp: Float = 0.30f,
    metric: Metric = Metric.OKLAB
): Raster {
    val w = input.width; val h = input.height
    val lab = argbToOkLab(input.argb) // L,a,b per pixel (interleaved)
    val out = IntArray(input.argb.size)

    fun nearestIndex(L: Float, A: Float, B: Float): Int {
        var best = 0
        var bestD = Float.POSITIVE_INFINITY
        var i = 0
        while (i < allowedLab.size) {
            val dl = allowedLab[i] - L
            val da = allowedLab[i + 1] - A
            val db = allowedLab[i + 2] - B
            val d = dl*dl + da*da + db*db
            if (d < bestD) { bestD = d; best = i / 3 }
            i += 3
        }
        return best
    }

    var p = 0
    for (y in 0 until h) {
        val y8 = (y and 7) * 8
        for (x in 0 until w) {
            val i3 = p * 3
            var L = lab[i3 + 0]
            val A = lab[i3 + 1]
            val B = lab[i3 + 2]
            // порог [0..63] -> [-0.5..+0.5] -> * amp
            val t = ((BAYER8[y8 + (x and 7)] + 0.5f) / 64f - 0.5f) * amp
            L = clamp01(L + t)
            val ai = nearestIndex(L, A, B)
            out[p] = allowedArgb[ai]
            p++
        }
    }
    return Raster(w, h, out)
}

private fun clamp01(v: Float): Float = max(0f, min(1f, v))

/** Классическая 8x8 матрица Байера (значения 0..63). */
private val BAYER8 = intArrayOf(
    0, 48, 12, 60, 3, 51, 15, 63,
    32, 16, 44, 28, 35, 19, 47, 31,
    8, 56, 4, 52, 11, 59, 7, 55,
    40, 24, 36, 20, 43, 27, 39, 23,
    2, 50, 14, 62, 1, 49, 13, 61,
    34, 18, 46, 30, 33, 17, 45, 29,
    10, 58, 6, 54, 9, 57, 5, 53,
    42, 26, 38, 22, 41, 25, 37, 21
).map { it.toFloat() }.toFloatArray()