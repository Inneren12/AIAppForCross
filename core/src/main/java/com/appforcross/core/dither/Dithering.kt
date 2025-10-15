
package com.appforcross.core.dither

import com.appforcross.core.color.Metric
import com.appforcross.core.color.argbToOkLab
import com.appforcross.core.image.Raster

enum class Dither { NONE, FLOYD_STEINBERG, ATKINSON }

private fun nearestAllowedIndex(l: Float, a: Float, b: Float, allowedLab: FloatArray, metric: Metric): Int {
    var best = 0
    var bestD = Float.POSITIVE_INFINITY
    var i = 0
    while (i < allowedLab.size) {
        val dl = allowedLab[i] - l
        val da = allowedLab[i + 1] - a
        val db = allowedLab[i + 2] - b
        val d = dl * dl + da * da + db * db // метрика — L2 в используемом пространстве
        if (d < bestD) { bestD = d; best = i / 3 }
        i += 3
    }
    return best
}

fun ditherFs(input: Raster, allowedLab: FloatArray, allowedArgb: IntArray, metric: Metric): Raster {
    val w = input.width; val h = input.height
    val lab = argbToOkLab(input.argb)
    val out = IntArray(input.argb.size)
    // Проходим сверху-вниз, слева-направо. Ошибку распределяем в OKLab.
    fun idx3(x: Int, y: Int): Int = (y * w + x) * 3
    fun addErr(x: Int, y: Int, el: Float, ea: Float, eb: Float, weight: Float) {
        if (x < 0 || x >= w || y < 0 || y >= h) return
        val i = idx3(x, y)
        lab[i + 0] += el * weight
        lab[i + 1] += ea * weight
        lab[i + 2] += eb * weight
    }
    for (y in 0 until h) {
        for (x in 0 until w) {
            val i3 = idx3(x, y)
            val L = lab[i3]; val A = lab[i3 + 1]; val B = lab[i3 + 2]
            val ai = nearestAllowedIndex(L, A, B, allowedLab, metric)
            val qL = allowedLab[ai * 3 + 0]
            val qA = allowedLab[ai * 3 + 1]
            val qB = allowedLab[ai * 3 + 2]
            val eL = L - qL; val eA = A - qA; val eB = B - qB
            out[y * w + x] = allowedArgb[ai]
            // Распределение ошибки FS:
            addErr(x + 1, y + 0, eL, eA, eB, 7f / 16f)
            addErr(x - 1, y + 1, eL, eA, eB, 3f / 16f)
            addErr(x + 0, y + 1, eL, eA, eB, 5f / 16f)
            addErr(x + 1, y + 1, eL, eA, eB, 1f / 16f)
        }
    }
    return Raster(w, h, out)
}

fun ditherAtkinson(input: Raster, allowedLab: FloatArray, allowedArgb: IntArray, metric: Metric): Raster {
    val w = input.width; val h = input.height
    val lab = argbToOkLab(input.argb)
    val out = IntArray(input.argb.size)
    fun idx3(x: Int, y: Int): Int = (y * w + x) * 3
    fun addErr(x: Int, y: Int, el: Float, ea: Float, eb: Float, weight: Float) {
        if (x < 0 || x >= w || y < 0 || y >= h) return
        val i = idx3(x, y)
        lab[i + 0] += el * weight
        lab[i + 1] += ea * weight
        lab[i + 2] += eb * weight
    }
    for (y in 0 until h) {
        for (x in 0 until w) {
            val i3 = idx3(x, y)
            val L = lab[i3]; val A = lab[i3 + 1]; val B = lab[i3 + 2]
            val ai = nearestAllowedIndex(L, A, B, allowedLab, metric)
            val qL = allowedLab[ai * 3 + 0]
            val qA = allowedLab[ai * 3 + 1]
            val qB = allowedLab[ai * 3 + 2]
            val eL = L - qL; val eA = A - qA; val eB = B - qB
            out[y * w + x] = allowedArgb[ai]
            val w8 = 1f / 8f
            addErr(x + 1, y + 0, eL, eA, eB, w8)
            addErr(x + 2, y + 0, eL, eA, eB, w8)
            addErr(x - 1, y + 1, eL, eA, eB, w8)
            addErr(x + 0, y + 1, eL, eA, eB, w8)
            addErr(x + 1, y + 1, eL, eA, eB, w8)
            addErr(x + 0, y + 2, eL, eA, eB, w8)
        }
    }
    return Raster(w, h, out)
}

fun dither(input: Raster, allowedLab: FloatArray, allowedArgb: IntArray, metric: Metric, algo: Dither): Raster =
    when (algo) {
        Dither.NONE -> input
        Dither.FLOYD_STEINBERG -> ditherFs(input, allowedLab, allowedArgb, metric)
        Dither.ATKINSON -> ditherAtkinson(input, allowedLab, allowedArgb, metric)
    }
