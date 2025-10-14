
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


/**
 * Анизотропный Флойда‑Штейнберга: усиливаем распространение ошибки вдоль касательной (tx,ty),
 * ослабляем поперёк. along=1.0, across≈0.3–0.5. Весами модифицируем стандартные FS-коэффициенты.
 */
fun ditherFsAniso(
    input: Raster,
    allowedLab: FloatArray,
    allowedArgb: IntArray,
    metric: Metric,
    tx: FloatArray,
    ty: FloatArray,
    along: Float,
    across: Float
    ): Raster {
    val w = input.width; val h = input.height
    val lab = argbToOkLab(input.argb)
    val out = IntArray(input.argb.size)
    fun idx(x: Int, y: Int): Int = y * w + x
    fun idx3(x: Int, y: Int): Int = (y * w + x) * 3
    fun addErr(x: Int, y: Int, el: Float, ea: Float, eb: Float, weight: Float) {
        if (x < 0 || x >= w || y < 0 || y >= h) return
        val i = idx3(x, y)
        lab[i + 0] += el * weight
        lab[i + 1] += ea * weight
        lab[i + 2] += eb * weight
    }
    val invSqrt2 = 0.70710677f
    for (y in 0 until h) {
        for (x in 0 until w) {
            val i3 = idx3(x, y)
            val L = lab[i3]; val A = lab[i3 + 1]; val B = lab[i3 + 2]
            val ai = nearestAllowedIndex(L, A, B, allowedLab, metric)
            val qL = allowedLab[ai * 3 + 0]
            val qA = allowedLab[ai * 3 + 1]
            val qB = allowedLab[ai * 3 + 2]
            val eL = L - qL; val eA = A - qA; val eB = B - qB
            out[idx(x, y)] = allowedArgb[ai]

            // Анизотропные множители по направлению касательной
            val tX = tx[idx(x, y)]
            val tY = ty[idx(x, y)]
            fun factor(dirX: Float, dirY: Float): Float {
                val dot = kotlin.math.abs(tX * dirX + tY * dirY) // [0..1]
                return across + (along - across) * dot
            }
            // Базовые направления FS
            val f10 = 7f / 16f * factor(1f, 0f)           // ( +1,  0)
            val f_11= 3f / 16f * factor(-invSqrt2, invSqrt2) // ( -1, +1) нормировано
            val f01 = 5f / 16f * factor(0f, 1f)           // (  0, +1)
            val f11 = 1f / 16f * factor(invSqrt2, invSqrt2)  // ( +1, +1) нормировано
            val sum = f10 + f_11 + f01 + f11
            if (sum <= 1e-6f) {
                // fallback: классический FS
                addErr(x + 1, y + 0, eL, eA, eB, 7f / 16f)
                addErr(x - 1, y + 1, eL, eA, eB, 3f / 16f)
                addErr(x + 0, y + 1, eL, eA, eB, 5f / 16f)
                addErr(x + 1, y + 1, eL, eA, eB, 1f / 16f)
            } else {
                val s = 1f / sum
                addErr(x + 1, y + 0, eL, eA, eB, f10 * s)
                addErr(x - 1, y + 1, eL, eA, eB, f_11 * s)
                addErr(x + 0, y + 1, eL, eA, eB, f01 * s)
                addErr(x + 1, y + 1, eL, eA, eB, f11 * s)
            }
        }
    }
    return Raster(w, h, out)
}