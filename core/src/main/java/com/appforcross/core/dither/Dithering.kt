
package com.appforcross.core.dither

import com.appforcross.core.color.Metric
import com.appforcross.core.color.argbToOkLab
import com.appforcross.core.image.Raster

enum class Dither { NONE, FLOYD_STEINBERG, ATKINSON }

// ===== ORDERED 8×8 constants (file-level) =====
// Standard Bayer 8×8 (0..63) and normalized [-0.5..+0.5]
private val BAYER_8x8: IntArray = intArrayOf(
    0, 48, 12, 60,  3, 51, 15, 63,
    32, 16, 44, 28, 35, 19, 47, 31,
    8, 56,  4, 52, 11, 59,  7, 55,
    40, 24, 36, 20, 43, 27, 39, 23,
    2, 50, 14, 62,  1, 49, 13, 61,
    34, 18, 46, 30, 33, 17, 45, 29,
    10, 58,  6, 54,  9, 57,  5, 53,
    42, 26, 38, 22, 41, 25, 37, 21
)
private val BAYER_8x8_NORM: FloatArray by lazy {
    FloatArray(64) { i -> ((BAYER_8x8[i].toFloat() + 0.5f) / 64f) - 0.5f }
}
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


// ───────────────────────── ORDERED 8×8 (Bayer) ─────────────────────────
/**
 * ORDERED 8×8 — константная амплитуда.
 */
fun ditherOrdered8x8(
    input: Raster,
    allowedLab: FloatArray,
    allowedArgb: IntArray,
    metric: Metric,
    amp: Float,
    mask: BooleanArray? = null
): Raster {
    return ditherOrdered8x8Provider(
        input, allowedLab, allowedArgb, metric,
        { _: Int, _: Int -> amp }, // типизированная лямбда
        mask
    )
}

/**
 * ORDERED 8×8 — амплитуда от провайдера (x,y)→amp.
 * Шум добавляем только в L-канал OKLab (чтобы не «цвели» блики).
 */
fun ditherOrdered8x8Provider(
    input: Raster,
    allowedLab: FloatArray,
    allowedArgb: IntArray,
    metric: Metric,
    ampProvider: (Int, Int) -> Float,
    mask: BooleanArray? = null
): Raster {
    val w = input.width
    val h = input.height
    val lab = argbToOkLab(input.argb)
    val out = IntArray(input.argb.size)
    fun idx3(p: Int): Int = p * 3
    var y = 0
    while (y < h) {
        val ry = (y and 7) * 8
        var x = 0
        while (x < w) {
            val p = y * w + x
            val i3 = idx3(p)
            val use = mask?.get(p) ?: true
            val ampV = if (use) ampProvider(x, y) else 0f
            val t = if (ampV > 0f) BAYER_8x8_NORM[ry + (x and 7)] else 0f
            val L = (lab[i3] + ampV * t).coerceIn(0f, 1f)
            val A = lab[i3 + 1]
            val B = lab[i3 + 2]
            val ai = nearestAllowedIndex(L, A, B, allowedLab, metric)
            out[p] = allowedArgb[ai]
            x++
        }
        y++
    }
    return Raster(w, h, out)
}

/**
 * ORDERED 8×8 по тайлам (обычно tile=8) с картой амплитуд wTiles×hTiles.
 */
fun ditherOrdered8x8Tiles(
    input: Raster,
    allowedLab: FloatArray,
    allowedArgb: IntArray,
    metric: Metric,
    ampTiles: FloatArray,
    wTiles: Int,
    hTiles: Int,
    tile: Int = 8,
    mask: BooleanArray? = null
): Raster {
    require(wTiles * hTiles == ampTiles.size) { "ampTiles size must be wTiles*hTiles" }
    val provider: (Int, Int) -> Float = { x: Int, y: Int ->
        val tx = kotlin.math.min(x / tile, wTiles - 1)
        val ty = kotlin.math.min(y / tile, hTiles - 1)
        ampTiles[ty * wTiles + tx]
    }
    return ditherOrdered8x8Provider(input, allowedLab, allowedArgb, metric, provider, mask)
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