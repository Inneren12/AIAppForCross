package com.appforcross.core.metrics

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min

/**
 * Быстрый SSIM по яркости (Luma/L*) с оконной оценкой (box window).
 * Реализация через интегральные суммы: скользящее окно win×win с шагом win.
 * Возвращает среднее SSIM по окнам [0..1].
 * */
object SsimMeter {

    /**
     * Вычислить SSIM для [orig] vs [test]. Битмапы должны быть одинакового размера.
     * Если размеры различаются — [test] будет приведён к размерам [orig] вне этого метода.
     */
    fun ssimY(orig: Bitmap, test: Bitmap, win: Int = 7): Float {
        require(win >= 3) { "win must be >= 3" }
        // Приводим тестовое изображение к ТОЧНО тем же размерам, что и у orig.
        val o = orig
        val t = if (test.width == o.width && test.height == o.height) {
            test
        } else {
                // выравниваем по обеим осям (а не только по ширине)
            Bitmap.createScaledBitmap(test, o.width, o.height, /*filter=*/true)
            }

        val w = o.width
        val h = o.height
        if (w < win || h < win) return 1f

        val xo = luma01(o)
        val xt = luma01(t)
        // интегральные суммы X, X^2 и XY для окон
        val intX  = integral(xo, w, h)
        val intX2 = integralSquared(xo, w, h)
        val intY  = integral(xt, w, h)          // размеры теперь гарантированно совпадают
        val intY2 = integralSquared(xt, w, h)
        val intXY = integralProduct(xo, xt, w, h)

        val step = win // дискретизация по окнам
        val area = win * win
        val c1 = 0.01f * 0.01f // L=1
        val c2 = 0.03f * 0.03f

        var sum = 0.0
        var cnt = 0
        var y = 0
        while (y + win <= h) {
            var x = 0
            while (x + win <= w) {
                val sumX  = rectSum(intX,  w, x, y, x + win - 1, y + win - 1)
                val sumY  = rectSum(intY,  w, x, y, x + win - 1, y + win - 1)
                val sumX2 = rectSum(intX2, w, x, y, x + win - 1, y + win - 1)
                val sumY2 = rectSum(intY2, w, x, y, x + win - 1, y + win - 1)
                val sumXY = rectSum(intXY, w, x, y, x + win - 1, y + win - 1)

                val mx = sumX / area
                val my = sumY / area
                val vx = (sumX2 / area) - mx * mx
                val vy = (sumY2 / area) - my * my
                val cov = (sumXY / area) - mx * my

                val num = (2f * mx * my + c1) * (2f * cov + c2)
                val den = (mx * mx + my * my + c1) * (vx + vy + c2)
                val ssim = if (den <= 0f) 1f else (num / den)
                    sum += ssim.toDouble()
                cnt++
                x += step
            }
            y += step
        }
        if (cnt == 0) return 1f
        val v = (sum / cnt).toFloat()
        return v.coerceIn(0f, 1f)
    }

    // ---------- internals ----------

    private fun luma01(bm: Bitmap): FloatArray {
        val w = bm.width; val h = bm.height
        val px = IntArray(w * h)
        bm.getPixels(px, 0, w, 0, 0, w, h)
        val out = FloatArray(px.size)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = (c) and 0xFF
            val y = 0.2126f * r + 0.7152f * g + 0.0722f * b
            out[i] = (y / 255f)
            i++
        }
        return out
    }

    private fun integral(src: FloatArray, w: Int, h: Int): FloatArray {
        val out = FloatArray((w + 1) * (h + 1))
        var y = 1
        var idx = 0
        while (y <= h) {
            var rowSum = 0f
            var x = 1
            while (x <= w) {
                rowSum += src[idx++]
                out[y * (w + 1) + x] = out[(y - 1) * (w + 1) + x] + rowSum
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
            var rowSum = 0f
            var x = 1
            while (x <= w) {
                val v = src[idx++]
                rowSum += v * v
                out[y * (w + 1) + x] = out[(y - 1) * (w + 1) + x] + rowSum
                x++
            }
            y++
        }
        return out
    }

    private fun integralProduct(a: FloatArray, b: FloatArray, w: Int, h: Int): FloatArray {
        val out = FloatArray((w + 1) * (h + 1))
        var y = 1
        var idx = 0
        while (y <= h) {
            var rowSum = 0f
            var x = 1
            while (x <= w) {
                rowSum += a[idx] * b[idx]
                idx++
                out[y * (w + 1) + x] = out[(y - 1) * (w + 1) + x] + rowSum
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

