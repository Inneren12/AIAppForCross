package com.appforcross.core.metrics

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.sqrt

/**
 * Возвращает подмножество цветов палитры, которые НЕ следует удалять/сливать
 * (они в основном "живут" на границах и в мелких деталях).
 * Работает по downscale‑изображению (≤512 px).
 */
object EdgeGuard {

    /**
     * @param ref512  исходник, downscale до ~512
     * @param palette палитра (ARGB)
     * @return цвета из палитры, которые стоит защитить от pruning/clamp
     */
    fun protectedColors(ref512: Bitmap, palette: IntArray): IntArray {
        if (palette.isEmpty() || ref512.width == 0 || ref512.height == 0) return intArrayOf()

        val w = ref512.width
        val h = ref512.height
        val size = w * h

        // Sobel по яркости
        val grad = FloatArray(size)
        fun lum(x: Int, y: Int): Double {
            val c = ref512.getPixel(x, y)
            return 0.299 * Color.red(c) + 0.587 * Color.green(c) + 0.114 * Color.blue(c)
        }
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val gx = -lum(x - 1, y - 1) - 2 * lum(x - 1, y) - lum(x - 1, y + 1) +
                        lum(x + 1, y - 1) + 2 * lum(x + 1, y) + lum(x + 1, y + 1)
                val gy = -lum(x - 1, y - 1) - 2 * lum(x, y - 1) - lum(x + 1, y - 1) +
                        lum(x - 1, y + 1) + 2 * lum(x, y + 1) + lum(x + 1, y + 1)
                grad[y * w + x] = sqrt(gx * gx + gy * gy).toFloat()
            }
        }

        // Порог ~90‑й перцентиль
        val sorted = grad.copyOf()
        sorted.sort()
        val thr = sorted[(sorted.size * 0.90f).toInt().coerceAtMost(sorted.size - 1)]

        // Мэппинг каждого пикселя к ближайшему цвету палитры и подсчёт статистики
        fun dist2(a: Int, b: Int): Int {
            val dr = Color.red(a) - Color.red(b)
            val dg = Color.green(a) - Color.green(b)
            val db = Color.blue(a) - Color.blue(b)
            return dr * dr + dg * dg + db * db
        }
        val total = IntArray(palette.size)
        val onEdge = IntArray(palette.size)

        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val px = ref512.getPixel(x, y)
                var bestIdx = 0
                var bestD = Int.MAX_VALUE
                for (i in palette.indices) {
                    val d = dist2(px, palette[i])
                    if (d < bestD) { bestD = d; bestIdx = i }
                }
                total[bestIdx]++
                if (grad[y * w + x] >= thr) onEdge[bestIdx]++
            }
        }

        fun isNearWhite(c: Int) = Color.red(c) > 245 && Color.green(c) > 245 && Color.blue(c) > 245
        fun isNearBlack(c: Int) = Color.red(c) < 12 && Color.green(c) < 12 && Color.blue(c) < 12

        val protected = mutableListOf<Int>()
        for (i in palette.indices) {
            val c = palette[i]

            // Белый/чёрный защищаем всегда, если присутствуют.
            if (isNearWhite(c) || isNearBlack(c)) {
                protected += c
                continue
            }

            val t = total[i]
            if (t == 0) continue
            val eRatio = onEdge[i].toDouble() / t

            // Высокая "краёвость" или очень маленький цвет, но почти весь — на границах.
            if (eRatio > 0.45 || (t < size / 200 && eRatio > 0.30)) {
                protected += c
            }
        }
        return protected.distinct().toIntArray()
    }
}
