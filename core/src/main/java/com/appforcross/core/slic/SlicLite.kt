package com.appforcross.core.slic

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.random.Random

/**
 * Очень лёгкая «SLIC‑lite» заглушка: регулярная сетка регионов (без итераций).
 * Достаточно для «стабилизации» цвета в больших плоских областях.
 */
object SlicLite {
    fun segmentGrid(src: Bitmap, regions: Int): IntArray {
        val w = src.width; val h = src.height
        val n = w * h
        val out = IntArray(n)
        if (regions <= 1) { for (i in 0 until n) out[i] = 0; return out }
        val cellsX = max(1, kotlin.math.sqrt(regions.toFloat() * w / h).roundToInt())
        val cellsY = max(1, (regions / cellsX).coerceAtLeast(1))
        val cw = max(1, w / cellsX)
        val ch = max(1, h / cellsY)
        var y = 0
        var id = 0
        while (y < h) {
            var x = 0
            var cx = 0
            while (x < w) {
                val xx = min(x + cw, w)
                val yy = min(y + ch, h)
                var j = y
                while (j < yy) {
                    var i0 = j * w + x
                    while (i0 < j * w + xx) {
                        out[i0] = id; i0++
                    }
                    j++
                }
                id++
                x += cw; cx++
            }
            y += ch
        }
        return out
    }
}

