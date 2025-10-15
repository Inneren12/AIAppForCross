package com.appforcross.editor.auto.photo.hq

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Выбор масштаба S на основании быстрой проверки гладкости/краёв.
 * Возвращает и кэшированный downscale, чтобы не пересоздавать дальше.
 */
object ScaleSelector {
    data class Probe(
        val bandingRisk: Float,   // 0..1 (ниже лучше)
        val edgeCoverage: Float,  // 0..1 (выше лучше)
        val flatShare: Float,     // 0..1
        val score: Float          // агрегированный балл
    )
    data class Choice(val S: Int, val bmpS: Bitmap, val probe: Probe)

    fun select(source: Bitmap, candidates: IntArray): Choice {
        var best: Choice? = null
        for (s in candidates) {
            val b = scaleToMaxSide(source, s)
            val p = probe(b)
            val c = Choice(s, b, p)
            if (best == null || c.probe.score > best!!.probe.score) best = c
        }
        return best!!
    }

    internal fun scaleToMaxSide(src: Bitmap, maxSide: Int): Bitmap {
        val w = src.width; val h = src.height
        if (max(w, h) == maxSide) return src
        val scale = maxSide.toFloat() / max(w, h).toFloat()
        val nw = (w * scale).toInt().coerceAtLeast(1)
        val nh = (h * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, nw, nh, true)
    }

    private fun probe(bmp: Bitmap): Probe {
        val w = bmp.width; val h = bmp.height; val n = w * h
        val px = IntArray(n); bmp.getPixels(px, 0, w, 0, 0, w, h)
        // L* surrogate = normalized luminance
        val L = FloatArray(n)
        var i = 0
        while (i < n) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            L[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
            i++
        }
        // Градиент (|∇|) и flatMask
        var gradSum = 0f
        var flat = 0
        var strong = 0
        val tFlat = 0.012f
        val tStrong = 0.06f
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val i0 = row + x
                val gx = L[i0 + 1] - L[i0 - 1]
                val gy = L[i0 + w] - L[i0 - w]
                val g = abs(gx) + abs(gy)
                gradSum += g
                if (g < tFlat) flat++
                if (g >= tStrong) strong++
            }
        }
        val flatShare = flat.toFloat() / n
        val edgeCoverage = strong.toFloat() / n
        // Banding-risk: локальная дисперсия очень низкая + малый ∇
        val risk = bandingRisk(L, w, h)
        val score = edgeCoverage - 0.6f * risk - 0.1f * flatShare
        return Probe(bandingRisk = risk, edgeCoverage = edgeCoverage, flatShare = flatShare, score = score)
    }

    private fun bandingRisk(L: FloatArray, w: Int, h: Int): Float {
        val win = 7
        val r = win / 2
        var risky = 0
        var total = 0
        var y = r
        while (y < h - r) {
            var x = r
            while (x < w - r) {
                var s = 0f; var s2 = 0f; var c = 0; var gmax = 0f
                var yy = y - r
                while (yy <= y + r) {
                    val row = yy * w
                    var xx = x - r
                    while (xx <= x + r) {
                        val v = L[row + xx]
                        s += v; s2 += v * v; c++
                        if (xx + 1 <= x + r) gmax = max(gmax, abs(L[row + xx + 1] - v))
                        if (yy + 1 <= y + r) gmax = max(gmax, abs(L[row + xx + w] - v))
                        xx++
                    }
                    yy++
                }
                val mean = s / c
                val varL = (s2 / c) - mean * mean
                if (varL < 0.0035f && gmax < 0.010f) risky++
                total++
                x += r
            }
            y += r
        }
        if (total == 0) return 0f
        return risky.toFloat() / total
    }
}
