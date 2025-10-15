package com.appforcross.editor.auto.photo.hq

import android.graphics.Bitmap
import android.util.Log
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * Минимальная сборка палитры (stub).
 * TODO: заменить на твой RAQ/merge/refill/anchors; сигнатура останется прежней.
 */
object PaletteBuilder {
    data class Palette(val argb: IntArray, val lab: FloatArray)

    fun build(bmpS: Bitmap, threadPalette: List<*>, p: PresetGate.Params): Palette {
        // Простой RGB‑kmeans как "заглушка", чтобы собрать проект и прогонять пайплайн.
        val k = max(p.kMin, 19).coerceAtMost(64)
        val colors = kmeansRgb(bmpS, k)
        val lab = FloatArray(colors.size * 3) { 0f } // okLab заглушка; не используется в Renderer stub
        Log.d("PhotoHQ.RAQ", "stub kmeansRGB k=$k → ${colors.size}")
        return Palette(colors, lab)
    }

    // --- very simple RGB k-means (few iters), deterministic seed ---
    private fun kmeansRgb(bmp: Bitmap, k: Int, iters: Int = 6): IntArray {
        val w = bmp.width; val h = bmp.height
        val n = w * h
        val px = IntArray(n); bmp.getPixels(px, 0, w, 0, 0, w, h)
        val rnd = Random(12345)
        val centers = IntArray(k)
        val step = max(1, n / k)
        var idx = 0
        for (i in 0 until k) {
            centers[i] = px[idx]
            idx = (idx + step) % n
        }
        val assign = IntArray(n)
        repeat(iters) {
            // assign
            for (i in 0 until n) {
                val p = px[i]
                var best = 0; var bd = Int.MAX_VALUE
                for (c in 0 until k) {
                    val d = rgbDist2(p, centers[c])
                    if (d < bd) { bd = d; best = c }
                }
                assign[i] = best
            }
            // update
            val sumR = IntArray(k); val sumG = IntArray(k); val sumB = IntArray(k); val cnt = IntArray(k)
            for (i in 0 until n) {
                val a = assign[i]; val p = px[i]
                sumR[a] += (p ushr 16) and 0xFF
                sumG[a] += (p ushr 8) and 0xFF
                sumB[a] += p and 0xFF
                cnt[a]++
            }
            for (c in 0 until k) {
                if (cnt[c] > 0) {
                    val r = (sumR[c] / cnt[c]).coerceIn(0, 255)
                    val g = (sumG[c] / cnt[c]).coerceIn(0, 255)
                    val b = (sumB[c] / cnt[c]).coerceIn(0, 255)
                    centers[c] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        return centers.distinct().toIntArray()
    }

    private fun rgbDist2(a: Int, b: Int): Int {
        val ar = (a ushr 16) and 0xFF; val ag = (a ushr 8) and 0xFF; val ab = a and 0xFF
        val br = (b ushr 16) and 0xFF; val bg = (b ushr 8) and 0xFF; val bb = b and 0xFF
        val dr = ar - br; val dg = ag - bg; val db = ab - bb
        return dr*dr + dg*dg + db*db
    }
}
