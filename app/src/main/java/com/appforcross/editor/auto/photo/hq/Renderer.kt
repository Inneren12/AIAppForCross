`package com.appforcross.editor.auto.photo.hq

import android.graphics.Bitmap
import android.util.Log
import kotlin.math.abs

/**
 * Мини‑рендер (stub): nearest‑map в RGB + один majority‑clean + базовые метрики.
 * TODO: заменить на твой гибридный дизер (ORDERED 8×8 + FS‑лента), метрики оставить.
 */
object Renderer {
    data class Frame(
        val bitmap: Bitmap,
        val usedSwatches: Int,
        val logs: Map<String, Any>
    )

    fun render(bmpS: Bitmap, palette: PaletteBuilder.Palette, p: PresetGate.Params): Frame {
        val w = bmpS.width; val h = bmpS.height; val n = w * h
        val src = IntArray(n); bmpS.getPixels(src, 0, w, 0, 0, w, h)
        val pal = palette.argb

        // nearest in RGB
        val ids = IntArray(n)
        for (i in 0 until n) ids[i] = nearestRgb(src[i], pal)

        // majority‑clean 3×3 (один проход) вне краёв (stub: применяем по всему кадру)
        val changed = majority3x3(ids, w, h)

        // compose
        val out = IntArray(n)
        for (i in 0 until n) out[i] = pal[ids[i]]
        val bmpOut = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmpOut.setPixels(out, 0, w, 0, 0, w, h)

        // базовые метрики «шиваемости»
        val m = computeMetrics(ids, w, h, used = pal.size)
        Log.d("PhotoHQ.Dither", "stub: FS=0%, ORD=0%, OFF=100%")
        val logs = mapOf(
            "banding" to m.banding,
            "fsPct" to 0f,
            "singletonPct" to m.singletonPct,
            "runLen50" to m.runLen50,
            "usedSwatches" to pal.size,
            "deltaE95" to m.deltaE95
        )
        return Frame(bitmap = bmpOut, usedSwatches = pal.size, logs = logs)
    }

    private fun nearestRgb(p: Int, pal: IntArray): Int {
        var best = 0; var bd = Int.MAX_VALUE
        for (i in pal.indices) {
            val d = rgbDist2(p, pal[i])
            if (d < bd) { bd = d; best = i }
        }
        return best
    }

    private fun rgbDist2(a: Int, b: Int): Int {
        val ar = (a ushr 16) and 0xFF; val ag = (a ushr 8) and 0xFF; val ab = a and 0xFF
        val br = (b ushr 16) and 0xFF; val bg = (b ushr 8) and 0xFF; val bb = b and 0xFF
        val dr = ar - br; val dg = ag - bg; val db = ab - bb
        return dr*dr + dg*dg + db*db
    }

    // Однопроходный majority 3×3: если центральный цвет — абсолютное меньшинство, заменить на мажоритарный.
    private fun majority3x3(ids: IntArray, w: Int, h: Int): Int {
        var changed = 0
        val copy = ids.clone()
        for (y in 1 until h-1) {
            val row = y*w
            for (x in 1 until w-1) {
                val i = row + x
                val c = copy[i]
                val hist = HashMap<Int, Int>(9)
                for (yy in -1..1) {
                    val r = (y+yy)*w
                    for (xx in -1..1) {
                        val id = copy[r + (x+xx)]
                        hist[id] = (hist[id] ?: 0) + 1
                    }
                }
                val maxEntry = hist.maxByOrNull { it.value }!!
                if (maxEntry.value >= 5 && maxEntry.key != c) {
                    ids[i] = maxEntry.key
                    changed++
                }
            }
        }
        return changed
    }

    // Простейшие метрики (stub). TODO: заменить на твои Block‑метрики/ΔE/EdgeSSIM и т.д.
    private data class M(val singletonPct: Float, val runLen50: Float, val banding: Float, val deltaE95: Float)
    private fun computeMetrics(ids: IntArray, w: Int, h: Int, used: Int): M {
        // singleton%
        var singles = 0
        for (y in 1 until h-1) {
            val row = y*w
            for (x in 1 until w-1) {
                val c = ids[row+x]
                var same = 0
                for (yy in -1..1) {
                    val r = (y+yy)*w
                    for (xx in -1..1) if (ids[r+(x+xx)] == c) same++
                }
                if (same == 1) singles++
            }
        }
        val singletonPct = singles.toFloat() / (w*h) * 100f
        // runLen50 (по строкам)
        val runs = ArrayList<Int>(w*h/2)
        for (y in 0 until h) {
            var r = 1
            for (x in 1 until w) {
                if (ids[y*w+x] == ids[y*w+x-1]) r++ else { runs.add(r); r = 1 }
            }
            runs.add(r)
        }
        runs.sort()
        val runLen50 = if (runs.isNotEmpty()) runs[runs.size/2].toFloat() else 0f
        // banding (proxy)
        val banding = (100f - runLen50).coerceAtLeast(0f) / 100f
        // deltaE95 (proxy): нет LAB → просто 0 как заглушка
        val deltaE95 = 0f
        return M(singletonPct, runLen50, banding, deltaE95)
    }
}