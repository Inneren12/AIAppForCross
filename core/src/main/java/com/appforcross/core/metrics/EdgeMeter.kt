package com.appforcross.core.metrics

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class EdgeFeatures(
    /** Доля «сильных» границ в картине [0..1]. */
    val density: Float,
    /** Доля границ исходника, сохранившихся в тестовом изображении [0..1]. */
    val preserve: Float
)

object EdgeMeter {

    /**
     * Оценка плотности сильных границ по Собелю.
     * Работает по яркости (luma) в диапазоне [0..255].
     */
    fun edgeDensity(src: Bitmap): Float {
        val w = src.width
        val h = src.height
        if (w < 3 || h < 3) return 0f
        val y = luma(src)
        val g = sobelGrad(y, w, h)
        val (mean, std) = meanStd(g)
        val thr = mean + 1.25f * std // адаптивный порог «сильных» границ
        var strong = 0
        val n = g.size
        for (i in 0 until n) if (g[i] >= thr) strong++
        return strong.toFloat() / n
    }

    /**
     * Сохранность границ: сколько «сильных» границ исходника нашлось в тесте (с учётом +-1 пиксели).
     * Внутри выравниваем размеры (масштабируем test к размеру orig).
     */
    // EdgeMeter.kt
    fun preserve(orig: Bitmap, test: Bitmap): Float {
        // 1) выравниваем размеры по обеим осям
        val w = orig.width
        val h = orig.height
        if (w < 3 || h < 3) return 0f

        val tAligned = if (test.width == w && test.height == h) {
            test
        } else {
            // важный момент: выравниваем по width И height, а не только по ширине
            Bitmap.createScaledBitmap(test, w, h, /*filter=*/true)
        }

        // 2) яркость и градиенты (размеры гарантированно совпадают)
        val yo = luma(orig)
        val yt = luma(tAligned)
        val go = sobelGrad(yo, w, h)
        val gt = sobelGrad(yt, w, h)

        // 3) адаптивные пороги «сильных» границ
        val (mo, so) = meanStd(go)
        val (mt, st) = meanStd(gt)
        val thro = mo + 1.25f * so
        val thrt = mt + 1.25f * st

        val eo = BooleanArray(w * h) { i -> go[i] >= thro }
        val et = BooleanArray(w * h) { i -> gt[i] >= thrt }

        // 4) совпадения с допуском 1px
        var edgesOrig = 0
        var preserved = 0
        for (y in 1 until h - 1) {
            var idx = y * w + 1
            for (x in 1 until w - 1) {
                if (eo[idx]) {
                    edgesOrig++
                    var ok = false
                    loop@ for (dy in -1..1) {
                        val row = (y + dy) * w
                        for (dx in -1..1) {
                            if (et[row + (x + dx)]) { ok = true; break@loop }
                        }
                    }
                    if (ok) preserved++
                }
                idx++
            }
        }
        if (edgesOrig == 0) return 1f
        return preserved.toFloat() / edgesOrig
    }

    /** Бинарная маска «сильных» границ с адаптивным порогом. */
    fun strongEdgeMask(src: Bitmap, k: Float = 1.25f): BooleanArray {
        val w = src.width
        val h = src.height
        if (w < 3 || h < 3) return BooleanArray(w * h)
        val y = luma(src)
        val g = sobelGrad(y, w, h)
        val (m, s) = meanStd(g)
        val thr = m + k * s
        val out = BooleanArray(w * h)
        var i = 0
        while (i < out.size) {
            out[i] = g[i] >= thr
            i++
        }
        return out
    }

    /**+
     * Поле касательных к границам для анизотропного FS.
     * Возвращает пару массивов (tx, ty) размера w*h — единичный вектор касательной в каждой точке.
     * В плоских областях (|∇|≈0) касательная по умолчанию (1,0).
     */
    fun tangentField(src: Bitmap): Pair<FloatArray, FloatArray> {
        val w = src.width
        val h = src.height
        val tx = FloatArray(w * h) { 1f }
        val ty = FloatArray(w * h) { 0f }
        if (w < 3 || h < 3) return tx to ty
        val y = luma(src)
        val invSqrt2 = 0.70710677f
        for (j in 1 until h - 1) {
            var idx = j * w + 1
            for (i in 1 until w - 1) {
                val jm = j - 1; val jp = j + 1
                val im = i - 1; val ip = i + 1
                // Собель-компоненты градиента
                val a = y[(jm) * w + im]; val b = y[(jm) * w + i];  val c = y[(jm) * w + ip]
                val d = y[(j ) * w + im]; val e = y[(j ) * w + i];  val f = y[(j ) * w + ip]
                val g0= y[(jp) * w + im]; val h0 = y[(jp) * w + i]; val k = y[(jp) * w + ip]
                val gx = (-a - 2f * d - g0 + c + 2f * f + k)
                val gy = (-a - 2f * b - c + g0 + 2f * h0 + k)
                val mag = hypot(gx, gy)
                if (mag > 1e-3f) {
                    // Касательная к границе = перпендикуляр к градиенту
                    tx[idx] = (-gy / mag)
                    ty[idx] = ( gx / mag)
                } else {
                        tx[idx] = 1f; ty[idx] = 0f
                    }
                idx++
            }
        }
        return tx to ty
    }

    /**
    +     * Маска «плоских» зон: |∇L| < thr, где L — яркость 0..1.
    +     * thr ~ 0.008–0.015 в зависимости от сцены (см. PhotoConfig.B5.FLAT_GRAD_T).
    +     */
    fun flatMask(src: Bitmap, thr: Float): BooleanArray {
        val w = src.width; val h = src.height; val n = w * h
        val y = lumaF(src) // 0..1
        val mask = BooleanArray(n)
        if (w < 3 || h < 3) return mask
        for (j in 1 until h - 1) {
            var i = 1
            val off = j * w
            while (i < w - 1) {
                val p = off + i
                val a = y[p - w - 1]; val b = y[p - w]; val c = y[p - w + 1]
                val d0= y[p - 1];     val f0= y[p + 1]
                val g = y[p + w - 1]; val h0= y[p + w]; val k = y[p + w + 1]
                val gx = (-a - 2f * d0 - g + c + 2f * f0 + k)
                val gy = (-a - 2f * b  - c + g + 2f * h0 + k)
                val mag = hypot(gx, gy)
                mask[p] = mag < thr
                i++
            }
        }
        return mask
    }

    // ---------------- internals ----------------

    /** Быстрая яркость 0..1 */
    internal fun lumaF(src: Bitmap): FloatArray {
        val w = src.width; val h = src.height; val n = w * h
        val px = IntArray(n); src.getPixels(px, 0, w, 0, 0, w, h)
        val out = FloatArray(n)
        var i = 0
        while (i < n) {
            val p = px[i]
            val r = ((p ushr 16) and 0xFF) / 255f
            val g = ((p ushr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            out[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b
            i++
        }
        return out
    }

    /** Яркость Y (sRGB): 0..255, одна компонента на пиксель. */
    private fun luma(src: Bitmap): FloatArray {
        val w = src.width; val h = src.height
        val px = IntArray(w * h)
        src.getPixels(px, 0, w, 0, 0, w, h)
        val y = FloatArray(px.size)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            y[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b
            i++
        }
        return y
    }

    /** Градиент Собеля: модуль градиента на каждый пиксель (границы 1px игнорируем). */
    private fun sobelGrad(y: FloatArray, w: Int, h: Int): FloatArray {
        val out = FloatArray(w * h)
        for (j in 1 until h - 1) {
            val jm = j - 1; val jp = j + 1
            var i = j * w + 1
            for (iX in 1 until w - 1) {
                val xm = iX - 1; val xp = iX + 1
                val a = y[jm * w + xm]; val b = y[jm * w + iX]; val c = y[jm * w + xp]
                val d = y[j  * w + xm]; val e = y[j  * w + iX]; val f = y[j  * w + xp]
                val g0 = y[jp * w + xm]; val h0 = y[jp * w + iX]; val k = y[jp * w + xp]
                val gx = -a - 2f * d - g0 + c + 2f * f + k
                val gy = -a - 2f * b - c + g0 + 2f * h0 + k
                out[i] = hypot(gx, gy)
                i++
            }
        }
        return out
    }

    private fun meanStd(v: FloatArray): Pair<Float, Float> {
        var sum = 0.0
        for (x in v) sum += x
        val mean = (sum / v.size).toFloat()
        var ss = 0.0
        for (x in v) {
            val d = x - mean
            ss += (d * d)
        }
        val std = sqrt((ss / v.size).toFloat())
        return mean to std
    }
}

