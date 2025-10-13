package com.appforcross.core.metrics

import android.graphics.Bitmap
import kotlin.math.*

/**
 * CIEDE2000: 95-й перцентиль по сетке семплов.
 */
object DeltaE {

    /**
     * Оценка δE2000, 95-й перцентиль, по равномерной сетке с шагом [stride] (px).
     * Битмапы должны быть одинакового размера; если нет — [test] будет приведён к размеру [orig].
     */
    fun de2000Perc95(orig: Bitmap, test: Bitmap, stride: Int = 4): Float {
        val o = if (orig.width == test.width && orig.height == test.height) orig
        else Downscale.toWidth(test, orig.width).let { _ -> orig }
        val t = if (test.width == orig.width && test.height == orig.height) test
        else Downscale.toWidth(test, orig.width)

        val w = o.width; val h = o.height
        if (w == 0 || h == 0) return 0f
        val pxO = IntArray(w * h)
        val pxT = IntArray(w * h)
        o.getPixels(pxO, 0, w, 0, 0, w, h)
        t.getPixels(pxT, 0, w, 0, 0, w, h)

        val vals = ArrayList<Float>(w * h / (stride * stride))
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val i = y * w + x
                val c1 = pxO[i]
                val c2 = pxT[i]
                val L1a1b1 = rgbToLab(c1)
                val L2a2b2 = rgbToLab(c2)
                vals.add(deltaE2000(L1a1b1, L2a2b2))
                x += stride
            }
            y += stride
        }
        if (vals.isEmpty()) return 0f
        vals.sort()
        val idx = ((vals.size - 1) * 0.95f).toInt().coerceIn(0, vals.lastIndex)
        return vals[idx]
    }

    // ---------------- internals ----------------

    private fun rgbToLab(c: Int): FloatArray {
        val r8 = (c shr 16) and 0xFF
        val g8 = (c shr 8) and 0xFF
        val b8 = c and 0xFF
        // sRGB -> linear
        fun srgbToLinear(u: Int): Double {
            val s = u / 255.0
            return if (s <= 0.04045) s / 12.92 else ((s + 0.055) / 1.055).pow(2.4)
        }
        val rl = srgbToLinear(r8)
        val gl = srgbToLinear(g8)
        val bl = srgbToLinear(b8)
        // linear RGB -> XYZ (D65)
        val X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
        val Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
        val Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
        // normalize by white D65
        val Xn = 0.95047; val Yn = 1.0; val Zn = 1.08883
        val xr = X / Xn; val yr = Y / Yn; val zr = Z / Zn
        fun f(t: Double): Double {
            val e = 216.0 / 24389.0 // (6/29)^3
            val k = 24389.0 / 27.0
            return if (t > e) t.pow(1.0 / 3.0) else (k * t + 16.0) / 116.0
        }
        val fx = f(xr); val fy = f(yr); val fz = f(zr)
        val L = (116.0 * fy - 16.0).toFloat()
        val a = (500.0 * (fx - fy)).toFloat()
        val b = (200.0 * (fy - fz)).toFloat()
        return floatArrayOf(L, a, b)
    }

    // CIEDE2000 (Sharma et al.) — вычисления в Double → результат в Float
    private fun deltaE2000(lab1: FloatArray, lab2: FloatArray): Float {
        val L1 = lab1[0].toDouble(); val a1 = lab1[1].toDouble(); val b1 = lab1[2].toDouble()
        val L2 = lab2[0].toDouble(); val a2 = lab2[1].toDouble(); val b2 = lab2[2].toDouble()

        val avgLp = (L1 + L2) / 2.0
        val C1 = hypot(a1, b1)
        val C2 = hypot(a2, b2)
        val avgC = (C1 + C2) / 2.0

        // G по avgC (до прайминга)
        val G = 0.5 * (1.0 - sqrt(
            avgC.pow(7.0) / (avgC.pow(7.0) + 25.0.pow(7.0))
        ))
        val a1p = (1.0 + G) * a1
        val a2p = (1.0 + G) * a2
        val C1p = hypot(a1p, b1)
        val C2p = hypot(a2p, b2)
        val avgCp = (C1p + C2p) / 2.0

        fun atan2Deg(y: Double, x: Double): Double {
            var h = atan2(y, x) * 180.0 / PI
            if (h < 0.0) h += 360.0
            return h
        }
        val h1p = if (C1p == 0.0) 0.0 else atan2Deg(b1, a1p)
        val h2p = if (C2p == 0.0) 0.0 else atan2Deg(b2, a2p)

        val dLp = L2 - L1
        val dCp = C2p - C1p
        val dhp = when {
            C1p * C2p == 0.0 -> 0.0
            abs(h2p - h1p) <= 180.0 -> h2p - h1p
            h2p - h1p > 180.0 -> (h2p - h1p) - 360.0
            else -> (h2p - h1p) + 360.0
        }
        val dHp = 2.0 * sqrt(C1p * C2p) *
                sin(Math.toRadians(dhp * 0.5))

        val avgHp = when {
            C1p * C2p == 0.0 -> h1p + h2p
            abs(h1p - h2p) <= 180.0 -> (h1p + h2p) * 0.5
            (h1p + h2p) < 360.0 -> (h1p + h2p + 360.0) * 0.5
            else -> (h1p + h2p - 360.0) * 0.5
        }

        val T = 1.0 -
                0.17 * cos(Math.toRadians(avgHp - 30.0)) +
                0.24 * cos(Math.toRadians(2.0 * avgHp)) +
                0.32 * cos(Math.toRadians(3.0 * avgHp + 6.0)) -
                0.20 * cos(Math.toRadians(4.0 * avgHp - 63.0))

        val dRo = 30.0 * exp(-(((avgHp - 275.0) / 25.0).pow(2.0)))
        val Rc  = 2.0 * sqrt(
            avgCp.pow(7.0) / (avgCp.pow(7.0) + 25.0.pow(7.0))
        )
        val Sl  = 1.0 + (0.015 * (avgLp - 50.0) * (avgLp - 50.0)) /
                sqrt(20.0 + (avgLp - 50.0) * (avgLp - 50.0))
        val Sc = 1.0 + 0.045 * avgCp
        val Sh = 1.0 + 0.015 * avgCp * T
        val Rt = -sin(Math.toRadians(2.0 * dRo)) * Rc

        val kL = 1.0; val kC = 1.0; val kH = 1.0
        val dl = dLp / (kL * Sl)
        val dc = dCp / (kC * Sc)
        val dh = dHp / (kH * Sh)

        val dE = sqrt(dl*dl + dc*dc + dh*dh + Rt * dc * dh)
        return dE.toFloat()
        }

    // надёжная степень через java.lang.Math
    private fun Double.pow(p: Double): Double = Math.pow(this, p)
    fun de2000(lab1: FloatArray, lab2: FloatArray): Float {
        return deltaE2000(lab1, lab2)
    }
}
