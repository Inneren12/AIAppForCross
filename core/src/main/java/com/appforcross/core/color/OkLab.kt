
package com.appforcross.core.color
/**
 * Конвертация ARGB -> OKLab (интерлив L,a,b) и метрики расстояния.
 * Без Android/Compose зависимостей.
 */
enum class Metric { OKLAB, CIE76_LAB, CIEDE2000 }
/**
 * Преобразует массив ARGB (0xAARRGGBB) в FloatArray OKLab, интерлив по 3 значения на пиксель: L,a,b.
 */
fun argbToOkLab(argb: IntArray): FloatArray {
    val n = argb.size
    val out = FloatArray(n * 3)
    var j = 0
    for (i in 0 until n) {
        val c = argb[i]
        val r = ((c shr 16) and 0xFF) / 255.0
        val g = ((c shr 8) and 0xFF) / 255.0
        val b = (c and 0xFF) / 255.0
        val rl = srgbToLinear(r)
        val gl = srgbToLinear(g)
        val bl = srgbToLinear(b)
        // OKLab (Björn Ottosson)
        val l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl
        val m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl
        val s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl
        val l_ = cbrt(l)
        val m_ = cbrt(m)
        val s_ = cbrt(s)
        val L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        val A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        val B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        out[j++] = L.toFloat()
        out[j++] = A.toFloat()
        out[j++] = B.toFloat()
    }
    return out
}

private fun srgbToLinear(c: Double): Double =
    if (c <= 0.04045) c / 12.92 else Math.pow((c + 0.055) / 1.055, 2.4)

private fun cbrt(x: Double): Double = Math.cbrt(x)
/**
 * Евклидово расстояние (квадрат) в зависимости от метрики.
 * Для CIEDE2000 возвращаем ΔE^2 (квадрат), чтобы быть консистентными со сравнением.
 */
fun distanceSq(
    l1: Float, a1: Float, b1: Float,
    l2: Float, a2: Float, b2: Float,
    metric: Metric
    ): Float {
    return when (metric) {
        Metric.OKLAB, Metric.CIE76_LAB -> {
            val dl = (l1 - l2)
            val da = (a1 - a2)
            val db = (b1 - b2)
            dl * dl + da * da + db * db
        }
        Metric.CIEDE2000 -> {
            // Упрощённая реализация: считаем ΔE2000 и возводим в квадрат для унификации сравнения.
            // Входные значения трактуем как L*, a*, b* той же шкалы.
            val de = deltaE2000(l1.toDouble(), a1.toDouble(), b1.toDouble(), l2.toDouble(), a2.toDouble(), b2.toDouble())
            (de * de).toFloat()
        }
    }
}

// Реализация ΔE2000 Входные/выходные значения в double.
private fun deltaE2000(L1: Double, a1: Double, b1: Double, L2: Double, a2: Double, b2: Double): Double {
    val kL = 1.0; val kC = 1.0; val kH = 1.0
    val C1 = Math.sqrt(a1 * a1 + b1 * b1)
    val C2 = Math.sqrt(a2 * a2 + b2 * b2)
    val Cm = (C1 + C2) / 2.0
    val G = 0.5 * (1.0 - Math.sqrt(Math.pow(Cm, 7.0) / (Math.pow(Cm, 7.0) + Math.pow(25.0, 7.0))))
    val a1p = (1.0 + G) * a1
    val a2p = (1.0 + G) * a2
    val C1p = Math.sqrt(a1p * a1p + b1 * b1)
    val C2p = Math.sqrt(a2p * a2p + b2 * b2)
    val h1p = Math.atan2(b1, a1p).let { if (it < 0) it + 2.0 * Math.PI else it }
    val h2p = Math.atan2(b2, a2p).let { if (it < 0) it + 2.0 * Math.PI else it }
    val dLp = L2 - L1
    val dCp = C2p - C1p
    val dhp = when {
        C1p * C2p == 0.0 -> 0.0
        Math.abs(h2p - h1p) <= Math.PI -> h2p - h1p
        h2p - h1p > Math.PI -> (h2p - h1p) - 2.0 * Math.PI
        else -> (h2p - h1p) + 2.0 * Math.PI
    }
    val dHp = 2.0 * Math.sqrt(C1p * C2p) * Math.sin(dhp / 2.0)
    val Lpm = (L1 + L2) / 2.0
    val Cpm = (C1p + C2p) / 2.0
    val hpBar = when {
        C1p * C2p == 0.0 -> h1p + h2p
        Math.abs(h1p - h2p) <= Math.PI -> (h1p + h2p) / 2.0
        (h1p + h2p) < 2.0 * Math.PI -> (h1p + h2p + 2.0 * Math.PI) / 2.0
        else -> (h1p + h2p - 2.0 * Math.PI) / 2.0
    }
    val T = 1.0 - 0.17 * Math.cos(hpBar - Math.toRadians(30.0)) +
            0.24 * Math.cos(2.0 * hpBar) +
            0.32 * Math.cos(3.0 * hpBar + Math.toRadians(6.0)) -
            0.20 * Math.cos(4.0 * hpBar - Math.toRadians(63.0))
    val dTheta = Math.toRadians(30.0) * Math.exp(-((Math.toDegrees(hpBar) - 275.0) / 25.0) * ((Math.toDegrees(hpBar) - 275.0) / 25.0))
    val Rc = 2.0 * Math.sqrt(Math.pow(Cpm, 7.0) / (Math.pow(Cpm, 7.0) + Math.pow(25.0, 7.0)))
    val Sl = 1.0 + (0.015 * (Lpm - 50.0) * (Lpm - 50.0)) / Math.sqrt(20.0 + (Lpm - 50.0) * (Lpm - 50.0))
    val Sc = 1.0 + 0.045 * Cpm
    val Sh = 1.0 + 0.015 * Cpm * T
    val Rt = -Math.sin(2.0 * dTheta) * Rc
    val dE = Math.sqrt(
        (dLp / (kL * Sl)) * (dLp / (kL * Sl)) +
                (dCp / (kC * Sc)) * (dCp / (kC * Sc)) + (dHp / (kH * Sh)) * (dHp / (kH * Sh)) + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))
    return dE
    }
