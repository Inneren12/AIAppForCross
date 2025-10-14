package com.appforcross.core.zones

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

enum class Zone { SKY, CLOUD, WATER, VEG, GROUND, BUILT, SKIN }

data class ZoneWeights(
    val sky: FloatArray,
    val cloud: FloatArray,
    val water: FloatArray,
    val veg: FloatArray,
    val ground: FloatArray,
    val built: FloatArray,
    val skin: FloatArray
    ) {
    fun widthHeightFrom(size: Int): Pair<Int, Int> = 0 to 0 // не используется, заглушка для совместимости
    }

/** Быстрое вычисление L/S/H + градиентов и soft‑masк зон на одном проходе. */
object ZoneMasks {
    data class Lsh(val L: FloatArray, val S: FloatArray, val H: FloatArray)

    fun compute(src: Bitmap): ZoneWeights {
        val w = src.width; val h = src.height; val n = w * h
        val px = IntArray(n); src.getPixels(px, 0, w, 0, 0, w, h)
        val L = FloatArray(n); val S = FloatArray(n); val H = FloatArray(n)
        // RGB→HSL (L∈[0..1], S∈[0..1], H∈[0..360))
        var i = 0
        while (i < n) {
            val p = px[i]
            val r = ((p shr 16) and 255) / 255f
            val g = ((p shr 8) and 255) / 255f
            val b = (p and 255) / 255f
            val maxc = max(r, max(g, b)); val minc = min(r, min(g, b))
            val l = 0.5f * (maxc + minc)
            val d = maxc - minc
            val s = if (d == 0f) 0f else d / (1f - abs(2f * l - 1f))
            var hDeg = 0f
            if (d != 0f) {
                hDeg = when (maxc) {
                    r -> 60f * (((g - b) / d) % 6f)
                    g -> 60f * (((b - r) / d) + 2f)
                    else -> 60f * (((r - g) / d) + 4f)
                }
                if (hDeg < 0f) hDeg += 360f
            }
            L[i] = l; S[i] = s; H[i] = hDeg
            i++
        }
        // Градиенты по L (Sobel lite)
        val gx = FloatArray(n); val gy = FloatArray(n)
        if (w >= 3 && h >= 3) {
            for (y in 1 until h - 1) {
                val off = y * w
                for (x in 1 until w - 1) {
                    val i0 = off + x
                    val im = i0 - 1; val ip = i0 + 1
                    val iu = i0 - w; val id = i0 + w
                    val a = L[iu - 1]; val b = L[iu]; val c = L[iu + 1]
                    val d0 = L[im];  val f0 = L[ip]
                    val g0 = L[id - 1]; val h0 = L[id]; val k = L[id + 1]
                    gx[i0] = (-a - 2f * d0 - g0 + c + 2f * f0 + k)
                    gy[i0] = (-a - 2f * b  - c  + g0 + 2f * h0 + k)
                }
            }
        }
        // Логиты зон
        val sky = FloatArray(n); val cloud = FloatArray(n); val water = FloatArray(n)
        val veg = FloatArray(n); val ground = FloatArray(n); val built = FloatArray(n)
        val skin = FloatArray(n)
        val T = 0.8f; val eps = 1e-3f
        i = 0
        while (i < n) {
            val l = L[i]; val s = S[i]; var hq = H[i]
            val ax = abs(gx[i]); val ay = abs(gy[i])
            val g = ax + ay + eps
            val coh = abs(ax - ay) / g
            val horiz = (ax - ay) / g
            if (hq < 0f) hq += 360f
            var zSky   =  2.2f*(l - 0.6f) - 1.2f*s - 0.6f*g + if (hq in 190f..260f) 0.4f else 0f
            var zCloud =  2.0f*(l - 0.7f) - 1.8f*s + 0.2f*g
            var zWater =  0.9f*(l - 0.5f) - 0.2f*s + 1.3f*horiz + if (hq in 180f..220f) 0.5f else 0f
            var zVeg   = -0.2f*(l - 0.5f) + 1.2f*(s - 0.2f) + if (hq in 70f..170f) 0.8f else 0f
            var zGround=  0.2f*(l - 0.5f) + 0.5f*s + if (hq in 20f..70f) 0.5f else 0f
            var zBuilt =  1.6f*coh + 0.6f*g
            // softmax
            // примитивный скин-логит: оттенки 10..50°, S 0.15..0.6, L 0.25..0.85 + невысокий ∇
            val hueOk = (hq >= 10f && hq <= 50f)
            val sOk = (s >= 0.15f && s <= 0.60f)
            val lOk = (l >= 0.25f && l <= 0.85f)
            val gFlat = 1f - (g * 2.5f) // чем меньше ∇, тем лучше
            var zSkin = (if (hueOk && sOk && lOk) 1.5f else 0f) + 0.6f * gFlat

            // softmax (7 классов)
            val e0 = fastExp((zSky)/T)
            val e1 = fastExp((zCloud)/T)
            val e2 = fastExp((zWater)/T)
            val e3 = fastExp((zVeg)/T)
            val e4 = fastExp((zGround)/T)
            val e5 = fastExp((zBuilt)/T)
            val e6 = fastExp((zSkin)/T)
            val sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + eps
            sky[i]   = (e0 / sum)
            cloud[i] = (e1 / sum)
            water[i] = (e2 / sum)
            veg[i]   = (e3 / sum)
            ground[i]= (e4 / sum)
            built[i] = (e5 / sum)
            skin[i]  = (e6 / sum)
            i++
        }
        return ZoneWeights(sky, cloud, water, veg, ground, built, skin)
    }

    private fun fastExp(x: Float): Float {
        // достаточно точная аппроксимация exp() для наших диапазонов логитов
        return exp(x)
    }
}
