package com.appforcross.editor.auto.photo.hq

import android.graphics.Bitmap
import kotlin.math.abs

/**
 * Preset Gate — детерминированный выбор набора параметров без ML.
 * Позже легко заменить реализацию pick(...) на TFLite.
 */
object PresetGate {
        data class Features(
        val skinRatio: Float,
        val flatRatio: Float,
        val darkRatio: Float,
        val edgeStrong: Float
    )

    enum class Preset { PORTRAIT_LIGHT, PORTRAIT_DARK, SKY_SEA, SNOW_NEUTRAL, LOW_KEY, GENERAL }

    data class Params(
        val kMin: Int,
        val minAfterMerge: Int,
        val skinMin: Int, val skyMin: Int, val waterMin: Int, val neutralMin: Int,
        val ampBase: Float, val ampStrong: Float, val fsTarget: Float
    )

    fun computeFeatures(src256: Bitmap): Features {
        val w = src256.width; val h = src256.height; val n = w * h
        val px = IntArray(n); src256.getPixels(px, 0, w, 0, 0, w, h)
        // Luma + YCbCr
        var skin = 0; var dark = 0
        var i = 0
        while (i < n) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            val y = (0.299f * r + 0.587f * g + 0.114f * b)
            val cb = -0.1687f * r - 0.3313f * g + 0.5f * b + 128f
            val cr = 0.5f * r - 0.4187f * g - 0.0813f * b + 128f
            val darkPix = y < 0.12f * 255f
            val skinPix = if (darkPix) (cb in 77f..140f && cr in 118f..180f)
            else           (cb in 80f..135f && cr in 125f..173f)
                if (darkPix) dark++
            if (skinPix) skin++
            i++
        }
        // Gradients
        val L = FloatArray(n)
        for (k in 0 until n) {
            val p = px[k]; val r = (p ushr 16) and 0xFF; val g = (p ushr 8) and 0xFF; val b = p and 0xFF
            L[k] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
        }
        var flat = 0; var strong = 0
        val tFlat = 0.012f; val tStrong = 0.06f
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val i0 = row + x
                val gx = L[i0 + 1] - L[i0 - 1]
                val gy = L[i0 + w] - L[i0 - w]
                val g = abs(gx) + abs(gy)
                if (g < tFlat) flat++
                if (g >= tStrong) strong++
            }
        }
        return Features(
            skinRatio = skin.toFloat() / n,
            flatRatio = flat.toFloat() / n,
            darkRatio = dark.toFloat() / n,
            edgeStrong = strong.toFloat() / n
        )
    }

    fun pick(f: Features): Pair<Preset, Params> = with(f) {
        when {
            skinRatio >= 0.10f && darkRatio < 0.35f ->
                Preset.PORTRAIT_LIGHT to Params(26, 24, 14, 2, 2, 4, 0.20f, 0.25f, 0.25f)
            skinRatio >= 0.10f ->
                Preset.PORTRAIT_DARK  to Params(28, 28, 16, 2, 2, 4, 0.20f, 0.25f, 0.25f)
            skinRatio < 0.05f && flatRatio >= 0.55f && edgeStrong <= 0.08f ->
                Preset.SKY_SEA        to Params(34, 30,  0,12,10, 6, 0.16f, 0.22f, 0.20f)
            darkRatio >= 0.45f ->
                Preset.LOW_KEY        to Params(26, 26,  6, 0, 0, 6, 0.12f, 0.18f, 0.20f)
            flatRatio >= 0.45f ->
                Preset.SNOW_NEUTRAL   to Params(34, 30,  0, 6, 4,10, 0.16f, 0.22f, 0.20f)
            else ->
                Preset.GENERAL        to Params(28, 26, 12, 4, 4, 6, 0.18f, 0.24f, 0.25f)
        }
    }
}
