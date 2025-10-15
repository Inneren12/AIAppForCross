package com.appforcross.editor.auto.photo.hq

import android.graphics.Bitmap
import com.appforcross.editor.photo.hq.HQLog
import kotlin.math.abs
import kotlin.math.max

/**
 * Scale – выбор итогового масштаба S:
 * 1) быстрый прогон кандидатов (по возрастанию) с жёсткими порогами качества → ранний выбор;
 * 2) если никто не прошёл – сводный скор (edge↑, banding↓, flat↓) → лучший S.
 * Возвращает S и уже отмасштабированный Bitmap (кэш).
 */
object Scale {

    data class Probe(
        val bandingRisk: Float,   // 0..1 (ниже лучше)
        val edgeCoverage: Float,  // 0..1 (выше лучше)
        val flatShare: Float,     // 0..1
        val skinRatio: Float,     // 0..1
        val score: Float,         // агрегированный балл
        val passed: Boolean       // прошёл ли жёсткие пороги
    )

    data class Choice(val S: Int, val bmpS: Bitmap, val probe: Probe)

    // Пороговые константы (можно вынести в PhotoConfig при желании)
    private const val T_STRONG = 0.06f       // порог "сильных" градиентов
    private const val T_FLAT   = 0.012f      // плоские/тонкие градиенты
    private const val T_DARK_L = 0.12f       // L* тёмного
    private const val EARLY_BANDING = 0.040f // "жёсткий" порог бэндинга
    private const val EARLY_EDGE    = 0.070f // "жёсткий" порог покрытия краёв

    /**
     * Главный вход: выбрать лучший масштаб из списка кандидатов.
     * Кандидаты должны идти от меньшего к большему (мы делаем ранний выход).
     */
    @JvmStatic
    fun select(source: Bitmap, candidates: IntArray): Choice {
        HQLog.s("scale.candidates: ${candidates.joinToString()}")

        var bestByScore: Choice? = null
        for (s in candidates) {
            val b = scaleToMaxSide(source, s)
            val pr = probe(b)
            HQLog.s("scale.probe: S=$s band=${"%.3f".format(pr.bandingRisk)} edge=${"%.3f".format(pr.edgeCoverage)} flat=${"%.3f".format(pr.flatShare)} skin=${"%.3f".format(pr.skinRatio)} score=${"%.3f".format(pr.score)} pass=${pr.passed}")

            // Ранний выход: первый кандидат, прошедший жёсткие пороги качества
            if (pr.passed) {
                HQLog.s("scale.early: pick S=$s (band≤$EARLY_BANDING, edge≥$EARLY_EDGE)")
                return Choice(s, b, pr)
            }
            // Иначе — копим лучший по скору
            val c = Choice(s, b, pr)
            if (bestByScore == null || c.probe.score > bestByScore!!.probe.score) {
                bestByScore = c
            }
        }
        // Никто не прошёл жёсткие пороги — берём лучший по скору
        val picked = bestByScore!!
        HQLog.s("scale.pick: S=${picked.S} (best by score=${"%.3f".format(picked.probe.score)})")
        return picked
    }

    /**
     * Масштабировать к максимальной стороне = maxSide (с сохранением пропорций).
     * Если размер уже совпадает — возвращает исходный bitmap (без копии).
     */
    @JvmStatic
    fun scaleToMaxSide(src: Bitmap, maxSide: Int): Bitmap {
        val w = src.width; val h = src.height
        val cur = max(w, h)
        if (cur == maxSide) return src
        val k = maxSide.toFloat() / cur.toFloat()
        val nw = (w * k).toInt().coerceAtLeast(1)
        val nh = (h * k).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, nw, nh, true)
    }

    // --------------------------- ВНУТРЕННЕЕ: Оценка кандидата ---------------------------

    private fun probe(bmp: Bitmap): Probe {
        val w = bmp.width; val h = bmp.height; val n = w * h
        val px = IntArray(n); bmp.getPixels(px, 0, w, 0, 0, w, h)

        // L* суррогат + YCbCr кожа (дёшево и стабильно)
        val L = FloatArray(n)
        var skin = 0
        var dark = 0
        var i = 0
        while (i < n) {
            val p = px[i]
            val r = (p ushr 16) and 0xFF
            val g = (p ushr 8) and 0xFF
            val b = p and 0xFF
            val y = (0.299f * r + 0.587f * g + 0.114f * b)        // 0..255
            val cb = -0.1687f * r - 0.3313f * g + 0.5f * b + 128f
            val cr =  0.5f   * r - 0.4187f * g - 0.0813f * b + 128f
            val Lstar = y / 255f
            L[i] = Lstar
            val isDark = Lstar < T_DARK_L
            val isSkin = if (isDark) (cb in 77f..140f && cr in 118f..180f)
            else          (cb in 80f..135f && cr in 125f..173f)
                if (isDark) dark++
            if (isSkin) skin++
            i++
        }
        val skinRatio = skin.toFloat() / n

        // Градиенты и маски: flat/strong
        var flat = 0
        var strong = 0
        for (y in 1 until h - 1) {
            val row = y * w
            for (x in 1 until w - 1) {
                val i0 = row + x
                val gx = L[i0 + 1] - L[i0 - 1]
                val gy = L[i0 + w] - L[i0 - w]
                val g = abs(gx) + abs(gy)
                if (g < T_FLAT) flat++
                if (g >= T_STRONG) strong++
            }
        }
        val flatShare = flat.toFloat() / n
        val edgeCoverage = strong.toFloat() / n

        // Risk бэндинга: окна 7×7 с низкой дисперсией и маленьким ∇
        val risk = bandingRisk(L, w, h)

        // Сводный скор (базовые веса)
        var score = 1.00f * edgeCoverage - 0.90f * risk - 0.15f * flatShare

        // Нюанс под портрет: если кожи достаточно, штрафуем высокий риск бэндинга
        if (skinRatio >= 0.10f) score -= 0.10f * max(0f, risk - 0.035f)
        // Нюанс под небо/море: если сцена очень "плоская" по краям, прижимаем риск
        if (skinRatio < 0.05f && flatShare >= 0.55f && edgeCoverage <= 0.08f) {
            score -= 0.10f * max(0f, risk - 0.030f)
        }

        // Жёсткие пороги для раннего прохода
        val passed = (risk <= EARLY_BANDING && edgeCoverage >= EARLY_EDGE)

        return Probe(
            bandingRisk = risk,
            edgeCoverage = edgeCoverage,
            flatShare = flatShare,
            skinRatio = skinRatio,
            score = score,
            passed = passed
        )
    }

    // Оценка «бэндинга»: считаем долю окон 7×7 с низкой дисперсией и малым локальным градиентом
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

/**
 * Алиас для совместимости: если где‑то остались старые вызовы ScaleSelector, они продолжают работать.
 */
@Deprecated("Use Scale instead")
object ScaleSelector {
    @JvmStatic
    fun select(source: Bitmap, candidates: IntArray): Scale.Choice =
        Scale.select(source, candidates)
    @JvmStatic
    fun scaleToMaxSide(source: Bitmap, maxSide: Int): Bitmap =
        Scale.scaleToMaxSide(source, maxSide)
}

