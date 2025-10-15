package com.appforcross.editor.engine

import android.graphics.Bitmap
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.asImageBitmap
import com.appforcross.core.apply.applyCentroidLut
import com.appforcross.core.color.Metric
import com.appforcross.core.color.argbToOkLab
import com.appforcross.core.dither.Dither
import com.appforcross.core.dither.dither
import com.appforcross.core.image.Raster
import com.appforcross.core.quant.kmeansLab
import com.appforcross.editor.model.*
import kotlin.math.*

class CoreEngine : EditorEngine {
    // === PREPROCESS ===
    // Порядок: denoise → autoLevels → brightness/contrast → gamma → tonal compression
    override fun applyPreprocess(src: ImageBitmap, pp: PreprocessState): ImageBitmap {
        val ab = src.asAndroidBitmap()
        val w = ab.width;
        val h = ab.height
        val n = w * h
        if (n == 0) return src

        val inPx = IntArray(n)
        ab.getPixels(inPx, 0, w, 0, 0, w, h)
        var px = inPx

        // 1) Шумоподавление (box blur радиусом 0/1/2/3)
        val radius = when (pp.denoise) {
            DenoiseLevel.NONE -> 0
            DenoiseLevel.LOW -> 1
            DenoiseLevel.MEDIUM -> 2
            DenoiseLevel.HIGH -> 3
        }
        if (radius > 0) px = boxBlur(px, w, h, radius)

        // 2) Автоуровни (линейная растяжка per-channel)
        if (pp.autoLevels) px = autoLevels(px, w, h)

        // 3) Яркость/контраст (-100..100)
        val add = (pp.brightnessPct.coerceIn(-100, 100) / 100f) * 255f
        val contrast = pp.contrastPct.coerceIn(-100, 100)
        val cFactor = (259f * (contrast + 255)) / (255f * (259 - contrast)) // стандартная формула

        // 4) Гамма (0.1..3.0) LUT
        val gamma = pp.gamma.coerceIn(0.1f, 3.0f)
        val powInv = 1f / gamma
        val gammaLut = IntArray(256) { v ->
            (255.0 * (v / 255.0).pow(powInv.toDouble())).roundToInt().coerceIn(0, 255)
        }

        // 5) Тональная компрессия (0..1) — сведение к среднему 128
        val tc = pp.tonalCompression.coerceIn(0f, 1f)
        val out = IntArray(n)
        var i = 0
        while (i < n) {
            val c = px[i]
            val a = (c ushr 24) and 0xFF
            var r = (c ushr 16) and 0xFF
            var g = (c ushr 8) and 0xFF
            var b = c and 0xFF
            // яркость/контраст
            r = (((r - 128f) * cFactor + 128f + add).roundToInt()).coerceIn(0, 255)
            g = (((g - 128f) * cFactor + 128f + add).roundToInt()).coerceIn(0, 255)
            b = (((b - 128f) * cFactor + 128f + add).roundToInt()).coerceIn(0, 255)
            // гамма
            r = gammaLut[r]; g = gammaLut[g]; b = gammaLut[b]
            // тональная компрессия
            if (tc > 0f) {
                r = lerpToMid(r, tc); g = lerpToMid(g, tc); b = lerpToMid(b, tc)
            }
            out[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
            i++
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(out, 0, w, 0, 0, w, h)
        return bmp.asImageBitmap()
    }

    override fun applySize(src: ImageBitmap, size: SizeState) = src

    override fun applyOptions(
        src: ImageBitmap,
        opt: OptionsState,
        metric: ColorMetric
    ): ImageBitmap {
        // База: картинка уже в "крестиках" (после Size/Palette).
        val raster = src.toRaster()
        val w = raster.width
        val h = raster.height
        if (w == 0 || h == 0) return src

        var argb = raster.argb.copyOf()

        // --- 0) (опц.) Предразмывание при ресэмплинге ---
        if (opt.resampling == ResamplingMode.AVERAGE_PER_CELL && opt.preBlurSigmaPx > 0f) {
            val r = kotlin.math.max(1, kotlin.math.ceil(opt.preBlurSigmaPx * 1.5f).toInt())
            argb = boxBlur(argb, w, h, r) // аппроксимация гаусса
        }

        // --- 1) Палитра «как есть» из изображения + (опц.) слияние близких цветов ---
        var allowed = uniqueColors(argb)
        if (allowed.isEmpty()) return src
        if (opt.mergeDeltaE > 0f && allowed.size > 1) {
            allowed = mergeByDeltaE(allowed, opt.mergeDeltaE, metric)
        }

        // --- 2) Квантование к allowed с (опц.) FS-дизерингом заданной силы ---
        argb = if (opt.fsStrengthPct > 0) {
            val strength = (opt.fsStrengthPct.coerceIn(0, 100)) / 100f
            ditherFsWithStrength(argb, w, h, allowed, strength)
        } else {
            mapToNearest(argb, allowed)
        }

        // --- 3) (опц.) Подчистка одиночных крестиков (8‑соседи, два прохода) ---
        if (opt.cleanSingles) {
            argb = cleanSingles8TwoPass(argb, w, h)
        }

        val out = Raster(w, h, argb)
        return out.toImageBitmap()
    }

    override fun kMeansQuantize(
        src: ImageBitmap,
        maxColors: Int,
        metric: ColorMetric,
        dithering: DitheringType
    ): ImageBitmap {
        if (maxColors <= 0) return src
        val raster = src.toRaster()
        val lab = argbToOkLab(raster.argb)
        val centersLab = kmeansLab(lab, maxColors, iters = 6, seed = 1337)
        // Строим средние цвета кластеров в sRGB (ARGB)
        val k = maxColors
        val sumsR = IntArray(k)
        val sumsG = IntArray(k)
        val sumsB = IntArray(k)
        val counts = IntArray(k)
        var i = 0
        while (i < raster.argb.size) {
            val l = lab[i * 3 + 0]
            val a = lab[i * 3 + 1]
            val b = lab[i * 3 + 2]
            var best = 0
            var bestD = Float.POSITIVE_INFINITY
            var c = 0
            while (c < k) {
                val dl = centersLab[c * 3 + 0] - l
                val da = centersLab[c * 3 + 1] - a
                val db = centersLab[c * 3 + 2] - b
                val d = dl * dl + da * da + db * db
                if (d < bestD) {
                    bestD = d; best = c
                }
                c++
            }
            val argb = raster.argb[i]
            sumsR[best] += (argb shr 16) and 0xFF
            sumsG[best] += (argb shr 8) and 0xFF
            sumsB[best] += argb and 0xFF
            counts[best]++
            i++
        }

        val allowedArgb = IntArray(k) { idx ->
            val cnt = counts[idx]
            if (cnt > 0) {
                val r = sumsR[idx] / cnt
                val g = sumsG[idx] / cnt
                val b = sumsB[idx] / cnt
                (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            } else 0xFF000000.toInt()
        }
        val allowedLab = argbToOkLab(allowedArgb)
        val out = if (dithering == DitheringType.NONE) {
            // Быстро: LUT «центр -> средний цвет кластера»
            val identity = IntArray(k) { it }
            applyCentroidLut(raster, centersLab, identity, allowedArgb)
        } else {
            // Дизеринг к множеству найденных K‑цветов
            val algo = when (dithering) {
                DitheringType.FLOYD_STEINBERG -> Dither.FLOYD_STEINBERG
                DitheringType.ATKINSON -> Dither.ATKINSON
                else -> Dither.NONE
            }
            val m = when (metric) {
                ColorMetric.OKLAB -> Metric.OKLAB
                ColorMetric.DE76 -> Metric.CIE76_LAB
                ColorMetric.DE2000 -> Metric.CIEDE2000
            }
            dither(raster, allowedLab, allowedArgb, m, algo)
        }
        return out.toImageBitmap()
    }

    // --- утилиты ImageBitmap <-> Raster ---
    private fun ImageBitmap.toRaster(): Raster {
        val bmp = asAndroidBitmap()
        val w = bmp.width
        val h = bmp.height
        val data = IntArray(w * h)
        bmp.getPixels(data, 0, w, 0, 0, w, h)
        return Raster(w, h, data)
    }

    private fun Raster.toImageBitmap(): ImageBitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bmp.setPixels(argb, 0, width, 0, 0, width, height)
        return bmp.asImageBitmap()
    }

    // === helpers for preprocess ===
    private fun lerpToMid(v: Int, t: Float): Int {
        val mid = 128f
        return (v + (mid - v) * t).roundToInt().coerceIn(0, 255)
    }

    private fun autoLevels(px: IntArray, w: Int, h: Int): IntArray {
        var rMin = 255;
        var gMin = 255;
        var bMin = 255
        var rMax = 0;
        var gMax = 0;
        var bMax = 0
        for (c in px) {
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            if (r < rMin) rMin = r; if (g < gMin) gMin = g; if (b < bMin) bMin = b
            if (r > rMax) rMax = r; if (g > gMax) gMax = g; if (b > bMax) bMax = b
        }
        val rS = if (rMax > rMin) 255f / (rMax - rMin) else 1f
        val gS = if (gMax > gMin) 255f / (gMax - gMin) else 1f
        val bS = if (bMax > bMin) 255f / (bMax - bMin) else 1f
        val out = IntArray(px.size)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            val a = (c ushr 24) and 0xFF
            val r = (((c ushr 16) and 0xFF) - rMin) * rS
            val g = (((c ushr 8) and 0xFF) - gMin) * gS
            val b = ((c and 0xFF) - bMin) * bS
            out[i] = (a shl 24) or (r.roundToInt().coerceIn(0, 255) shl 16) or
                    (g.roundToInt().coerceIn(0, 255) shl 8) or
                    (b.roundToInt().coerceIn(0, 255))
            i++
        }
        return out
    }

    private fun boxBlur(src: IntArray, w: Int, h: Int, radius: Int): IntArray {
        fun clamp(x: Int, lo: Int, hi: Int) = if (x < lo) lo else if (x > hi) hi else x
        val tmp = IntArray(src.size)
        val out = IntArray(src.size)
        val kernel = 2 * radius + 1
        // Горизонталь
        for (y in 0 until h) {
            var sumR = 0;
            var sumG = 0;
            var sumB = 0;
            var sumA = 0
            val row = y * w
            var x = -radius
            while (x <= radius) {
                val c = src[row + clamp(x, 0, w - 1)]
                sumA += (c ushr 24) and 0xFF
                sumR += (c ushr 16) and 0xFF
                sumG += (c ushr 8) and 0xFF
                sumB += c and 0xFF
                x++
            }
            x = 0
            while (x < w) {
                val idx = row + x
                tmp[idx] = ((sumA / kernel) shl 24) or ((sumR / kernel) shl 16) or
                        ((sumG / kernel) shl 8) or (sumB / kernel)
                val xOut = clamp(x - radius, 0, w - 1)
                val xIn = clamp(x + radius + 1, 0, w - 1)
                val cOut = src[row + xOut];
                val cIn = src[row + xIn]
                sumA += ((cIn ushr 24) and 0xFF) - ((cOut ushr 24) and 0xFF)
                sumR += ((cIn ushr 16) and 0xFF) - ((cOut ushr 16) and 0xFF)
                sumG += ((cIn ushr 8) and 0xFF) - ((cOut ushr 8) and 0xFF)
                sumB += (cIn and 0xFF) - (cOut and 0xFF)
                x++
            }
        }
        // Вертикаль
        for (x in 0 until w) {
            var sumR = 0;
            var sumG = 0;
            var sumB = 0;
            var sumA = 0
            var y = -radius
            while (y <= radius) {
                val c = tmp[clamp(y, 0, h - 1) * w + x]
                sumA += (c ushr 24) and 0xFF
                sumR += (c ushr 16) and 0xFF
                sumG += (c ushr 8) and 0xFF
                sumB += c and 0xFF
                y++
            }
            y = 0
            while (y < h) {
                val idx = y * w + x
                out[idx] = ((sumA / kernel) shl 24) or ((sumR / kernel) shl 16) or
                        ((sumG / kernel) shl 8) or (sumB / kernel)
                val yOut = clamp(y - radius, 0, h - 1)
                val yIn = clamp(y + radius + 1, 0, h - 1)
                val cOut = tmp[yOut * w + x];
                val cIn = tmp[yIn * w + x]
                sumA += ((cIn ushr 24) and 0xFF) - ((cOut ushr 24) and 0xFF)
                sumR += ((cIn ushr 16) and 0xFF) - ((cOut ushr 16) and 0xFF)
                sumG += ((cIn ushr 8) and 0xFF) - ((cOut ushr 8) and 0xFF)
                sumB += (cIn and 0xFF) - (cOut and 0xFF)
                y++
            }
        }
        return out
    }

    // === helpers for options ===
    private fun uniqueColors(px: IntArray): IntArray {
        val set = LinkedHashSet<Int>()
        for (c in px) set.add(c or (0xFF shl 24)) // нормализуем альфу
        return set.toIntArray()
    }

    // Слияние близких цветов по порогу ΔE (используем OKLab/DE76 как евклидово расстояние).
    private fun mergeByDeltaE(allowed: IntArray, deltaE: Float, metric: ColorMetric): IntArray {
        if (allowed.size <= 1) return allowed
        val labs = argbToOkLab(allowed) // по 3 float на цвет
        val thr2 = deltaE * deltaE
        // Кластеры с усреднением в sRGB (дешево и устойчиво для палитры)
        val reps = ArrayList<Int>()
        val sums = ArrayList<IntArray>() // [sumR,sumG,sumB,count]
        var i = 0
        while (i < allowed.size) {
            val c = allowed[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            var best = -1
            var bestD = Float.POSITIVE_INFINITY
            var k = 0
            while (k < reps.size) {
                // сравниваем с текущим репрезентативом репса в OKLab
                val rep = reps[k]
                val idxRep = allowed.indexOf(rep)
                val dl = labs[idxRep * 3 + 0] - labs[i * 3 + 0]
                val da = labs[idxRep * 3 + 1] - labs[i * 3 + 1]
                val db = labs[idxRep * 3 + 2] - labs[i * 3 + 2]
                val d2 = dl * dl + da * da + db * db
                if (d2 < bestD) {
                    bestD = d2; best = k
                }
                k++
            }
            if (best >= 0 && bestD <= thr2) {
                val s = sums[best]
                s[0] += r; s[1] += g; s[2] += b; s[3] += 1
                // обновлять labs репрезентатива не критично — порог «втягивает» близкие
            } else {
                reps.add(c)
                sums.add(intArrayOf(r, g, b, 1))
            }
            i++
        }
        // усреднённые sRGB
        val out = IntArray(reps.size)
        var j = 0
        while (j < reps.size) {
            val s = sums[j]
            val cnt = kotlin.math.max(1, s[3])
            val rr = s[0] / cnt
            val gg = s[1] / cnt
            val bb = s[2] / cnt
            out[j] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
            j++
        }
        return out
    }

    // Квантование без дизеринга: ближайший из allowed (OKLab-евклид)
    private fun mapToNearest(px: IntArray, allowed: IntArray): IntArray {
        val out = IntArray(px.size)
        val labAllowed = argbToOkLab(allowed)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            val rr = (c shr 16) and 0xFF
            val gg = (c shr 8) and 0xFF
            val bb = c and 0xFF
            // быстро приближённый Lab для одного пикселя через локальный перевод
            // (используем argbToOkLab для палитры, для пикселя — минимальная эвристика в sRGB:
            //  на практике для квантования палитры после к‑means хватает sRGB Евклида)
            var best = 0
            var bestD = Int.MAX_VALUE
            var k = 0
            while (k < allowed.size) {
                val a = allowed[k]
                val dr = rr - ((a shr 16) and 0xFF)
                val dg = gg - ((a shr 8) and 0xFF)
                val db = bb - (a and 0xFF)
                val d = dr * dr + dg * dg + db * db
                if (d < bestD) {
                    bestD = d; best = k
                }
                k++
            }
            out[i] = (0xFF shl 24) or (allowed[best] and 0x00FFFFFF)
            i++
        }
        return out
    }

    // Флойд‑Штейнберг с регулируемой силой (0..1), расчёт в sRGB.
    private fun ditherFsWithStrength(
        px: IntArray,
        w: Int,
        h: Int,
        allowed: IntArray,
        strength: Float
    ): IntArray {
        val out = IntArray(px.size)
        val rf = FloatArray(px.size)
        val gf = FloatArray(px.size)
        val bf = FloatArray(px.size)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            rf[i] = ((c shr 16) and 0xFF).toFloat()
            gf[i] = ((c shr 8) and 0xFF).toFloat()
            bf[i] = (c and 0xFF).toFloat()
            i++
        }
        // предрасчёт sRGB палитры
        val pr = IntArray(allowed.size) { (allowed[it] shr 16) and 0xFF }
        val pg = IntArray(allowed.size) { (allowed[it] shr 8) and 0xFF }
        val pb = IntArray(allowed.size) { allowed[it] and 0xFF }
        fun nearestIndex(r: Float, g: Float, b: Float): Int {
            var bi = 0;
            var bd = Float.POSITIVE_INFINITY
            var k = 0
            while (k < allowed.size) {
                val dr = r - pr[k]
                val dg = g - pg[k]
                val db = b - pb[k]
                val d = dr * dr + dg * dg + db * db
                if (d < bd) {
                    bd = d; bi = k
                }
                k++
            }
            return bi
        }

        val s = strength.coerceIn(0f, 1f)
        val idx = { x: Int, y: Int -> y * w + x }
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val p = idx(x, y)
                val r0 = rf[p].coerceIn(0f, 255f)
                val g0 = gf[p].coerceIn(0f, 255f)
                val b0 = bf[p].coerceIn(0f, 255f)
                val ni = nearestIndex(r0, g0, b0)
                val r1 = pr[ni].toFloat()
                val g1 = pg[ni].toFloat()
                val b1 = pb[ni].toFloat()
                out[p] = (0xFF shl 24) or (pr[ni] shl 16) or (pg[ni] shl 8) or pb[ni]
                val er = (r0 - r1) * s
                val eg = (g0 - g1) * s
                val eb = (b0 - b1) * s
                // распределяем ошибку (FS 7/3/5/1)
                if (x + 1 < w) {
                    val q = idx(
                        x + 1,
                        y
                    ); rf[q] += er * (7f / 16f); gf[q] += eg * (7f / 16f); bf[q] += eb * (7f / 16f)
                }
                if (y + 1 < h) {
                    if (x > 0) {
                        val q = idx(
                            x - 1,
                            y + 1
                        ); rf[q] += er * (3f / 16f); gf[q] += eg * (3f / 16f); bf[q] += eb * (3f / 16f)
                    }
                    {
                        val q = idx(
                            x,
                            y + 1
                        ); rf[q] += er * (5f / 16f); gf[q] += eg * (5f / 16f); bf[q] += eb * (5f / 16f)
                    }
                    if (x + 1 < w) {
                        val q = idx(
                            x + 1,
                            y + 1
                        ); rf[q] += er * (1f / 16f); gf[q] += eg * (1f / 16f); bf[q] += eb * (1f / 16f)
                    }
                }
                x++
            }
            y++
        }
        return out
    }

    // Удаление «одиночных»: 8‑соседи, два прохода (туда‑обратно) с голосованием большинства
    private fun cleanSingles8TwoPass(px: IntArray, w: Int, h: Int): IntArray {
        fun pass(inp: IntArray): IntArray {
            val out = inp.copyOf()
            var y = 1
            while (y < h - 1) {
                var x = 1
                while (x < w - 1) {
                    val p = y * w + x
                    val c = out[p]
                    val n = intArrayOf(
                        out[p - 1], out[p + 1], out[p - w], out[p + w],
                        out[p - w - 1], out[p - w + 1], out[p + w - 1], out[p + w + 1]
                    )
                    var same = 0
                    for (v in n) if (v == c) same++
                    if (same == 0) {
                        // голосование большинства
                        val counts = HashMap<Int, Int>(8)
                        for (v in n) counts[v] = (counts[v] ?: 0) + 1
                        var bestColor = c; var bestCnt = 0
                        for ((col, cnt) in counts) if (cnt > bestCnt) { bestCnt = cnt; bestColor = col }
                        if (bestCnt >= 2) out[p] = bestColor
                    }
                    x++
                }
                y++
            }
            return out
        }
        return pass(pass(px))
    }
}
