package com.appforcross.editor.photo.hq

import android.graphics.Bitmap
import android.graphics.Bitmap.Config.ARGB_8888
import com.appforcross.core.color.Metric
import com.appforcross.core.color.argbToOkLab
import com.appforcross.core.dither.Dither
import com.appforcross.core.dither.dither
import com.appforcross.core.dither.ditherOrderedBayer8
import com.appforcross.core.image.Raster
import com.appforcross.core.metrics.BandingMeter
import com.appforcross.core.metrics.DeltaE
import com.appforcross.core.metrics.Downscale
import com.appforcross.core.metrics.EdgeMeter
import com.appforcross.core.metrics.SsimMeter
import com.appforcross.core.palette.Swatch
import com.appforcross.core.quant.kmeansLab
import com.appforcross.core.apply.applyCentroidLut
import com.appforcross.editor.auto.photo.PhotoConfig
import kotlin.math.max
import kotlin.math.min
import com.appforcross.core.metrics.EdgeGuard
import com.appforcross.core.dither.ditherFsAniso
import com.appforcross.core.zones.Zone
import com.appforcross.core.zones.ZoneMasks
import com.appforcross.core.raq.PaletteAllocator
import com.appforcross.core.raq.RAQBounds
import com.appforcross.core.slic.SlicLite
import kotlin.math.pow
import com.appforcross.core.dither.ditherOrdered8x8Tiles

/**
 * PhotoHQ Orchestrator (этап B.3):
 *  — авто‑тюнер на превью (≤512): перебирает {K, blur, dither} и считает метрики,
 *  — финальный рендер БЕЗ PhotoPipeline: protected‑merge палитры, затем пер‑пиксельный гибридный дизеринг
 *    (ORDERED в banding‑окнах, FS на краях/текстурах), выбор размера S по метрикам.
 */

// ─────────────────────────── Anti‑sand tuning (PHOTO) ──────────────────────────
// Включатели по умолчанию: можно временно выключить, если понадобится A/B.
private const val PRE_EDGE_AWARE_BLUR: Boolean = true      // предсглаживание вне сильных краёв
private const val PRE_BLUR_SIGMA_PX: Float = 0.7f          // ~σ 0.6–0.8
private const val MAJORITY_CLEAN_ENABLED: Boolean = true   // majority 3×3 перед дизером
// Максимальный размер «тонкой» компоненты (в пикселях) для guard majority/сглаживания
private const val THIN_COMPONENT_MAX = 24
private const val TONE_CURVE_ENABLED: Boolean = true
private const val TONE_GAMMA_IN: Float = 0.83f        // gamma_in ≈ 1/1.9
private const val TONE_HIGHLIGHT_COMP: Float = 0.06f // компрессия верхних 12% L*
private const val NEUTRAL_ANCHORS_ON: Boolean = true
private const val NEUTRAL_ANCHORS_N: Int = 6         // 5–7 серых уровней

// [B5-C] Блок-метрики 10×10 (лог)
private const val BLOCK_METRICS_ON: Boolean = true
private const val SKIN_COLORS95_MAX: Float = 8.0f
private const val SKIN_SINGLETON_MAX: Float = 1.5f

object PhotoHQ {

    data class RunResult(
        val image: Bitmap,
        val gridWidth: Int,
        val usedSwatches: List<Swatch>
    )

    private data class Pick(val K: Int, val blur: Float, val mode: PhotoConfig.DitherMode, val score: Float)

    // ABC: локальные пороги (без изменения PhotoConfig)
    private const val EARLY_EDGE_SSIM_MIN = 0.70f
    private const val EARLY_BANDING_MAX   = 0.04f
    private const val EARLY_SSIM_MIN      = 0.66f
    private const val MIN_AFTER_MERGE     = 24  // защита от «схлопывания» палитры

    /** Авто: превью‑подбор K/blur/dither → финальный прогон с фиксированными параметрами. */
    fun runAuto(
        source: Bitmap,
        threadPalette: List<Swatch>,
        preset: PhotoConfig.Params,
        sizes: IntArray,
        enableDescreen: Boolean
    ): RunResult {
        HQLog.pre("runAuto: src=${source.width}x${source.height}, pal=${threadPalette.size}, sizes=${sizes.joinToString()}, descreen=$enableDescreen")

        // 1) превью
        val prev = Downscale.toMaxSide(source, 512)  // ≤512 px — быстрые метрики и рендер кандидатов

        // 2) A: узкий перебор кандидатов (ускорение)
        val mid = (preset.kMin + preset.kMax) / 2
        val Ks = intArrayOf(
            max(preset.kMin, mid - 4),
            mid,
            min(mid + 4, preset.kMax),
            min(mid + 8, preset.kMax)
        ).distinct().sorted()
        val blurs = floatArrayOf(0f)                          // только 0.0
        val modes = arrayOf(PhotoConfig.DitherMode.FS)        // портрет: базово FS

        var best: Pick? = null
        for (K in Ks) {
            for (b in blurs) {
                for (m in modes) {
                    val test = previewCandidate(prev, threadPalette, K, m, b)
                    // метрики качества на превью (твои реализации)
                    val ssim = SsimMeter.ssimY(prev, test, win = 7)                     // ↑ лучше 1.0
                    val edge = EdgeMeter.preserve(prev, test)                            // ↑ лучше 1.0
                    val band = BandingMeter.bandingScore(prev, test, win = 7)            // ↓ лучше 0.0
                    val de95 = DeltaE.de2000Perc95(prev, test, stride = 4)               // ↓ лучше 0.0
                    val score = 0.52f * ssim + 0.23f * edge - 0.20f * band - 0.05f * (de95 / 10f)
                    HQLog.auto("K=$K blur=${"%.3f".format(b)} mode=$m → SSIM=${"%.3f".format(ssim)} Edge=${"%.3f".format(edge)} Band=${"%.3f".format(band)} ΔE95=${"%.3f".format(de95)} score=${"%.3f".format(score)}")
                    if (best == null || score > best!!.score) best = Pick(K, b, m, score)
                }
            }
        }
        val pick = best ?: Pick(preset.kMax, 0f, PhotoConfig.DitherMode.ORDERED, 0f)
        HQLog.auto("picked: K=${pick.K}, blur=${"%.3f".format(pick.blur)}, mode=${pick.mode}, score=${"%.3f".format(pick.score)}")

        // 3) B: собрать и «заморозить» палитру один раз на первом S (сильное ускорение)
        val s0 = sizes.firstOrNull() ?: 200
        val (frozenArgb, frozenLab) = buildPaletteForS(source, threadPalette, pick.K, pick.blur, s0)

        // 4) перебор размеров S c ранним выходом (C)
        var bestS = sizes.firstOrNull() ?: 200
        var bestScore = Float.NEGATIVE_INFINITY
        var bestImage: Bitmap? = null
        var bestUsed: List<Swatch> = emptyList()
        for ((idx, S) in sizes.withIndex()) {
            val (imgS, usedS) = renderWithFrozenPaletteAtS(
                source = source,
                threadPalette = threadPalette,
                paletteArgb = frozenArgb,
                paletteLab = frozenLab,
                preferMode = pick.mode, // базовый выбор
                targetWidthSt = S
            )
            // оценим качество против даунскейла оригинала до ширины S
            val origS = scaleToWidth(source, S)
            val ssim = SsimMeter.ssimY(origS, imgS, win = 7)
            val edge = EdgeMeter.preserve(origS, imgS)
            val band = BandingMeter.bandingScore(origS, imgS, win = 7)
            val de95 = DeltaE.de2000Perc95(origS, imgS, stride = 4)
            val score = 0.52f * ssim + 0.23f * edge - 0.20f * band - 0.05f * (de95 / 10f)
            HQLog.s("S=$S: SSIM=${"%.3f".format(ssim)} Edge=${"%.3f".format(edge)} Band=${"%.3f".format(band)} ΔE95=${"%.3f".format(de95)} score=${"%.3f".format(score)} used=${usedS.size}")
            if (score > bestScore) {
                bestScore = score; bestS = S; bestImage = imgS; bestUsed = usedS
            }
            // C: ранний выход — только если хватает и EdgeSSIM, и обычного SSIM
            if (idx == 0 && edge >= PhotoConfig.B5.EARLY_EDGE_SSIM_MIN &&
                band <= PhotoConfig.B5.EARLY_BANDING_MAX &&
                ssim >= PhotoConfig.B5.EARLY_SSIM_MIN) {
                HQLog.s("Early-exit: S=$S (EdgeSSIM≥${PhotoConfig.B5.EARLY_EDGE_SSIM_MIN}, SSIM≥${PhotoConfig.B5.EARLY_SSIM_MIN}, Banding≤${PhotoConfig.B5.EARLY_BANDING_MAX})")
                break
            }
        }
        val outBmp = bestImage ?: scaleToWidth(source, bestS)
        HQLog.out("done: grid=$bestS, used=${bestUsed.size}")
        return RunResult(outBmp, bestS, bestUsed)
    }

    /** Ручной сценарий: фиксированный S (ширина в стежках), опциональный принудительный dither. */
    fun runManual(
        source: Bitmap,
        threadPalette: List<Swatch>,
        preset: PhotoConfig.Params,
        targetS: Int,
        enableDescreen: Boolean,
        forceDitherMode: PhotoConfig.DitherMode?
    ): RunResult {
        HQLog.pre("runManual: src=${source.width}x${source.height}, pal=${threadPalette.size}, S=$targetS, descreen=$enableDescreen, dither=$forceDitherMode")
        // Даже в manual подберём K по превью в рамках пресета (kMin..kMax), если диапазон > 1
        val prev = Downscale.toMaxSide(source, 512)
        val ks = if (preset.kMin == preset.kMax) listOf(preset.kMax)
        else generateSequence(preset.kMin) { it + max(2, preset.kStep) }
            .takeWhile { it <= preset.kMax }.toList()
        var bestK = preset.kMax
        var bestScore = Float.NEGATIVE_INFINITY
        for (k in ks) {
            val test = previewCandidate(prev, threadPalette, k, forceDitherMode ?: PhotoConfig.DitherMode.ORDERED, 0f)
            val ssim = SsimMeter.ssimY(prev, test, win = 7)
            val edge = EdgeMeter.preserve(prev, test)
            val band = BandingMeter.bandingScore(prev, test, win = 7)
            val de95 = DeltaE.de2000Perc95(prev, test, stride = 4)
            val score = 0.52f * ssim + 0.23f * edge - 0.20f * band - 0.05f * (de95 / 10f)
            if (score > bestScore) { bestScore = score; bestK = k }
        }
        val (img, used) = renderAtSize(
            source = source,
            threadPalette = threadPalette,
            K = bestK,
            blur = 0f,
            preferMode = forceDitherMode ?: PhotoConfig.DitherMode.ORDERED,
            targetWidthSt = targetS
        )
        HQLog.out("done: grid=$targetS, used=${used.size}")
        return RunResult(img, targetS, used)
    }

// ------------------------------ helpers (prev‑рендер кандидатов) ------------------------------

    // Дилатация бинарной маски на r пикселей (для «ленты» вокруг края)
    private fun dilateMask(src: BooleanArray, w: Int, h: Int, r: Int): BooleanArray {
        if (r <= 0) return src
        val out = BooleanArray(src.size)
        val rr = r * r
        for (y in 0 until h) for (x in 0 until w) {
            var hit = false
            var dy = -r
            while (!hit && dy <= r) {
                val yy = y + dy; if (yy !in 0 until h) { dy++; continue }
                var dx = -r
                while (!hit && dx <= r) {
                    val xx = x + dx; if (xx !in 0 until w) { dx++; continue }
                    if (dx*dx + dy*dy <= rr && src[yy*w + xx]) hit = true
                    dx++
                }
                dy++
            }
            out[y*w + x] = hit
        }
        return out
    }

    // Быстрый индикатор риска бэндинга по L* (окно 7×7)
    private fun bandingMask7(l: FloatArray, w: Int, h: Int,
                             varThr: Float = 0.0035f, gradThr: Float = 0.010f): BooleanArray {
        val out = BooleanArray(l.size)
        val win = 7; val r = win / 2
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0f; var s2 = 0f; var cnt = 0; var g = 0f
            val y0 = (y - r).coerceAtLeast(0); val y1 = (y + r).coerceAtMost(h - 1)
            val x0 = (x - r).coerceAtLeast(0); val x1 = (x + r).coerceAtMost(w - 1)
            var yy = y0
            while (yy <= y1) {
                val row = yy * w
                var xx = x0
                while (xx <= x1) {
                    val v = l[row + xx]; s += v; s2 += v * v; cnt++
                    if (xx + 1 <= x1) g = maxOf(g, kotlin.math.abs(l[row + xx + 1] - v))
                    if (yy + 1 <= y1) g = maxOf(g, kotlin.math.abs(l[(yy + 1) * w + xx] - v))
                    xx++
                }
                yy++
            }
            val mean = s / cnt
            val varL = (s2 / cnt) - mean * mean
            out[y*w + x] = (varL < varThr && g < gradThr)
        }
        return out
    }

    /** Построить превью‑кандидата (квантовать в K ниток палитры и задизерить режимом [mode]). */
    private fun previewCandidate(
        prev: Bitmap,
        threadPalette: List<Swatch>,
        K: Int,
        mode: PhotoConfig.DitherMode,
        blur: Float
    ): Bitmap {
        // Локальный источник для превью-рендера
        val srcPrev = if (blur > 0f) gauss3x3(prev) else prev
        var (allowedArgb, allowedLab) = buildAllowedFromKMeans(srcPrev, threadPalette, K)
        // Расширяем разнообразие: разнести дубликаты по альтернативным ниткам
        val diversified = diversifyDuplicatesBeforeMerge(allowedArgb, allowedLab, threadPalette)
        allowedArgb = diversified.first; allowedLab = diversified.second
        // Protected-merge: берём защиту по референс-превью (≤512) от исходного prev
        val prot = if (PhotoConfig.B4.PROTECTED_MERGE)
            EdgeGuard.protectedColors(Downscale.toMaxSide(prev, 512), allowedArgb) else IntArray(0)
        val merged = mergeAllowedLab(allowedArgb, allowedLab, PhotoConfig.B2.MERGE_LAB_SQ, prot)
        allowedArgb = merged.first; allowedLab = merged.second
        val raster = Raster(srcPrev.width, srcPrev.height, bitmapToIntArray(srcPrev))
        val outRaster = when (mode) {
            PhotoConfig.DitherMode.NONE -> quantNearest(raster, allowedLab, allowedArgb)
            PhotoConfig.DitherMode.FS -> dither(raster, allowedLab, allowedArgb, Metric.OKLAB, Dither.FLOYD_STEINBERG)
            PhotoConfig.DitherMode.ORDERED -> ditherOrderedBayer8(
                raster, allowedLab, allowedArgb,
                amp = PhotoConfig.B2.ORDERED_AMP, metric = Metric.OKLAB
            )
        }
        return intArrayToBitmap(outRaster.argb, srcPrev.width, srcPrev.height)
    }

    /** Строим K центров (OKLab‑kmeans) и «снэпаем» каждый центр к ближайшей нитке из активной палитры. */
    private fun buildAllowedFromKMeans(src: Bitmap, threads: List<Swatch>, K: Int): Pair<IntArray, FloatArray> {
        val w = src.width; val h = src.height
        val px = IntArray(w * h)
        src.getPixels(px, 0, w, 0, 0, w, h)
        val imgLab = argbToOkLab(px)                                 // OKLab для всех пикселей
        val centers = kmeansLab(imgLab, K, iters = 6, seed = 42)      // K центров в OKLab (твоя реализация)

        // Строим ARGB-массив из текущей палитры ниток (без сторонних экстеншенов)
        val threadArgb = IntArray(threads.size) { idx -> threads[idx].argb }
        val threadLab = argbToOkLab(threadArgb)
        // для каждого центра — ближайшая нитка
        val picked = IntArray(K) { 0 }
        var uniq = 0
        var c = 0
        while (c < K) {
            val j = c * 3
            val L = centers[j]; val A = centers[j + 1]; val B = centers[j + 2]
            var best = 0; var bestD = Float.POSITIVE_INFINITY; var i = 0
            while (i < threadLab.size) {
                val dl = threadLab[i] - L
                val da = threadLab[i + 1] - A
                val db = threadLab[i + 2] - B
                val d = dl*dl + da*da + db*db
                if (d < bestD) { bestD = d; best = i / 3 }
                i += 3
            }
            picked[c] = threadArgb[best]
            c++
        }
        // лёгкая уникализация (возможны дубли при «схлопывании» центров)
        val uniqList = picked.toMutableList().distinct()
        val allowedArgb = uniqList.toIntArray()
        val allowedLab = argbToOkLab(allowedArgb)
        return allowedArgb to allowedLab
    }

    // B: собрать палитру (RAQ+merge+refill) один раз на первом S и вернуть её (ARGB+Lab)
    private fun buildPaletteForS(
        source: Bitmap,
        threadPalette: List<Swatch>,
        K: Int,
        blur: Float,
        targetWidthSt: Int
    ): Pair<IntArray, FloatArray> {
        val srcS = scaleToWidth(source, targetWidthSt)
        val srcForClustering = if (blur > 0f) gauss3x3(srcS) else srcS
        // ── RAQ (замороженная палитра на первом S): база + капы по зонам (включая SKIN/GROUND/CLOUD)
        val wz = ZoneMasks.compute(srcS)
        val edgeMaskStrong = EdgeMeter.strongEdgeMask(srcS, 1.25f)
        // 1) База (глобальные центры)
        val baseK = kotlin.math.min(PhotoConfig.B5.BASE_MIN, K.coerceAtLeast(0))
        val base = if (baseK > 0) buildAllowedFromKMeans(srcForClustering, threadPalette, baseK)
        else (IntArray(0) to FloatArray(0))
            // 2) Капы по зонам
        val rest = (K - baseK).coerceAtLeast(0)
        var caps = if (rest > 0) {
            val b = RAQBounds(
                PhotoConfig.B5.SKY_MIN,   PhotoConfig.B5.SKY_MAX,
                PhotoConfig.B5.CLOUD_MIN, PhotoConfig.B5.CLOUD_MAX,
                PhotoConfig.B5.WATER_MIN, PhotoConfig.B5.WATER_MAX,
                PhotoConfig.B5.VEG_MIN,   PhotoConfig.B5.VEG_MAX,
                PhotoConfig.B5.GROUND_MIN,PhotoConfig.B5.GROUND_MAX,
                PhotoConfig.B5.BUILT_MIN, PhotoConfig.B5.BUILT_MAX,
                PhotoConfig.B5.SKIN_MIN,  PhotoConfig.B5.SKIN_MAX
            )
            PaletteAllocator.allocateCaps(wz, edgeMaskStrong, rest, b)
        } else emptyMap()
        // Жёстко гарантируем мин. для кожи/тёмного фона; избыток снимаем с второстепенных зон
        fun enforceMinCaps(c: MutableMap<Zone, Int>, restAll: Int) {
            fun raise(z: Zone, minV: Int) { c[z] = maxOf(c[z] ?: 0, minV) }
            raise(Zone.SKIN,   PhotoConfig.B5.SKIN_MIN)
            raise(Zone.GROUND, PhotoConfig.B5.GROUND_MIN)
            var sum = c.values.sum()
            if (sum > restAll) {
                var over = sum - restAll
                val donors = arrayOf(Zone.BUILT, Zone.VEG, Zone.WATER, Zone.SKY, Zone.CLOUD)
                for (z in donors) {
                    if (over <= 0) break
                    val cur = c[z] ?: 0
                    val dec = minOf(cur, over)
                    c[z] = cur - dec
                    over -= dec
                }
            }
        }
        if (rest > 0) {
            val fixed = caps.toMutableMap()
            enforceMinCaps(fixed, rest)
            caps = fixed
            HQLog.auto("RAQ.capsFixed=$caps")
        }
        HQLog.auto("RAQ.before=$K caps=$caps (base=$baseK rest=$rest)")
        // 3) Зональные центры по капам
        val zoneArgb = ArrayList<Int>()
        val zoneLab = ArrayList<Float>()
        if (rest > 0) {
            val pxS = bitmapToIntArray(srcForClustering)
            val labS = argbToOkLab(pxS)
            fun addCenters(mask: FloatArray, kZone: Int) {
                if (kZone <= 0) return
                val sub = pickLabSubsetByWeight(labS, mask, srcForClustering.width, srcForClustering.height,
                    maxCount = 20000, keepFrac = 0.35f)
                if (sub.isEmpty()) return
                val centers = kmeansLab(sub, kZone, iters = 5, seed = 131)
                val snapped = snapCentersToThreads(centers, threadPalette)
                val snappedLab = argbToOkLab(snapped)
                for (c in snapped) zoneArgb.add(c)
                var j = 0; while (j < snappedLab.size) { zoneLab.add(snappedLab[j++]) }
            }
            addCenters(wz.sky,    caps[Zone.SKY]   ?: 0)
            addCenters(wz.cloud,  caps[Zone.CLOUD] ?: 0)
            addCenters(wz.water,  caps[Zone.WATER] ?: 0)
            addCenters(wz.veg,    caps[Zone.VEG]   ?: 0)
            addCenters(wz.ground, caps[Zone.GROUND]?: 0)
            addCenters(wz.built,  caps[Zone.BUILT] ?: 0)
            addCenters(wz.skin,   caps[Zone.SKIN]  ?: 0)
        }
        // 4) Сборка итогового множества
        var allowedArgb = IntArray(0); var allowedLab = FloatArray(0)
        fun append(argb: IntArray, lab: FloatArray) {
            if (argb.isEmpty()) return
            val newA = IntArray(allowedArgb.size + argb.size)
            System.arraycopy(allowedArgb, 0, newA, 0, allowedArgb.size)
            System.arraycopy(argb, 0, newA, allowedArgb.size, argb.size)
            allowedArgb = newA
            val newL = FloatArray(allowedLab.size + lab.size)
            System.arraycopy(allowedLab, 0, newL, 0, allowedLab.size)
            System.arraycopy(lab, 0, newL, allowedLab.size, lab.size)
            allowedLab = newL
        }
        append(base.first, base.second)
        append(zoneArgb.toIntArray(), zoneLab.toFloatArray())
        // 5) Protected-цвета: нейтральные и skin-якоря
        val anchorsNeutral = if (NEUTRAL_ANCHORS_ON) neutralAnchorsARGB(NEUTRAL_ANCHORS_N) else IntArray(0)
        val anchorsSkin = skinAnchorsARGB(srcS, threadPalette, intArrayOf(35, 55, 70, 82))
        append(anchorsNeutral, argbToOkLab(anchorsNeutral))
        append(anchorsSkin, argbToOkLab(anchorsSkin))
        val prot = (EdgeGuard.protectedColors(Downscale.toMaxSide(source, 512), allowedArgb)
                + anchorsNeutral + anchorsSkin).distinct().toIntArray()
        val beforeMerge = allowedArgb.size
        val merged = mergeAllowedLab(allowedArgb, allowedLab, PhotoConfig.B2.MERGE_LAB_SQ, prot)
        allowedArgb = merged.first; allowedLab = merged.second
        val afterMerge = allowedArgb.size
        // 6) Гарантированный минимум после merge — при необходимости добираем по ошибке
        if (allowedArgb.size < PhotoConfig.B5.MIN_AFTER_MERGE) {
            val rasterS = Raster(srcS.width, srcS.height, bitmapToIntArray(srcS))
            val preNoDither = quantNearest(rasterS, allowedLab, allowedArgb)
            val need = PhotoConfig.B5.MIN_AFTER_MERGE - allowedArgb.size
            val (argbR, labR) = refillByError(srcS, preNoDither, threadPalette, need)
            allowedArgb = allowedArgb + argbR
            allowedLab  = allowedLab  + labR
            val merged2 = mergeAllowedLab(allowedArgb, allowedLab, PhotoConfig.B2.MERGE_LAB_SQ, prot)
            allowedArgb = merged2.first; allowedLab = merged2.second
            HQLog.auto("RAQ.refill: +$need → palette=${allowedArgb.size}")
        }
        // 7) Логи RAQ
        val npx = (srcS.width * srcS.height).coerceAtLeast(1)
        val share = fun(mask: FloatArray) = (mask.sum() / npx).coerceIn(0f,1f)
        HQLog.auto("RAQ.afterMerge=$afterMerge K0=$beforeMerge → Kfinal=${allowedArgb.size}")
        HQLog.auto("zonesShare={SKIN=${"%.1f".format(100f*share(wz.skin))}%, SKY=${"%.1f".format(100f*share(wz.sky))}%, WATER=${"%.1f".format(100f*share(wz.water))}%, VEG=${"%.1f".format(100f*share(wz.veg))}%, GROUND=${"%.1f".format(100f*share(wz.ground))}%, BUILT=${"%.1f".format(100f*share(wz.built))}%}")
        HQLog.auto("protectedKept=${prot.size}")
        HQLog.auto("RAQ(frozen): palette=${allowedArgb.size}")
        return allowedArgb to allowedLab
    }

    /** Простой merge очень близких цветов (евклид в OKLab^2), без EdgeGuard. */
    private fun mergeAllowedLab(
        argb: IntArray,
        lab: FloatArray,
        thrSq: Float,
        protectedArgb: IntArray? = null
    ): Pair<IntArray, FloatArray> {
        if (argb.isEmpty()) return argb to lab
        val keep = BooleanArray(argb.size) { true }
        val protSet = protectedArgb?.toHashSet() ?: emptySet()
        // дельта-порог для skin/нейтралей (чуть жёстче базового)
        val skinScale = 0.60f
        val neutralScale = 0.70f
        val neutralChromaSq = 0.035f * 0.035f
        val stepGuardMin = 0.015f
        val stepGuardMax = 0.080f
        val mergedDistances = ArrayList<Float>(16)
        var i = 0
        while (i < argb.size) {
            if (keep[i]) {
                val j0 = i * 3
                val l0 = lab[j0]; val a0 = lab[j0 + 1]; val b0 = lab[j0 + 2]
                var j = i + 1
                while (j < argb.size) {
                    if (keep[j]) {
                        // Не сливаем пару, если один из цветов защищён
                        if (protSet.contains(argb[i]) || protSet.contains(argb[j])) { j++; continue }
                        val jj = j * 3
                        val dl = lab[jj] - l0
                        val da = lab[jj + 1] - a0
                        val db = lab[jj + 2] - b0
                        // Анти-перетемнение
                        if (kotlin.math.abs(dl) > 0.06f) { j++; continue }
                        val d2 = dl*dl + da*da + db*db
                        // Определяем тип пары
                        val isNeutralPair = (a0*a0 + b0*b0) < neutralChromaSq &&
                                (lab[jj+1]*lab[jj+1] + lab[jj+2]*lab[jj+2]) < neutralChromaSq
                        val isSkinPair = isSkinArgbColor(argb[i]) && isSkinArgbColor(argb[j])
                        // Gradient-Guard: для skin/нейтралей не схлопывать «ступени» по L*
                        val absDl = kotlin.math.abs(dl)
                        if ((isNeutralPair || isSkinPair) && absDl in stepGuardMin..stepGuardMax) { j++; continue }
                        val thr2 = when {
                            isSkinPair    -> thrSq * skinScale
                            isNeutralPair -> thrSq * neutralScale
                            else          -> thrSq
                        }
                        if (d2 <= thr2) {
                            keep[j] = false
                            mergedDistances.add(kotlin.math.sqrt(d2))
                        }
                    }
                    j++
                }
            }
            i++
        }
        val outArgb = ArrayList<Int>(argb.size)
        val outLab = ArrayList<Float>(lab.size)
        for (k in argb.indices) if (keep[k]) {
            outArgb.add(argb[k])
            val kk = k * 3
            outLab.add(lab[kk]); outLab.add(lab[kk + 1]); outLab.add(lab[kk + 2])
        }
        if (mergedDistances.isNotEmpty()) {
            val top = mergedDistances.sorted().take(6).joinToString(prefix="[", postfix="]") { "%.3f".format(it) }
            HQLog.auto("topMergePairsΔ≈ $top")
        }
        return outArgb.toIntArray() to outLab.toFloatArray()
    }

    /** Быстрые skin-якоря: берём средний оттенок кожи и фиксируем 3–4 уровня L*. */
    private fun skinAnchorsARGB(src: Bitmap, threads: List<Swatch>, levelsL: IntArray): IntArray {
        val mask = skinMaskYCbCr(src)
        val w = src.width; val h = src.height; val n = w*h
        val px = IntArray(n); src.getPixels(px,0,w,0,0,w,h)
        var sr = 0; var sg = 0; var sb = 0; var cnt = 0
        var i = 0
        while (i < n) {
            if (mask[i]) {
                val p = px[i]
                sr += (p ushr 16) and 0xFF
                sg += (p ushr 8) and 0xFF
                sb += p and 0xFF
                cnt++
            }
            i++
        }
        if (cnt == 0) return IntArray(0)
        val r = (sr / cnt).coerceIn(0,255)
        val g = (sg / cnt).coerceIn(0,255)
        val b = (sb / cnt).coerceIn(0,255)
        val baseLab = argbToOkLab(intArrayOf((0xFF shl 24) or (r shl 16) or (g shl 8) or b))
        val a0 = baseLab[1] * 0.65f // слегка уменьшаем насыщенность
        val b0 = baseLab[2] * 0.65f
        val centers = FloatArray(levelsL.size * 3)
        var c = 0
        for (L8 in levelsL) {
            val L = (L8 / 100f).coerceIn(0.0f, 1.0f)
            centers[c] = L; centers[c+1] = a0; centers[c+2] = b0
            c += 3
        }
        return snapCentersToThreads(centers, threads)
    }

    /** Heuristic: skin check in YCbCr (single color). */
    private fun isSkinArgbColor(c: Int): Boolean {
        val r = (c ushr 16) and 0xFF
        val g = (c ushr 8) and 0xFF
        val b = c and 0xFF
        val y  =  0.299f * r + 0.587f * g + 0.114f * b
        val cb = -0.1687f * r - 0.3313f * g + 0.5f   * b + 128f
        val cr =  0.5f   * r - 0.4187f * g - 0.0813f * b + 128f
        return (y > 70f && cb in 85f..135f && cr in 133f..180f)
    }

    /**
     * Разнообразить палитру до merge: если несколько центров «соскакивают» в одну и ту же нитку,
     * пытаемся для дубликатов выбрать альтернативную нитку из ближайших (c «не темнить» штрафом).
     * Возвращаем такой же (argb, lab), но без повторов по argb (насколько возможно).
     */
    private fun diversifyDuplicatesBeforeMerge(
        argb: IntArray,
        lab: FloatArray,
        threads: List<Swatch>,
        maxAlternatives: Int = 6,
        lightnessPenalty: Float = 0.20f
    ): Pair<IntArray, FloatArray> {
        if (argb.isEmpty()) return argb to lab
        val threadArgb = IntArray(threads.size) { threads[it].argb }
        val threadLab = argbToOkLab(threadArgb)
        val used = HashSet<Int>(argb.size * 2)
        val outA = IntArray(argb.size)
        val outL = FloatArray(lab.size)
        var i = 0; var k = 0
        while (i < argb.size) {
            val L = lab[k]; val A = lab[k + 1]; val B = lab[k + 2]
            var pick = -1
            var best = Float.POSITIVE_INFINITY
            // 1) если текущая нитка ещё не занята — оставляем как есть
            if (used.add(argb[i])) {
                pick = argb[i]
            } else {
                // 2) ищем альтернативу среди ближайших ниток
                // грубо: один проход по всем ниткам c сортировкой «на лету»
                // (для простоты и предсказуемости)
                var j = 0; var chosen = -1
                var rank = 0
                // соберём top-N кандидатов
                val candIdx = ArrayList<Int>(maxAlternatives)
                val candCost = ArrayList<Float>(maxAlternatives)
                while (j < threadLab.size) {
                    val t = j / 3
                    val dl = threadLab[j] - L
                    val da = threadLab[j + 1] - A
                    val db = threadLab[j + 2] - B
                    val d = dl*dl + da*da + db*db
                    val cost = d + lightnessPenalty * kotlin.math.abs(dl)
                    // вставка в топ-N
                    var pos = candCost.indexOfFirst { cost < it }
                    if (pos == -1) pos = candCost.size
                    if (pos < maxAlternatives) {
                        candIdx.add(pos, t)
                        candCost.add(pos, cost)
                        if (candIdx.size > maxAlternatives) { candIdx.removeLast(); candCost.removeLast() }
                    }
                    j += 3
                }
                for (idx in candIdx) {
                    val cArgb = threadArgb[idx]
                    if (used.add(cArgb)) { chosen = cArgb; break }
                }
                if (chosen != -1) pick = chosen else pick = argb[i] // fallback
            }
            outA[i] = pick
            // lab соответствуем выбранной нитке (точная OKLab выбранной нитки)
            val tid = threadArgb.indexOf(pick).coerceAtLeast(0)
            val tj = tid * 3
            if (tj in 0 until threadLab.size) {
                outL[k] = threadLab[tj]; outL[k + 1] = threadLab[tj + 1]; outL[k + 2] = threadLab[tj + 2]
            } else {
                // на всякий случай — сохраняем исходный центр
                outL[k] = L; outL[k + 1] = A; outL[k + 2] = B
            }
            i++; k += 3
        }
        return outA to outL
    }

    // ------------------------------ B5 helpers: зональные выборки и «снэп» центров ------------------------------
    /** Выбрать подмножество LAB-пикселей по весам зоны. */
    private fun pickLabSubsetByWeight(
        labFull: FloatArray,
        weight: FloatArray,
        w: Int, h: Int,
        maxCount: Int,
        keepFrac: Float
    ): FloatArray {
        val n = w * h
        val idx = (0 until n).sortedByDescending { weight[it] }
        val take = (n * keepFrac).toInt().coerceIn(1, n)
        val step = maxOf(1, take / maxCount.coerceAtLeast(1))
        val out = ArrayList<Float>(maxCount * 3)
        var c = 0; var i = 0
        while (i < take && c < maxCount) {
            val p = idx[i]
            val j3 = p * 3
            out.add(labFull[j3 + 0]); out.add(labFull[j3 + 1]); out.add(labFull[j3 + 2])
            i += step; c++
        }
        return out.toFloatArray()
    }

    /** Снэп центров OKLab к ближайшим ниткам активной палитры. */
    private fun snapCentersToThreads(centersLab: FloatArray, threads: List<Swatch>): IntArray {
        val threadArgb = IntArray(threads.size) { threads[it].argb }
        val threadLab = argbToOkLab(threadArgb)
        val out = IntArray(centersLab.size / 3)
        val used = HashSet<Int>()
        var c = 0
        while (c < centersLab.size) {
            val L = centersLab[c]; val A = centersLab[c + 1]; val B = centersLab[c + 2]
            var pick = -1; var bestCost = Float.POSITIVE_INFINITY
            var tries = 0; var i = 0
            while (i < threadLab.size) {
                val j = i / 3
                val dl = threadLab[i] - L
                val da = threadLab[i + 1] - A
                val db = threadLab[i + 2] - B
                val d = dl*dl + da*da + db*db
                val cost = d + 0.20f * kotlin.math.abs(dl)   // «не темнить»
                if (cost < bestCost && (j !in used || tries >= 4)) {
                    bestCost = cost; pick = j
                }
                i += 3
            }
            used.add(pick)
            out[c / 3] = threadArgb[pick]
            c += 3
        }
        return out
    }

    // ------------------------------ финальный HQ‑рендер в выбранный S ------------------------------

    /** Полный рендер одного размера S: protected‑merge + гибридный дизеринг (ORDERED в banding, FS иначе). */
    private fun renderAtSize(
        source: Bitmap,
        threadPalette: List<Swatch>,
        K: Int,
        blur: Float,
        preferMode: PhotoConfig.DitherMode,
        targetWidthSt: Int
    ): Pair<Bitmap, List<Swatch>> {
        // 1) масштаб в S (каждый пиксель = 1 стежок)
        val srcS = scaleToWidth(source, targetWidthSt)
        val srcForClustering = if (blur > 0f) gauss3x3(srcS) else srcS
        // --- B5: зоны + RAQ распределение палитры ---
        val wz = ZoneMasks.compute(srcS)
        val edgeMaskStrong = EdgeMeter.strongEdgeMask(srcS, 1.25f)
        // База (глобальные центры)
        val baseK = kotlin.math.min(PhotoConfig.B5.BASE_MIN, K.coerceAtLeast(0))
        val base = if (baseK > 0) {
            val (argbB, labB) = buildAllowedFromKMeans(srcForClustering, threadPalette, baseK)
            argbB to labB
        } else IntArray(0) to FloatArray(0)
        // RAQ по зонам (остаток после базы)
        val rest = (K - baseK).coerceAtLeast(0)
        var caps = if (rest > 0) {
            val b = RAQBounds(
                PhotoConfig.B5.SKY_MIN,   PhotoConfig.B5.SKY_MAX,
                PhotoConfig.B5.CLOUD_MIN, PhotoConfig.B5.CLOUD_MAX,
                PhotoConfig.B5.WATER_MIN, PhotoConfig.B5.WATER_MAX,
                PhotoConfig.B5.VEG_MIN,   PhotoConfig.B5.VEG_MAX,
                PhotoConfig.B5.GROUND_MIN,PhotoConfig.B5.GROUND_MAX,
                PhotoConfig.B5.BUILT_MIN, PhotoConfig.B5.BUILT_MAX,
                PhotoConfig.B5.SKIN_MIN,  PhotoConfig.B5.SKIN_MAX
            )
            PaletteAllocator.allocateCaps(wz, edgeMaskStrong, rest, b)
        } else emptyMap()
        // ── Жёсткая гарантия min для кожи/тёмного фона; избыток снимаем с «менее критичных» зон
        if (rest > 0) {
            val fixed = caps.toMutableMap()
            fun maxTo(key: Zone, minVal: Int) { fixed[key] = kotlin.math.max(fixed[key] ?: 0, minVal) }
            maxTo(Zone.SKIN,   PhotoConfig.B5.SKIN_MIN)
            maxTo(Zone.GROUND, PhotoConfig.B5.GROUND_MIN)
            var sum = fixed.values.sum()
            if (sum > rest) {
                var overflow = sum - rest
                // порядок отдачи: BUILT → VEG → WATER → SKY → CLOUD
                val donors = arrayOf(Zone.BUILT, Zone.VEG, Zone.WATER, Zone.SKY, Zone.CLOUD)
                for (z in donors) {
                    if (overflow <= 0) break
                    val cur = (fixed[z] ?: 0)
                    val dec = kotlin.math.min(cur, overflow)
                    fixed[z] = cur - dec
                    overflow -= dec
                }
            }
            caps = fixed
        }
        // Жёсткая гарантия min для SKIN/GROUND, лишнее снимаем с второстепенных зон
        if (rest > 0) {
            val fixed = caps.toMutableMap()
            fun raise(z: Zone, minVal: Int) { fixed[z] = maxOf(fixed[z] ?: 0, minVal) }
            raise(Zone.SKIN,   PhotoConfig.B5.SKIN_MIN)
            raise(Zone.GROUND, PhotoConfig.B5.GROUND_MIN)
            var sum = fixed.values.sum()
            if (sum > rest) {
                var over = sum - rest
                val donors = arrayOf(Zone.BUILT, Zone.VEG, Zone.WATER, Zone.SKY, Zone.CLOUD)
                for (z in donors) {
                    if (over <= 0) break
                    val cur = fixed[z] ?: 0
                    val dec = minOf(cur, over)
                    fixed[z] = cur - dec
                    over -= dec
                }
            }
            caps = fixed
        }
        HQLog.auto("RAQ: base=$baseK rest=$rest caps=$caps")

        // Зональные центры
        val zoneArgb = ArrayList<Int>()
        val zoneLab = ArrayList<Float>()
        if (rest > 0) {
            // подготовим LAB массива для исходника S (чтобы выбирать подмножества)
            val pxS = bitmapToIntArray(srcForClustering)
            val labS = argbToOkLab(pxS)
            fun addCentersForZone(mask: FloatArray, k: Int) {
                if (k <= 0) return
                val sub = pickLabSubsetByWeight(labS, mask, srcForClustering.width, srcForClustering.height,
                    maxCount = PhotoConfig.B5.ZONE_SAMPLE_MAX, keepFrac = PhotoConfig.B5.ZONE_KEEP_FRAC)
                if (sub.isEmpty()) return
                val centers = kmeansLab(sub, k, iters = 5, seed = 131)
                // снап к ближайшим ниткам
                val snapped = snapCentersToThreads(centers, threadPalette)
                val snappedLab = argbToOkLab(snapped)
                for (c in snapped) zoneArgb.add(c)
                var j = 0; while (j < snappedLab.size) { zoneLab.add(snappedLab[j++]) }
            }
            // SKIN в приоритете (портреты)
            addCentersForZone(wz.skin,  caps[Zone.SKIN]  ?: 0)
            addCentersForZone(wz.sky,   caps[Zone.SKY]   ?: 0)
            addCentersForZone(wz.cloud, caps[Zone.CLOUD] ?: 0)
            addCentersForZone(wz.water, caps[Zone.WATER] ?: 0)
            addCentersForZone(wz.veg,   caps[Zone.VEG]   ?: 0)
            addCentersForZone(wz.ground,caps[Zone.GROUND]?: 0)
            addCentersForZone(wz.built, caps[Zone.BUILT] ?: 0)
        }
        // Сборка итогового допустимого множества
        var allowedArgb = IntArray(0)
        var allowedLab = FloatArray(0)
        fun append(argb: IntArray, lab: FloatArray) {
            if (argb.isEmpty()) return
            val newA = IntArray(allowedArgb.size + argb.size)
            System.arraycopy(allowedArgb, 0, newA, 0, allowedArgb.size)
            System.arraycopy(argb, 0, newA, allowedArgb.size, argb.size)
            allowedArgb = newA
            val newL = FloatArray(allowedLab.size + lab.size)
            System.arraycopy(allowedLab, 0, newL, 0, allowedLab.size)
            System.arraycopy(lab, 0, newL, allowedLab.size, lab.size)
            allowedLab = newL
        }
        append(base.first, base.second)
        append(zoneArgb.toIntArray(), zoneLab.toFloatArray())
        // Protected‑merge на референс‑превью (≤512)
        val ref512 = Downscale.toMaxSide(source, 512)
        val prot = if (PhotoConfig.B4.PROTECTED_MERGE)
            EdgeGuard.protectedColors(ref512, allowedArgb) else IntArray(0)
        var merged = mergeAllowedLab(allowedArgb, allowedLab, PhotoConfig.B2.MERGE_LAB_SQ, prot)
        allowedArgb = merged.first; allowedLab = merged.second

        // Подготовим растр и no-dither для последующих шагов (и для возможного refill)
        val rasterS = Raster(srcS.width, srcS.height, bitmapToIntArray(srcS))
        var preNoDither = quantNearest(rasterS, allowedLab, allowedArgb)
        // Гарантия минимума цветов для портрета (если провалились из-за merge)
        if (allowedArgb.size < PhotoConfig.B5.MIN_AFTER_MERGE) {
            val need = PhotoConfig.B5.MIN_AFTER_MERGE - allowedArgb.size
            val (argbR, labR) = refillByError(srcS, preNoDither, threadPalette, need)
            allowedArgb = allowedArgb + argbR
            allowedLab  = allowedLab  + labR
            merged = mergeAllowedLab(allowedArgb, allowedLab, PhotoConfig.B2.MERGE_LAB_SQ, prot)
            allowedArgb = merged.first; allowedLab = merged.second
            // пересчёт no-dither уже на дополненной палитре
            preNoDither = quantNearest(rasterS, allowedLab, allowedArgb)
            HQLog.auto("RAQ.refill: +$need → palette=${allowedArgb.size}")
        }
        HQLog.auto("RAQ: palette total=${allowedArgb.size} (after merge+protect)")

        // 2) предварительный no-dither квант для banding-mask (уже есть preNoDither)
        val imgNoDither = intArrayToBitmap(preNoDither.argb, srcS.width, srcS.height)
        val bMask = BandingMeter.bandingMask(srcS, imgNoDither, win = 7) // true → риск бэндинга
        val bShare = bMask.count { it }.toFloat() / (srcS.width * srcS.height).toFloat()
        HQLog.mask("S=$targetWidthSt bandingWin=${"%.1f".format(100f * bShare)}% win=7")
        // Плоские зоны (низкий градиент)
        val flat = EdgeMeter.flatMask(srcS, PhotoConfig.B5.FLAT_GRAD_T)

        // 3) два кандидата: ORDERED (для banding‑окна) и FS (для краёв/текстур)
        val ordered = ditherOrderedBayer8(rasterS, allowedLab, allowedArgb, amp = PhotoConfig.B2.ORDERED_AMP, metric = Metric.OKLAB)
        val fs = when (preferMode) {
            PhotoConfig.DitherMode.NONE -> preNoDither
            PhotoConfig.DitherMode.FS, PhotoConfig.DitherMode.ORDERED -> {
                if (PhotoConfig.B4.ANISO_FS) {
                    val (tx, ty) = EdgeMeter.tangentField(srcS)
                    HQLog.auto("dither: fsAniso=ON along=${PhotoConfig.B4.FS_ALONG} across=${PhotoConfig.B4.FS_ACROSS}")
                    ditherFsAniso(rasterS, allowedLab, allowedArgb, Metric.OKLAB, tx, ty, PhotoConfig.B4.FS_ALONG, PhotoConfig.B4.FS_ACROSS)
                } else {
                        HQLog.auto("dither: fsAniso=OFF")
                    dither(rasterS, allowedLab, allowedArgb, Metric.OKLAB, Dither.FLOYD_STEINBERG)
                }
            }
        }

        // 4) пер-пиксельный гибрид:
        //    — ORDERED в banding-окнах,
        //    — NO-DITHER в плоских областях (если включено),
        //    — FS/FS-aniso на остальных.
        val out = IntArray(rasterS.argb.size)
        var i = 0
        // порог для кожи и тёмного фона
        val luma = luma01(srcS)
        while (i < out.size) {
            val isSkin = wz.skin[i] > 0.5f
            val isDark = luma[i] < PhotoConfig.B5.DARK_L_T
            out[i] =
                if (bMask[i]) {
                    // риск бэндинга — мягкий ordered
                    ordered.argb[i]
                } else if (isSkin && !edgeMaskStrong[i]) {
                    // кожа — ordered меньшей амплитудой (ровная «тканевая» текстура)
                    ordered.argb[i]
                    } else if (PhotoConfig.B5.NO_DITHER_IN_FLAT && flat[i]) {
                        preNoDither.argb[i]
                    } else if (isDark && !edgeMaskStrong[i]) {
                        // очень тёмный фон — без дезера
                    preNoDither.argb[i]
                    } else {
                        fs.argb[i]
                    }
            i++
        }

        // 5) (опц.) пост-чистка «песка»: одиночные пиксели → цвет большинства
        if (PhotoConfig.B5.CLEAN_SINGLETONS) {
            cleanSingletonsMajority1(
                out, srcS.width, srcS.height,
                protect = edgeMaskStrong, // не трогаем сильные края
                scopeFlat = flat          // работаем в плоских зонах
            )

        }

        // 6) (опц.) SLIC-lite: стабилизируем плоские регионы (не трогаем сильные края)
        if (PhotoConfig.B5.SLIC_ON) {
            val labels = SlicLite.segmentGrid(srcS, PhotoConfig.B5.SLIC_REGIONS)
            val edge = edgeMaskStrong
            val n = out.size
            // Для SKY/WATER регионы объединяем к «мажорному» цвету
            val sky = wz.sky; val water = wz.water
            val tmp = out.clone()
            val regionMap = HashMap<Int, MutableMap<Int, Int>>(128)
            var p = 0
            while (p < n) {
                if (!edge[p] && (sky[p] + water[p] > 0.4f)) {
                    val id = labels[p]
                    val map = regionMap.getOrPut(id) { HashMap() }
                    map[tmp[p]] = (map[tmp[p]] ?: 0) + 1
                }
                p++
            }
            val majority = HashMap<Int, Int>(regionMap.size)
            for ((id, hist) in regionMap) {
                var bestColor = 0; var bestCnt = 0
                for ((c, cnt) in hist) if (cnt > bestCnt) { bestCnt = cnt; bestColor = c }
                val sum = hist.values.sum().coerceAtLeast(1)
                if (bestCnt.toFloat() / sum >= PhotoConfig.B5.SLIC_MAJORITY_THRESHOLD)
                    majority[id] = bestColor
            }
            var q = 0
            while (q < n) {
                if (!edge[q] && (sky[q] + water[q] > 0.4f)) {
                    val m = majority[labels[q]]
                    if (m != null) out[q] = m
                }
                q++
            }
            HQLog.auto("slic: regions=${majority.size} majority-applied")
        }

        val bmp = intArrayToBitmap(out, srcS.width, srcS.height)

        // 5) список реально использованных Swatch (по финальному изображению)
        val used = collectUsedSwatches(threadPalette, out)
        // [B5-C] Блок-метрики 10×10 (глобально и по skin-маске)
        if (BLOCK_METRICS_ON) {
            val bmAll = computeBlockMetrics(out, srcS.width, srcS.height, null)
            // Локально считаем маску кожи, чтобы не зависеть от области видимости переменной
            val bmSkin = computeBlockMetrics(out, srcS.width, srcS.height, skinMaskYCbCr(srcS))
            HQLog.s("block: Colors95=${"%.2f".format(bmAll.colors95)} Singleton=${"%.2f".format(bmAll.singletonPct)}% RunLen50=${"%.2f".format(bmAll.runLen50)} Chg/100=${"%.2f".format(bmAll.chgPer100)} (skin: C95=${"%.2f".format(bmSkin.colors95)}, S=${"%.2f".format(bmSkin.singletonPct)}%, RL50=${"%.2f".format(bmSkin.runLen50)})")
        }
        return bmp to used
    }

    /** Рендер одного S с уже готовой (замороженной) палитрой — без RAQ (быстро). */
    /** Рендер одного S с уже готовой (замороженной) палитрой — без RAQ (быстро). */
    private fun renderWithFrozenPaletteAtS(
        source: Bitmap,
        threadPalette: List<Swatch>,
        paletteArgb: IntArray,
        paletteLab: FloatArray,
        preferMode: PhotoConfig.DitherMode,
        targetWidthSt: Int
    ): Pair<Bitmap, List<Swatch>> {
        val srcS = scaleToWidth(source, targetWidthSt)
        val rasterS = Raster(srcS.width, srcS.height, bitmapToIntArray(srcS))
        val n = rasterS.argb.size

        // 0) Базовые карты и маски
        // «Сильные» края (по модулю градиента): берём как !flatMask(τ=0.06)
        // реальные «сильные» края и узкая лента вокруг них для FS
        val strongEdge = EdgeMeter.strongEdgeMask(srcS, 0.12f)
        val edgeRibbon = dilateMask(strongEdge, srcS.width, srcS.height, r = PhotoConfig.B5.FS_RIBBON_RADIUS)
        // Узкая «лента» вокруг сильных краёв (FS только здесь)

        // [B5-D] 0a) Тон/гамма: мягкая S-кривая по L* перед квантованием
        if (TONE_CURVE_ENABLED) {
            applyToneCurveL(rasterS, srcS.width, srcS.height, TONE_GAMMA_IN, TONE_HIGHLIGHT_COMP)
            HQLog.pre("tone: gamma=$TONE_GAMMA_IN, highlights=$TONE_HIGHLIGHT_COMP")
        }

        // 0a) Edge-aware предсглаживание по L* (вне сильных краёв), затем первичный квант (NO-DITHER)
        if (PRE_EDGE_AWARE_BLUR) {
            applyEdgeAwareBlurL(rasterS, srcS.width, srcS.height, strongEdge, PRE_BLUR_SIGMA_PX)
            HQLog.pre("pre.blur: edge-aware L σ=$PRE_BLUR_SIGMA_PX (off strong edges)")
        }
        // Хрома‑компрессия в светах (до квантования)
        compressChromaInHighlights(paletteLab)
        val preNoDither = quantNearest(rasterS, paletteLab, paletteArgb) // OFF вариант
        // Majority-clean 3×3 до дизера, вне сильных/тонких структур
        if (MAJORITY_CLEAN_ENABLED) {
            val thinMask = computeThinMask(strongEdge, srcS.width, srcS.height, THIN_COMPONENT_MAX)
            val guard = BooleanArray(n) { i -> strongEdge[i] || thinMask[i] }
            val changed = majority3x3(preNoDither.argb, srcS.width, srcS.height, guard)
            if (changed > 0) HQLog.s("majority.clean: changed=$changed (guard=edges+thin<$THIN_COMPONENT_MAX)")
            else HQLog.s("majority.clean: skipped")
                    }
        val imgNoDither = intArrayToBitmap(preNoDither.argb, srcS.width, srcS.height)
        val bandMask = BandingMeter.bandingMask(srcS, imgNoDither, win = 7) // где усиливать ORDERED
        // Skin-маска (быстрая YCbCr-эвристика)
        val skinMaskLocal = skinMaskYCbCr(srcS)

        // Быстрая «яркость» (Y) для детекта тёмного фона (порог ~0.10)
        val darkMask = BooleanArray(n)
        run {
            val a = rasterS.argb
            var i = 0
            while (i < n) {
                val p = a[i]
                val r = (p shr 16) and 0xFF
                val g = (p shr 8) and 0xFF
                val b = p and 0xFF
                // Y' в [0,1]
                val y = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
                darkMask[i] = y < 0.10f
                i++
            }
        }

        // 1) Гибридный дизеринг
        // ORDERED 8×8 по тайлам: базовая и усиленная амплитуда; усиливаем только в окнах риска бэндинга
        val AMP_BASE   = PhotoConfig.B5.ORDERED_AMP_BASE
        val AMP_STRONG = PhotoConfig.B5.ORDERED_AMP_STRONG
        val tile = 8
        val wTiles = (srcS.width + tile - 1) / tile
        val hTiles = (srcS.height + tile - 1) / tile
        val ampTiles = FloatArray(wTiles * hTiles) { AMP_BASE }
        // Усиливаем амплитуду в тех тайлах, где доля banding-пикселей ≥ 25%
        run {
            var ty = 0
            while (ty < hTiles) {
                val y0 = ty * tile; val y1 = kotlin.math.min(srcS.height, y0 + tile)
                var tx = 0
                while (tx < wTiles) {
                    val x0 = tx * tile; val x1 = kotlin.math.min(srcS.width, x0 + tile)
                    var cnt = 0; var flagged = 0
                    var y = y0
                    while (y < y1) {
                        val row = y * srcS.width
                        var x = x0
                        while (x < x1) {
                            val ii = row + x
                            if (bandMask[ii] && !strongEdge[ii]) flagged++
                            cnt++; x++
                        }
                        y++
                    }
                    val share = if (cnt > 0) flagged.toFloat() / cnt else 0f
                    if (share >= 0.25f) ampTiles[ty * wTiles + tx] = AMP_STRONG
                    tx++
                }
                ty++
            }
        }

        // Где применяем ORDERED: skin ∪ flat, вне ленты и не слишком тёмно
        val darkMask = BooleanArray(n) { lStarPlane[it] < PhotoConfig.B5.DARK_L_T }
        val flatMask = EdgeMeter.flatMask(srcS, PhotoConfig.B5.FLAT_GRAD_T)
        val orderedMask = BooleanArray(n) { i ->
            (skinMaskLocal[i] || flatMask[i]) && !edgeRibbon[i] && !darkMask[i]
        }
        val orderedAll = if (PhotoConfig.B5.USE_BLUE_NOISE) {
            com.appforcross.core.dither.ditherOrdered8x8TilesBN(
                rasterS, paletteLab, paletteArgb, Metric.OKLAB,
                ampTiles, wTiles, hTiles, tile, orderedMask, blueNoise = true
            )
        } else {
                com.appforcross.core.dither.ditherOrdered8x8Tiles(
                rasterS, paletteLab, paletteArgb, Metric.OKLAB,
                ampTiles, wTiles, hTiles, tile, orderedMask
            )
        }

        // Один проход ORDERED с тайловой амплитудой
        val orderedAll = ditherOrdered8x8Tiles(
            input = rasterS,
            allowedLab = paletteLab,
            allowedArgb = paletteArgb,
            metric = Metric.OKLAB,
            ampTiles = ampTiles,
            wTiles = wTiles,
            hTiles = hTiles,
            tile = tile,
            mask = null // маску можно передать при необходимости ограничить область ORDERED
        )

        // FS‑анизотропный по краю: вдоль 1.0, поперёк 0.40
        val fs = when (preferMode) {
            PhotoConfig.DitherMode.NONE -> preNoDither
            else -> {
                val (tx, ty) = EdgeMeter.tangentField(srcS)
                ditherFsAniso(rasterS, paletteLab, paletteArgb, Metric.OKLAB, tx, ty, 1.00f, 0.40f)
            }
        }

        // 2) Смешивание по правилам + счётчики для логов
        val out = IntArray(n)
        var cntFS = 0; var cntORDb = 0; var cntORDs = 0; var cntOFF = 0
        var skinFS = 0; var skinORD = 0; var skinOFF = 0
        var bgFS = 0; var bgORD = 0; var bgOFF = 0
        var i = 0
        while (i < n) {
            val useFS  = strongEdge[i]                 // по сильным краям — FS
            val useOFF = !useFS && darkMask[i]         // очень тёмный фон — OFF
            val useORD = !useFS && !useOFF             // остальное — ORDERED (тайловая амплитуда)

            val isSkin = skinMaskLocal[i] && !strongEdge[i]
            val px = when {
                useFS -> {
                    if (isSkin) skinFS++ else bgFS++
                    cntFS++; fs.argb[i]
                }
                useOFF -> {
                    if (isSkin) skinOFF++ else bgOFF++
                    cntOFF++; preNoDither.argb[i]
                }
                else -> { // ORDERED
                    if (isSkin) skinORD++ else bgORD++
                    if (bandMask[i]) cntORDs++ else cntORDb++ // для логов base/strong
                    orderedAll.argb[i]
                }
            }
            out[i] = px
            i++
        }

        // 3) Логи дезеринга
        val tot = n.coerceAtLeast(1)
        val ordTot = (cntORDb + cntORDs).coerceAtLeast(1)
        val fsPct = 100f * cntFS / tot
        val ordPct = 100f * (cntORDb + cntORDs) / tot
        val offPct = 100f * cntOFF / tot
        // orderedAmp: mean и p95 из дискретного распределения {base,strong}
        val meanAmp = (AMP_BASE * cntORDb + AMP_STRONG * cntORDs) / ordTot
        val p95Amp = run {
            val thr = (ordTot * 0.95f).toInt()
            if (cntORDs >= thr) AMP_STRONG else AMP_BASE
        }
        HQLog.s(
            "dither.stats: FS=${"%.1f".format(fsPct)}%, ORD=${"%.1f".format(ordPct)}%, OFF=${"%.1f".format(offPct)}% " +
                    "(skin: FS=$skinFS, ORD=$skinORD, OFF=$skinOFF; bg: FS=$bgFS, ORD=$bgORD, OFF=$bgOFF)"
        )

        // Пост-чистка одиночек ПОСЛЕ смешивания: scope = skin ∪ flat, guard = strongEdge
        if (PhotoConfig.B5.CLEAN_SINGLETONS && !postCleanApplied) {
            val flatMask = EdgeMeter.flatMask(srcS, PhotoConfig.B5.FLAT_GRAD_T)
            val scope = BooleanArray(n) { i -> skinMaskLocal[i] || flatMask[i] }
            cleanSingletonsMajority1(
                out, srcS.width, srcS.height,
                protect = strongEdge,
                protect = edgeRibbon,
                scopeFlat = scope
            )
            HQLog.s("post.clean: majority1 applied (skin∪flat)")
            postCleanApplied = true
        }
        HQLog.s(
            "orderedAmp: mean=${"%.3f".format(meanAmp)}, p95=${"%.3f".format(p95Amp)}; fsAniso: along=1.00, across=0.40"
        )
        // Пост-чистка одиночек ПОСЛЕ смешивания (skin ∪ flat, вне сильных краёв)
        if (PhotoConfig.B5.CLEAN_SINGLETONS) {
            val flatMask = EdgeMeter.flatMask(srcS, PhotoConfig.B5.FLAT_GRAD_T)
            val scope = BooleanArray(n) { i -> skinMaskLocal[i] || flatMask[i] }
            cleanSingletonsMajority1(
                out, srcS.width, srcS.height,
                protect = strongEdge,
                scopeFlat = scope
            )
            HQLog.s("post.clean: majority1 applied (skin∪flat)")
        }
        val bmp = intArrayToBitmap(out, srcS.width, srcS.height)
        val used = collectUsedSwatches(threadPalette, out)
        return bmp to used
    }

    /** Быстрый skin‑детектор в YCbCr (8‑битовая эвристика; для портретов достаточно). */
    private fun skinMaskYCbCr(img: Bitmap): BooleanArray {
        val w = img.width; val h = img.height; val n = w * h
        val out = BooleanArray(n)
        val buf = IntArray(n); img.getPixels(buf, 0, w, 0, 0, w, h)
        var i = 0
        while (i < n) {
            val p = buf[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            // ITU‑R BT.601 approx
            val y  =  0.299f * r + 0.587f * g + 0.114f * b
            val cb = -0.1687f * r - 0.3313f * g + 0.5f   * b + 128f
            val cr =  0.5f   * r - 0.4187f * g - 0.0813f * b + 128f
            // коридор для кожи + отсечение по яркости
            out[i] = (y > 70f && cb in 85f..135f && cr in 133f..180f)
            i++
        }
        return out
    }

    /** Список реально использованных ниток (пересечение финальных пикселей с активной палитрой). */
    private fun collectUsedSwatches(palette: List<Swatch>, outArgb: IntArray): List<Swatch> {
        val used = HashSet<Int>(64)
        for (px in outArgb) used.add(px)
        val res = ArrayList<Swatch>(used.size)
        // Сохраним порядок «по частоте» для стабильности UI
        val freq = HashMap<Int, Int>(used.size)
        for (px in outArgb) if (px in used) freq[px] = (freq[px] ?: 0) + 1
        val sorted = freq.entries.sortedByDescending { it.value }.map { it.key }
        val byArgb = palette.groupBy { it.argb }
        for (argb in sorted) {
            val sws = byArgb[argb]
            if (!sws.isNullOrEmpty()) res.add(sws.first())
        }
        return res
    }

    /** Удаление одиночных пикселей: если у пикселя нет ни одного 8-соседа того же цвета — красим в цвет большинства соседей. */
    private fun cleanSingletonsMajority1(
        argb: IntArray,
        w: Int, h: Int,
        protect: BooleanArray?,
        scopeFlat: BooleanArray?
    ) {
        val n = argb.size
        val out = argb.clone()
        fun idx(x: Int, y: Int) = y * w + x
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val p = idx(x, y)
                if ((protect != null && protect[p]) || (scopeFlat != null && !scopeFlat[p])) continue
                val c = argb[p]
                var same = 0
                val hist = HashMap<Int, Int>(8)
                var yy = -1
                while (yy <= 1) {
                    var xx = -1
                    while (xx <= 1) {
                        if (xx != 0 || yy != 0) {
                            val q = argb[idx(x + xx, y + yy)]
                            if (q == c) same++
                            hist[q] = (hist[q] ?: 0) + 1
                        }
                        xx++
                    }
                    yy++
                }
                if (same == 0) {
                    // перекрашиваем в цвет большинства соседей
                    var bestC = c; var bestN = 0
                    for ((cc, nn) in hist) if (nn > bestN) { bestN = nn; bestC = cc }
                    out[p] = bestC
                }
            }
        }
        System.arraycopy(out, 0, argb, 0, n)
    }

    /** Яркость 0..1 для bitmap (локально, чтобы не тянуть из EdgeMeter). */
    private fun luma01(src: Bitmap): FloatArray {
        val w = src.width; val h = src.height; val n = w*h
        val px = IntArray(n); src.getPixels(px,0,w,0,0,w,h)
        val out = FloatArray(n)
        var i=0; while(i<n){
            val p=px[i]; val r=((p ushr 16) and 0xFF)/255f; val g=((p ushr 8) and 0xFF)/255f; val b=(p and 0xFF)/255f
            out[i]=0.2126f*r+0.7152f*g+0.0722f*b; i++
        }
        return out
    }

    // [B5-D] Тон/гамма (S-кривая по L*): gamma_in и мягкая компрессия хайлайтов
    private fun applyToneCurveL(
        work: Raster, w: Int, h: Int,
        gammaIn: Float, hiComp: Float
    ) {
        val px = work.argb
        val invG = 1f / gammaIn.coerceAtLeast(1e-3f)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            val a = (c ushr 24) and 0xFF
            var r = (c ushr 16) and 0xFF
            var g = (c ushr 8) and 0xFF
            var b = c and 0xFF
            // простая гамма на каналах (приближение L*)
            fun gpow(v: Int): Int {
                val x = (v / 255f).coerceIn(0f, 1f)
                val y = x.toDouble().pow(invG.toDouble()).toFloat()
                return (y * 255f + 0.5f).toInt().coerceIn(0, 255)
            }
            r = gpow(r); g = gpow(g); b = gpow(b)
            // мягкая компрессия верхних 12% (hiComp)
            val l = (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255f
            if (l > 1f - hiComp) {
                val t = (l - (1f - hiComp)) / hiComp
                val k = 1f - 0.25f * t // слегка приглушаем
                r = (r * k).toInt().coerceIn(0, 255)
                g = (g * k).toInt().coerceIn(0, 255)
                b = (b * k).toInt().coerceIn(0, 255)
            }
            px[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
            i++
        }
    }

    private fun compressChromaInHighlights(lab: FloatArray) {
        var i = 0
        while (i < lab.size) {
            val L = lab[i]
            val t = ((L - 0.75f) / 0.20f).coerceIn(0f, 1f) // 0.75..0.95
            val k = 1f - 0.6f * t                           // до −60% хромы
            lab[i + 1] *= k
            lab[i + 2] *= k
            i += 3
        }
    }

    // [B5-D] Серые якоря по L* (равномерные уровни) для защиты нейтралей
    private fun neutralAnchorsARGB(n: Int): IntArray {
        val k = n.coerceIn(3, 7)
        val out = IntArray(k)
        for (i in 0 until k) {
            val t = (i + 1).toFloat() / (k + 1)
            val v = (t * 255f + 0.5f).toInt().coerceIn(0, 255)
            out[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }
        return out
    }

    // [B5-C] Блок-метрики 10×10 (глобально и по маске)
    private data class BlockMetrics(
        val colors95: Float,
        val singletonPct: Float,
        val runLen50: Float,
        val chgPer100: Float
    )
    private fun computeBlockMetrics(
        argb: IntArray, w: Int, h: Int, mask: BooleanArray?
    ): BlockMetrics {
        val bw = 10; val bh = 10
        val blocks = ArrayList<Int>()
        var singletons = 0; var total = 0
        // Colors per block
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val colors = HashSet<Int>()
                var yy = y
                val yMax = kotlin.math.min(y + bh, h)
                val xMax = kotlin.math.min(x + bw, w)
                while (yy < yMax) {
                    var xx = x
                    val off = yy * w
                    while (xx < xMax) {
                        val i = off + xx
                        if (mask == null || mask[i]) colors.add(argb[i])
                        xx++
                    }
                    yy++
                }
                blocks.add(colors.size)
                x += bw
            }
            y += bh
        }
        // 95-й перцентиль
        val sorted = blocks.sorted()
        val c95 = if (sorted.isEmpty()) 0f else sorted[(sorted.size * 95) / 100].toFloat()
        // Singleton
        for (py in 1 until h - 1) {
            val off = py * w
            for (px in 1 until w - 1) {
                val i = off + px
                if (mask != null && !mask[i]) continue
                val c = argb[i]
                var same = 0
                for (dy in -1..1) for (dx in -1..1) if (dx != 0 || dy != 0) {
                    if (argb[i + dy * w + dx] == c) same++
                }
                if (same == 0) singletons++
                total++
            }
        }
        val singPct = if (total == 0) 0f else (100f * singletons / total)
            // RunLen (по «змейке»): медиана длины прогонов + смены цвета на 100 шагов
        val runs = ArrayList<Int>(w * h / 4)
        var changes = 0; var steps = 0
        for (row in 0 until h) {
            val dir = if (row % 2 == 0) 1 else -1
            var x0 = if (dir > 0) 0 else w - 1
            var prev = argb[row * w + x0]
            var len = 1
            var x = x0 + dir
            while (x in 0 until w) {
                val i = row * w + x
                if (mask != null && !mask[i]) { x += dir; continue }
                val c = argb[i]
                steps++
                if (c == prev) {
                    len++
                } else {
                        runs.add(len); len = 1; prev = c; changes++
                    }
                x += dir
            }
            runs.add(len)
        }
        val rSorted = runs.sorted()
        val run50 = if (rSorted.isEmpty()) 0f else rSorted[rSorted.size / 2].toFloat()
        val chg100 = if (steps == 0) 0f else 100f * changes / steps
        return BlockMetrics(c95, singPct, run50, chg100)
    }

    /** Дозаполнение палитры, если после merge слишком мало цветов: берём пики ошибки ΔE. */
    private fun refillByError(
        srcS: Bitmap,
        noDither: Raster,
        threads: List<Swatch>,
                need: Int
    ): Pair<IntArray, FloatArray> {
        if (need <= 0) return IntArray(0) to FloatArray(0)
        val w = srcS.width; val h = srcS.height; val n = w*h
        val a = IntArray(n); srcS.getPixels(a,0,w,0,0,w,h)
        val b = noDither.argb
        // ΔE как критерий «куда добавить»
        val labA = argbToOkLab(a); val labB = argbToOkLab(b)
        val err = FloatArray(n)
        var i=0; while(i<n){ val j=i*3; val dl=labA[j]-labB[j]; val da=labA[j+1]-labB[j+1]; val db=labA[j+2]-labB[j+2]; err[i]=dl*dl+da*da+db*db; i++ }
        // берём топ-пики ошибок
        val idx = (0 until n).sortedByDescending { err[it] }.take(need * 120)
        val sub = FloatArray(idx.size*3)
        var k=0; for(p in idx){ val j=p*3; sub[k++]=labA[j]; sub[k++]=labA[j+1]; sub[k++]=labA[j+2] }
        val centers = kmeansLab(sub, need, iters=4, seed=777)
        val snapped = snapCentersToThreads(centers, threads)
        val snappedLab = argbToOkLab(snapped)
        return snapped to snappedLab
    }

    // ------------------------------ утилиты: масштаб/квант/гаусс ------------------------------

    private fun scaleToWidth(src: Bitmap, targetW: Int): Bitmap {
        if (src.width == targetW) return src
        val ratio = targetW.toFloat() / src.width
        val targetH = (src.height * ratio).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, targetW, targetH, true)
    }

    /** Квант «ближайшего соседа» без распространения ошибок (для режима NONE). */
    private fun quantNearest(input: Raster, allowedLab: FloatArray, allowedArgb: IntArray): Raster {
        val lab = argbToOkLab(input.argb)
        val out = IntArray(input.argb.size)
        var p = 0
        fun nearestIndex(L: Float, A: Float, B: Float): Int {
            var best = 0
            var bestD = Float.POSITIVE_INFINITY
            var i = 0
            while (i < allowedLab.size) {
                val dl = allowedLab[i] - L
                val da = allowedLab[i + 1] - A
                val db = allowedLab[i + 2] - B
                val d = dl*dl + da*da + db*db
                if (d < bestD) { bestD = d; best = i / 3 }
                i += 3
            }
            return best
        }
        while (p < out.size) {
            val i3 = p * 3
            val ai = nearestIndex(lab[i3], lab[i3 + 1], lab[i3 + 2])
            out[p] = allowedArgb[ai]
            p++
        }
        return Raster(input.width, input.height, out)
    }

    /** Быстрый 3x3 «гаусс‑лайт» (1,2,1; 2,4,2; 1,2,1)/16 по каждому каналу RGB. */
    private fun gauss3x3(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height
        val inPx = IntArray(w * h)
        src.getPixels(inPx, 0, w, 0, 0, w, h)
        fun at(x: Int, y: Int): Int = inPx[y * w + x]
        val outPx = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                var r = 0; var g = 0; var b = 0
                fun acc(xx: Int, yy: Int, wgt: Int) {
                    val cx = min(max(xx, 0), w - 1)
                    val cy = min(max(yy, 0), h - 1)
                    val c = at(cx, cy)
                    r += ((c shr 16) and 0xFF) * wgt
                    g += ((c shr 8) and 0xFF) * wgt
                    b += (c and 0xFF) * wgt
                }
                acc(x - 1, y - 1, 1); acc(x, y - 1, 2); acc(x + 1, y - 1, 1)
                acc(x - 1, y, 2);     acc(x, y, 4);     acc(x + 1, y, 2)
                acc(x - 1, y + 1, 1); acc(x, y + 1, 2); acc(x + 1, y + 1, 1)
                val rr = (r / 16).coerceIn(0, 255)
                val gg = (g / 16).coerceIn(0, 255)
                val bb = (b / 16).coerceIn(0, 255)
                outPx[y * w + x] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
            }
        }
        val out = Bitmap.createBitmap(w, h, ARGB_8888)
        out.setPixels(outPx, 0, w, 0, 0, w, h)
        return out
    }

    private fun bitmapToIntArray(bm: Bitmap): IntArray {
        val w = bm.width; val h = bm.height
        val px = IntArray(w * h)
        bm.getPixels(px, 0, w, 0, 0, w, h)
        return px
    }
    private fun intArrayToBitmap(px: IntArray, w: Int, h: Int): Bitmap {
        val out = Bitmap.createBitmap(w, h, ARGB_8888)
        out.setPixels(px, 0, w, 0, 0, w, h)
        return out
    }

    // ── Helpers, которых не хватало: edge-aware blur, majority 3×3 и thin-mask ──
    private fun applyEdgeAwareBlurL(
        work: Raster, w: Int, h: Int, edgeMask: BooleanArray, sigma: Float
    ) {
        val src = work.argb
        val out = src.clone()
        fun luma(c: Int): Int {
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            return (0.2126 * r + 0.7152 * g + 0.0722 * b).toInt()
        }
        for (y in 1 until h - 1) {
            val yw = y * w
            for (x in 1 until w - 1) {
                val i = yw + x
                if (edgeMask[i]) continue
                var sR = 0; var sG = 0; var sB = 0; var cnt = 0
                for (dy in -1..1) for (dx in -1..1) {
                    val j = i + dy * w + dx
                    if (edgeMask[j]) continue
                    val c = src[j]
                    sR += (c ushr 16) and 0xFF
                    sG += (c ushr 8) and 0xFF
                    sB += c and 0xFF
                    cnt++
                }
                if (cnt > 0) {
                    val r = (sR / cnt).coerceIn(0, 255)
                    val g = (sG / cnt).coerceIn(0, 255)
                    val b = (sB / cnt).coerceIn(0, 255)
                    out[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        System.arraycopy(out, 0, work.argb, 0, out.size)
    }

    private fun majority3x3(ids: IntArray, w: Int, h: Int, guard: BooleanArray): Int {
        var changed = 0
        val out = ids.clone()
        for (y in 1 until h - 1) {
            val yw = y * w
            for (x in 1 until w - 1) {
                val i = yw + x
                if (guard[i]) continue
                val center = ids[i]
                val hist = HashMap<Int, Int>(8)
                for (dy in -1..1) for (dx in -1..1) if (dx != 0 || dy != 0) {
                    val v = ids[i + dy * w + dx]
                    hist[v] = (hist[v] ?: 0) + 1
                }
                var best = center; var cntBest = 0
                for ((k, v) in hist) if (v > cntBest) { cntBest = v; best = k }
                if (best != center) { out[i] = best; changed++ }
            }
        }
        if (changed > 0) System.arraycopy(out, 0, ids, 0, ids.size)
        return changed
    }

    private fun computeThinMask(edge: BooleanArray, w: Int, h: Int, maxArea: Int): BooleanArray {
        // упрощённая эвристика: "узкие" края = мало соседей-краёв
        val out = BooleanArray(edge.size)
        fun neigh(i: Int): Int {
            var c = 0
            val x = i % w; val y = i / w
            for (dy in -1..1) for (dx in -1..1) {
                if (dx == 0 && dy == 0) continue
                val nx = x + dx; val ny = y + dy
                if (nx in 0 until w && ny in 0 until h) {
                    val j = ny * w + nx
                    if (edge[j]) c++
                }
            }
            return c
        }
        for (i in edge.indices) if (edge[i] && neigh(i) <= 2) out[i] = true
        return out
    }

}

