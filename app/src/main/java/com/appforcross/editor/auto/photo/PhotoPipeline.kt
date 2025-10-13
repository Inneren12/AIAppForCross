package com.appforcross.editor.auto.photo

import android.graphics.Bitmap
import com.appforcross.core.palette.Swatch
import java.util.Arrays
import java.util.Locale
import kotlin.math.*
import java.util.Random
import kotlin.collections.get
import kotlin.compareTo
import kotlin.ranges.rangeTo
import kotlin.text.toFloat

/**
 * Фото-ветка (Спринт 1): минимально рабочая версия.
 *  - лёгкий descreen/denoise/WB,
 *  - Kneedle-подбор K (ΔE95 + BandingIndex),
 *  - взвешенный K-means++ (edge-aware), mini-batch K-means,
 *  - ordered/blue-noise dither (edge-aware), анти-конфетти,
 *  - перебор S и PASS по фото-порогам, Score(S)≥0.75 -> минимальный S.
 */
object PhotoPipeline {

    data class Result(
        val image: Bitmap,         // итоговая сетка S (клетка=пиксель) с дитором
        val gridWidth: Int,
        val gridHeight: Int,
        val gridSizeS: Int,
        val usedSwatches: List<Swatch>,
        val metrics: Metrics
    )

    data class Metrics(
        val dE95: Float,
        val edgeSSIM: Float,
        val hfRetain: Float,
        val banding: Float,
        val confetti: Float,
        val score: Float,
        val passed: Boolean,
        val reasons: List<String>
    )

    // ────────────────────────── Публичный API ──────────────────────────

    fun run(
        source: Bitmap,
        threadPalette: List<Swatch>,
        preset: PhotoConfig.Params = PhotoConfig.Landscape,
        sizes: IntArray = PhotoConfig.defaultSizes,
        enableDescreen: Boolean = false
    ): Result {
        require(source.width > 0 && source.height > 0) { "Empty bitmap" }
        require(threadPalette.isNotEmpty()) { "Thread palette is empty" }

        // 1) Подготовка
        val src0 = scaleToMaxSide(source, 1024)
        var work = if (enableDescreen) descreenLite(src0) else src0
        work = denoiseLite(work)
        work = grayWorldWB(work)

        // 2) Выбор K (Kneedle v2: ROI-вес + DitherEnergy на сабсемпле)
        val edgeMask = sobelMask(work)
        val k = chooseK(work, preset, edgeMask, threadPalette)

        // 3) Квантование (взвешенный K-means++ + mini-batch K-means + регуляризаторы)
        val centers = kmeansWork(work, k, preset, edgeMask, threads = threadPalette)

        // 4) Снап в нитки (+merge chroma twins), shortlists и диторинг
        val mapped0 = centersToThreads(centers, threadPalette)
        val mappedPalette = mergeChromaTwins(mapped0, thr = 0.03f)
        val short = if (preset.shortlistM > 0)
            buildShortlists(work, mappedPalette, preset.shortlistM, block = 10) else null
        val dithered = orderedDitherEdgeAware(
            work, mappedPalette, edgeMask,
            orderedBias = preset.orderedBias,
            shortlists = short
        )

        // Анти-конфетти
        val clean = if (preset.antiConfetti) cullIsolatesToMajority(dithered) else dithered

        // 5) Перебор S и метрики
        var best: Result? = null
        var bestScore = -1f
        for (S in sizes) {
            val H = max(1, (work.height * (S.toFloat() / work.width)).roundToInt())
            val srcS = Bitmap.createScaledBitmap(work, S, H, true)              // «эталон» для ΔE/Edge/HF
            val gridS = Bitmap.createScaledBitmap(clean, S, H, false)           // результат (дитор уже в пикселях)

            val m = measurePhotoMetrics(srcS, gridS, mappedPalette, preset)
            val (passed, reasons) = passPhoto(m, preset)
            val used = collectUsedSwatches(gridS, mappedPalette)

            val res = Result(
                gridS, S, H, S, used,
                m.copy(passed = passed, reasons = reasons)
            )
            if (passed) {
                best = res; break // минимальный S с PASS
            } else if (m.score > bestScore) {
                best = res; bestScore = m.score
            }
        }
        return requireNotNull(best) { "PhotoPipeline: no candidate produced" }
    }

    // ──────────────────────── Kneedle: выбор K ─────────────────────────

    private fun chooseK(
        bmp: Bitmap,
        p: PhotoConfig.Params,
        edgeMask: BooleanArray,
        threads: List<Swatch>
    ): Int {
        val ks = (p.kMin..p.kMax step p.kStep).toList()
        val errs = FloatArray(ks.size)
        val bands = FloatArray(ks.size)
        val energy = FloatArray(ks.size)
        // ROI-веса (0..1) по карте градиента
        val roi = edgeMagnitude(bmp)
        // компактная копия для быстрого прогона дитера
        val probe = scaleToMaxSide(bmp, 256)
        val edgeProbe = sobelMask(probe)

        for ((idx, k) in ks.withIndex()) {
            val centers = kmeansWork(bmp, k, p, edgeMask, dryRun = true)
            val (de95, band) = errorForKWeighted(bmp, centers, p, roi)
            errs[idx] = de95; bands[idx] = band
            val pal = centersToThreads(centers, threads)
            energy[idx] = ditherEnergyProbe(probe, pal, edgeProbe)
        }
        // Нормируем и считаем комб. ошибку (ΔE95^ROI + Banding + DitherEnergy)
        val eMax = max(errs.maxOrNull() ?: 1f, 1f)
        val bMax = max(bands.maxOrNull() ?: 1f, 1f)
        val dMax = max(energy.maxOrNull() ?: 1f, 1f)
        val comb = FloatArray(ks.size) { i ->
            0.5f * (errs[i] / eMax) +
                    0.3f * (bands[i] / bMax) +
                    0.2f * (energy[i] / dMax)
        }

        // Kneedle: ищем «локоть» — максимальное отклонение от прямой между концами
        val x0 = ks.first().toFloat(); val x1 = ks.last().toFloat()
        val y0 = comb.first(); val y1 = comb.last()
        var bestI = 0; var bestDev = -1f
        for (i in comb.indices) {
            val xi = ks[i].toFloat()
            val yLine = y0 + (y1 - y0) * (xi - x0) / (x1 - x0)
            val dev = (yLine - comb[i]).toFloat()
            if (dev > bestDev && dev >= p.kneedleEps) { bestDev = dev; bestI = i }
        }
        return ks[bestI]
    }

    private fun errorForKWeighted(
        bmp: Bitmap,
        centers: FloatArray,
        p: PhotoConfig.Params,
        roi: FloatArray
    ): Pair<Float, Float> {
        val (lab, w, h) = toLab(bmp)
        val n = w * h
        val d = FloatArray(n)
        var i = 0
        while (i < n) {
            val L = lab[i*3]; val A = lab[i*3+1]; val B = lab[i*3+2]
            var best = Float.POSITIVE_INFINITY; var j = 0
            while (j < centers.size) {
                val dL = L - centers[j]; val dA = A - centers[j+1]; val dB = B - centers[j+2]
                val dist = sqrt(p.wL * (dL * dL) + dA * dA + dB * dB)
                if (dist < best) best = dist
                j += 3
            }
            val wR = 1f + p.rhoRoi * roi[i]
            d[i] = (best * wR)
            i++
        }
        val de95 = percentile(d, 95f)
        val band = bandingIndexFromLQuant(lab, centers, w, h)
        return de95 to band
    }

    // ────────────────── K-means++ (edge-weighted) ─────────────────────

    private fun kmeansWork(
        bmp: Bitmap,
        K: Int,
        p: PhotoConfig.Params,
        edgeMask: BooleanArray,
        dryRun: Boolean = false,
        threads: List<Swatch>? = null
    ): FloatArray {
        val (lab, w, h) = toLab(bmp)
        val n = w * h

        // Веса: 1 + rhoEdge * 1[edge]
        val weights = FloatArray(n) { i -> if (edgeMask[i]) (1f + p.rhoEdge) else 1f }

        // Сидирование: weighted K-means++
        val centers = weightedKppSeeds(lab, weights, K, p)

        if (dryRun) return centers

        // Mini-batch KMeans (простая реализация) + регуляризаторы (v1)
        val rnd = Random(42)
        val iters = 12
        val batch = min(4000, n)
        val sum = FloatArray(K*3)
        val cnt = FloatArray(K)
        repeat(iters) {
            Arrays.fill(sum, 0f); Arrays.fill(cnt, 0f)
            var t = 0
            while (t < batch) {
                val i = rnd.nextInt(n)
                val L = lab[i*3]; val A = lab[i*3+1]; val B = lab[i*3+2]
                var best = 0; var bestD = Float.POSITIVE_INFINITY; var j = 0
                while (j < centers.size) {
                    val dL=L-centers[j]; val dA=A-centers[j+1]; val dB=B-centers[j+2]
                    val dist = p.wL*(dL*dL) + dA*dA + dB*dB
                    if (dist < bestD) { bestD = dist; best = j/3 }
                    j += 3
                }
                val wgt = weights[i]
                sum[best*3  ] += L*wgt
                sum[best*3+1] += A*wgt
                sum[best*3+2] += B*wgt
                cnt[best] += wgt
                t++
            }
            for (k in 0 until K) {
                val c = max(1e-6f, cnt[k])
                centers[k*3  ] = sum[k*3  ] / c
                centers[k*3+1] = sum[k*3+1] / c
                centers[k*3+2] = sum[k*3+2] / c
            }
            // --- Regularizers: L‑ladder и Confetti‑proxy ---
            if (p.lambdaBand > 0f) enforceLLadder(centers, cnt, p.lambdaBand)
            if (p.lambdaConfetti > 0f) confettiProxyNudge(lab, w, h, centers, p.lambdaConfetti)
            // --- Thread look‑ahead (проекция с cap) ---
            if (threads != null && p.threadSnapCap > 0f) {
                projectTowardsThreads(centers, threads, cap = p.threadSnapCap)
            }
        }
        return centers
    }

    private fun weightedKppSeeds(lab: FloatArray, w: FloatArray, K: Int, p: PhotoConfig.Params): FloatArray {
        val n = w.size
        val rnd = Random(13)
        val centers = FloatArray(K*3)

        // Первый seed — пропорционально весу
        var sumW = 0.0
        for (i in 0 until n) sumW += w[i]
        var r = rnd.nextDouble() * sumW
        var idx = 0; var acc = 0.0
        while (idx < n) { acc += w[idx]; if (acc >= r) break; idx++ }
        centers[0] = lab[idx*3]; centers[1] = lab[idx*3+1]; centers[2] = lab[idx*3+2]

        val dist2 = FloatArray(n) { Float.POSITIVE_INFINITY }
        var placed = 1
        while (placed < K) {
            // Обновляем D(x)^2 до ближайшего центра
            var i = 0
            while (i < n) {
                val dL=lab[i*3]-centers[(placed-1)*3]
                val dA=lab[i*3+1]-centers[(placed-1)*3+1]
                val dB=lab[i*3+2]-centers[(placed-1)*3+2]
                val d = p.wL*(dL*dL) + dA*dA + dB*dB
                if (d < dist2[i]) dist2[i] = d
                i++
            }
            // Выбор следующего семени: P(i) ∝ D(i)^2 * w(i)
            var sumP = 0.0
            i = 0; while (i < n) { sumP += dist2[i]*w[i]; i++ }
            r = rnd.nextDouble() * max(1e-9, sumP)
            acc = 0.0; idx = 0
            while (idx < n) { acc += dist2[idx]*w[idx]; if (acc >= r) break; idx++ }
            centers[placed*3  ] = lab[idx*3  ]
            centers[placed*3+1] = lab[idx*3+1]
            centers[placed*3+2] = lab[idx*3+2]
            placed++
        }
        return centers
    }

    // ──────────────────────── Dither & mapping ─────────────────────────

    private fun centersToThreads(centers: FloatArray, threads: List<Swatch>): IntArray {
        val out = IntArray(centers.size/3)
        // Предрасчёт ниток в OKLab
        val tL = FloatArray(threads.size); val tA = FloatArray(threads.size); val tB = FloatArray(threads.size)
        for (i in threads.indices) {
            val c = threads[i].argb
            val lab = rgbToOkLab((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            tL[i]=lab[0]; tA[i]=lab[1]; tB[i]=lab[2]
        }
        for (k in 0 until out.size) {
            val L=centers[k*3]; val A=centers[k*3+1]; val B=centers[k*3+2]
            var best = 0; var bestD = Float.POSITIVE_INFINITY
            var i=0
            while (i < threads.size) {
                val dL=L-tL[i]; val dA=A-tA[i]; val dB=B-tB[i]
                val d = dL*dL + dA*dA + dB*dB
                if (d < bestD) { bestD = d; best = i }
                i++
            }
            out[k] = threads[best].argb or (0xFF shl 24)
        }
        return out
    }

    private fun orderedDitherEdgeAware(
        src: Bitmap,
        palette: IntArray,
        edge: BooleanArray,
        orderedBias: Boolean,
        shortlists: Array<IntArray>? = null,
        block: Int = 10
    ): Bitmap {
        val w = src.width; val h = src.height
        val out = IntArray(w*h)
        val (lab, _, _) = toLab(src)
        val nbx = (w + block - 1) / block

        // Bayer 8x8
        val bayer = intArrayOf(
            0,48,12,60,3,51,15,63,
            32,16,44,28,35,19,47,31,
            8,56,4,52,11,59,7,55,
            40,24,36,20,43,27,39,23,
            2,50,14,62,1,49,13,61,
            34,18,46,30,33,17,45,29,
            10,58,6,54,9,57,5,53,
            42,26,38,22,41,25,37,21
        )

        val pL = FloatArray(palette.size); val pA = FloatArray(palette.size); val pB = FloatArray(palette.size)
        for (i in palette.indices) {
            val c = palette[i]
            val labT = rgbToOkLab((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            pL[i]=labT[0]; pA[i]=labT[1]; pB[i]=labT[2]
        }

        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val i = y*w + x
                val L = lab[i*3]; val A = lab[i*3+1]; val B = lab[i*3+2]

                // Кандидаты: либо shortlist блока, либо вся палитра
                val cand: IntArray? = shortlists?.getOrNull((y / block) * nbx + (x / block))
                // Находим 2 ближайших цвета
                var b1Idx=-1
                var b2Idx=-1
                var d1=Float.POSITIVE_INFINITY
                var d2=Float.POSITIVE_INFINITY
                var k=0
                val end = cand?.size ?: palette.size
                while (k < end) {
                    val idx = cand?.get(k) ?: k
                    val dl=L-pL[idx]; val da=A-pA[idx]; val db=B-pB[idx]
                    val dist = dl*dl + da*da + db*db
                    if (dist < d1) { d2=d1; b2Idx=b1Idx; d1=dist; b1Idx=idx }
                    else if (dist < d2) { d2=dist; b2Idx=idx }
                    k++
                }

                // Сила дитера: на краях приглушаем
                val edgeDamp = if (edge[i]) 0.25f else 1f
                val t = ((bayer[(y and 7)*8 + (x and 7)] + 0.5f) / 64f)
                val prob = (d2 / (d1 + d2 + 1e-9f)).toFloat()
                val bias = if (orderedBias) 0.1f else 0f
                val chooseFirst = t * edgeDamp + bias < prob
                val c = if (chooseFirst) palette[b1Idx] else palette[b2Idx]
                out[i] = c or (0xFF shl 24)

                x++
            }
            y++
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun collectUsedSwatches(bmp: Bitmap, palette: IntArray): List<Swatch> {
        // Преобразуем в минимальные Swatch с автогенерацией имени по HEX
        return palette.distinct().map { argb ->
            val hex = String.format("#%06X", argb and 0xFFFFFF)
            Swatch(name = hex, code = hex, argb = argb)
        }
    }

    // ──────────────────────── Метрики PHOTO ────────────────────────────

    private fun measurePhotoMetrics(ref: Bitmap, grid: Bitmap, palette: IntArray, p: PhotoConfig.Params): Metrics {
        val dE95 = deltaE95Between(ref, grid)
        val edgeSSIM = edgeSSIM(edgeMap(ref), edgeMap(grid))
        val hfRetain = hfRetainRatio(ref, grid)
        val band = bandingIndex(grid)
        val confetti = confettiRatio(grid)
        val score = scorePhoto(dE95, edgeSSIM, hfRetain, band, confetti)
        val reasons = mutableListOf<String>()
        return Metrics(dE95, edgeSSIM, hfRetain, band, confetti, score, passed = false, reasons)
    }

    private fun passPhoto(m: Metrics, p: PhotoConfig.Params): Pair<Boolean, List<String>> {
        val reasons = mutableListOf<String>()
        if (m.dE95 > p.dE95Max) reasons += "ΔE95>${fmt(p.dE95Max)}"
        if (m.edgeSSIM < p.edgeSSIMMin) reasons += "EdgeSSIM<${fmt(p.edgeSSIMMin)}"
        if (m.hfRetain < p.hfRetainMin) reasons += "HF<${fmt(p.hfRetainMin)}"
        if (m.banding > p.bandingMax) reasons += "Banding>${fmt(p.bandingMax)}"
        if (m.confetti > p.confettiMax) reasons += "Confetti>${fmt(p.confettiMax)}"
        val pass = reasons.isEmpty() && m.score >= p.scoreMin
        return pass to reasons
    }

    private fun scorePhoto(dE95: Float, edge: Float, hf: Float, band: Float, conf: Float): Float {
        fun normDE(x: Float) = (1f - (x / 24f)).coerceIn(0f, 1f)
        return (0.30f * normDE(dE95)
                + 0.20f * (1f - band)
                + 0.15f * edge
                + 0.15f * hf
                + 0.10f * (1f - conf)
                + 0.10f * 1f // запас под ROI/DitherEnergy на следующих спринтах
                ).coerceIn(0f, 1f)
    }

    // ──────────────────────── Вспомогательные ─────────────────────────

    private fun scaleToMaxSide(src: Bitmap, maxSide: Int): Bitmap {
        val side = max(src.width, src.height)
        if (side <= maxSide) return src
        val sc = maxSide.toFloat() / side
        return Bitmap.createScaledBitmap(src, max(1, (src.width*sc).roundToInt()), max(1, (src.height*sc).roundToInt()), true)
    }

    private fun descreenLite(src: Bitmap): Bitmap {
        // Упрощённый дескрин: мягкий гаусс + лёгкий bilateral (приближённый).
        val g = gaussianBlur(src, radius = 1)
        return bilateralLite(g, iters = 1)
    }
    private fun denoiseLite(src: Bitmap): Bitmap = median3x3(src)
    private fun grayWorldWB(src: Bitmap): Bitmap {
        val w=src.width; val h=src.height; val n=w*h
        val px=IntArray(n); src.getPixels(px,0,w,0,0,w,h)
        var sr=0L; var sg=0L; var sb=0L
        for (c in px){ sr+=(c ushr 16) and 0xFF; sg+=(c ushr 8) and 0xFF; sb+=c and 0xFF }
        val mr=sr.toFloat()/n; val mg=sg.toFloat()/n; val mb=sb.toFloat()/n
        val avg=(mr+mg+mb)/3f
        val kr = avg / max(1f, mr)
        val kg = avg / max(1f, mg)
        val kb = avg / max(1f, mb)
        var i=0
        while (i<n){
            val c=px[i]
            val r = (((c ushr 16) and 0xFF).toFloat() * kr).roundToInt().coerceIn(0, 255)
            val g = (((c ushr 8) and 0xFF).toFloat() * kg).roundToInt().coerceIn(0, 255)
            val b = ((c and 0xFF).toFloat() * kb).roundToInt().coerceIn(0, 255)
            px[i]=(0xFF shl 24) or (r shl 16) or (g shl 8) or b; i++
        }
        val out=Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888)
        out.setPixels(px,0,w,0,0,w,h); return out
    }

    private fun gaussianBlur(src: Bitmap, radius: Int): Bitmap {
        if (radius <= 0) return src
        val w=src.width; val h=src.height; val n=w*h
        val px=IntArray(n); src.getPixels(px,0,w,0,0,w,h)
        val out=IntArray(n)
        val kernel = when(radius){
            1 -> floatArrayOf(1f,2f,1f)
            2 -> floatArrayOf(1f,4f,6f,4f,1f)
            else -> floatArrayOf(1f,2f,1f)
        }
        fun passX(inp:IntArray, out:IntArray){
            var y=0
            while (y<h){
                var x=0
                while (x<w){
                    var ar=0f; var ag=0f; var ab=0f; var norm=0f
                    var k=-radius
                    while (k<=radius){
                        val xx=(x+k).coerceIn(0,w-1)
                        val c=inp[y*w+xx]; val r=(c ushr 16) and 0xFF; val g=(c ushr 8) and 0xFF; val b=c and 0xFF
                        val wv=kernel[k+radius]; norm+=wv
                        ar+=r*wv; ag+=g*wv; ab+=b*wv; k++
                    }
                    out[y*w+x]=(0xFF shl 24) or ((ar/norm).roundToInt() shl 16) or ((ag/norm).roundToInt() shl 8) or (ab/norm).roundToInt(); x++
                }
                y++
            }
        }
        val tmp=IntArray(n); passX(px,tmp); passX(tmp,out)
        val dst=Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888); dst.setPixels(out,0,w,0,0,w,h); return dst
    }

    private fun median3x3(src: Bitmap): Bitmap {
        val w=src.width; val h=src.height; val n=w*h
        val px=IntArray(n); src.getPixels(px,0,w,0,0,w,h)
        val out=IntArray(n)
        var y=0
        while (y<h){
            var x=0
            while (x<w){
                val rs=IntArray(9); val gs=IntArray(9); val bs=IntArray(9); var t=0
                var yy=max(0,y-1)
                while (yy<=min(h-1,y+1)){
                    var xx=max(0,x-1)
                    while (xx<=min(w-1,x+1)){
                        val c=px[yy*w+xx]; rs[t]=(c ushr 16) and 0xFF; gs[t]=(c ushr 8) and 0xFF; bs[t]=c and 0xFF
                        t++; xx++
                    }
                    yy++
                }
                rs.sort(); gs.sort(); bs.sort()
                out[y*w+x]=(0xFF shl 24) or (rs[4] shl 16) or (gs[4] shl 8) or bs[4]; x++
            }
            y++
        }
        val dst=Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888); dst.setPixels(out,0,w,0,0,w,h); return dst
    }

    private fun bilateralLite(src: Bitmap, iters: Int): Bitmap {
        var b = src
        repeat(iters) {
            b = gaussianBlur(b, radius = 1)
        }
        return b
    }

    private fun sobelMask(bmp: Bitmap): BooleanArray = edgeMap(bmp)

    // ---- Метрики/карты, повторяем минимально нужное из дискретной ветки ----

    private fun edgeMap(bmp: Bitmap): BooleanArray {
        val w=bmp.width; val h=bmp.height
        val px=IntArray(w*h); bmp.getPixels(px,0,w,0,0,w,h)
        val Y=IntArray(w*h)
        var i=0; while(i<px.size){ val c=px[i]; Y[i]=(54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256; i++ }
        val mag=IntArray(w*h)
        var y=1; while(y<h-1){ var x=1; while(x<w-1){
            val p=y*w+x
            val gx=-Y[p-w-1]-2*Y[p-1]-Y[p+w-1] + Y[p-w+1]+2*Y[p+1]+Y[p+w+1]
            val gy= Y[p-w-1]+2*Y[p-w]+Y[p-w+1] - Y[p+w-1]-2*Y[p+w]-Y[p+w+1]
            mag[p]=abs(gx)+abs(gy); x++ }
            y++ }
        val nz=mag.filter{it>0}.sorted()
        val thr= if (nz.isEmpty()) 255 else nz[(nz.size*0.90f).toInt().coerceAtMost(nz.size-1)]
        return BooleanArray(w*h){ i2 -> mag[i2] >= thr }
    }

    private fun edgeSSIM(a: BooleanArray, b: BooleanArray): Float {
        val n=min(a.size,b.size)
        var muA=0.0; var muB=0.0
        for(i in 0 until n){ if(a[i]) muA+=1.0; if(b[i]) muB+=1.0 }
        muA/=n; muB/=n
        var va=0.0; var vb=0.0; var cab=0.0
        for(i in 0 until n){
            val xa= if(a[i]) 1.0 else 0.0; val xb= if(b[i]) 1.0 else 0.0
            va+=(xa-muA)*(xa-muA); vb+=(xb-muB)*(xb-muB); cab+=(xa-muA)*(xb-muB)
        }
        val C1=0.01*0.01; val C2=0.03*0.03
        val num=(2*muA*muB + C1)*(2*cab + C2)
        val den=(muA*muA + muB*muB + C1)*(va + vb + C2)
        return (num/den).toFloat().coerceIn(0f,1f)
    }

    private fun hfRetainRatio(a: Bitmap, b: Bitmap): Float {
        fun energy(x: Bitmap): Double {
            val w=x.width; val h=x.height
            val px=IntArray(w*h); x.getPixels(px,0,w,0,0,w,h)
            var sum=0.0
            var y=1;
            while(y<h-1){
                var x2=1;
                while(x2<w-1){
                val i=y*w+x2
                val gx = ((px[i+1] and 0xFF) - (px[i-1] and 0xFF)).toDouble()
                val gy = ((px[i+w] and 0xFF) - (px[i-w] and 0xFF)).toDouble()
                sum += abs(gx) + abs(gy)
                    x2++
                }
                y++
            }
            return sum
        }
        val ea=energy(a); val eb=energy(b)
        if (ea <= 1e-9) return 1f
        return (eb/ea).toFloat().coerceIn(0f,1f)
    }

    private fun deltaE95Between(a: Bitmap, b: Bitmap): Float {
        val w=min(a.width,b.width); val h=min(a.height,b.height)
        val A=IntArray(w*h); val B=IntArray(w*h)
        a.getPixels(A,0,w,0,0,w,h); b.getPixels(B,0,w,0,0,w,h)
        val diffs=FloatArray(w*h)
        var i=0
        while (i<w*h){
            val c1=A[i]; val c2=B[i]
            val l1=rgbToOkLab((c1 ushr 16) and 0xFF, (c1 ushr 8) and 0xFF, c1 and 0xFF)
            val l2=rgbToOkLab((c2 ushr 16) and 0xFF, (c2 ushr 8) and 0xFF, c2 and 0xFF)
            val dL=l1[0]-l2[0]; val dA=l1[1]-l2[1]; val dB=l1[2]-l2[2]
            diffs[i]=sqrt(dL*dL + dA*dA + dB*dB); i++
        }
        return percentile(diffs, 95f)
    }

    private fun bandingIndex(bmp: Bitmap): Float {
        val (lab, w, h) = toLab(bmp)
        // Простая эвристика: доля длинных плоских пробегов L* по строкам/столбцам
        fun runs1d(values: FloatArray, len: Int, stride: Int): Int {
            var bad = 0
            var i = 0
            while (i < len-3) {
                val l0=values[i]; val l1=values[i+stride]; val l2=values[i+2*stride]; val l3=values[i+3*stride]
                if (abs(l0-l1)<0.005f && abs(l1-l2)<0.005f && abs(l2-l3)<0.005f) bad++
                i += stride
            }
            return bad
        }
        var bad=0; var all=0
        var y=0
        while (y<h){
            bad += runs1d(lab, w, 3)
            all += w; y++
        }
        var x=0
        while (x<w){
            bad += runs1d(lab, h, w*3)
            all += h; x++
        }
        return (bad.toFloat()/max(1,all).toFloat()).coerceIn(0f,1f)
    }

    private fun confettiRatio(bmp: Bitmap): Float {
        val w=bmp.width; val h=bmp.height
        val px=IntArray(w*h); bmp.getPixels(px,0,w,0,0,w,h)
        var bad=0; var all=0
        var y=1
        while (y<h-1){
            var x=1
            while (x<w-1){
                val i=y*w+x; val c=px[i]
                all++
                if (px[i-1]!=c && px[i+1]!=c && px[i-w]!=c && px[i+w]!=c) bad++
                x++
            }
            y++
        }
        return if (all==0) 0f else bad.toFloat()/all.toFloat()
    }

    private fun cullIsolatesToMajority(src: Bitmap): Bitmap {
        val w=src.width; val h=src.height; val n=w*h
        val px=IntArray(n); src.getPixels(px,0,w,0,0,w,h)
        var y=1
        while (y<h-1){
            var x=1
            while (x<w-1){
                val i=y*w+x; val c=px[i]
                if (px[i-1]!=c && px[i+1]!=c && px[i-w]!=c && px[i+w]!=c){
                    val a=px[i-1]; val b=px[i+1]; val d=px[i-w]; val e=px[i+w]
                    val repl = when {
                        a==b || a==d || a==e -> a
                        b==d || b==e -> b
                        d==e -> d
                        else -> a
                    }
                    px[i]=repl
                }
                x++
            }
            y++
        }
        val out=Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888)
        out.setPixels(px,0,w,0,0,w,h); return out
    }

    // ---- Представление / конвертации ----
    private data class LabBuf(val lab: FloatArray, val w:Int, val h:Int)
    private fun toLab(b: Bitmap): LabBuf {
        val w=b.width; val h=b.height; val n=w*h
        val px=IntArray(n); b.getPixels(px,0,w,0,0,w,h)
        val lab=FloatArray(n*3)
        var i=0
        while (i<n){
            val c=px[i]
            val L=rgbToOkLab((c ushr 16) and 0xFF,(c ushr 8) and 0xFF,c and 0xFF)
            lab[i*3]=L[0]; lab[i*3+1]=L[1]; lab[i*3+2]=L[2]; i++
        }
        return LabBuf(lab,w,h)
    }

    private fun percentile(arr: FloatArray, p: Float): Float {
        if (arr.isEmpty()) return 0f
        val tmp = arr.copyOf(); Arrays.sort(tmp)
        val idx = ((p/100f) * (tmp.size-1)).roundToInt().coerceIn(0, tmp.size-1)
        return tmp[idx]
    }

    // ---- Shortlists: топ‑M ближайших ниток на блок 10×10 по среднему цвета блока ----
    private fun buildShortlists(src: Bitmap, palette: IntArray, M: Int, block: Int): Array<IntArray> {
        val (lab, w, h) = toLab(src)
        val nbx = (w + block - 1) / block
        val nby = (h + block - 1) / block
        val pL = FloatArray(palette.size); val pA = FloatArray(palette.size); val pB = FloatArray(palette.size)
        for (i in palette.indices) {
            val c = palette[i]
            val t = rgbToOkLab((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            pL[i]=t[0]; pA[i]=t[1]; pB[i]=t[2]
        }
        fun nearestM(meanL:Float, meanA:Float, meanB:Float): IntArray {
            val tmp = (0 until palette.size).map { k ->
                val dL=meanL-pL[k]; val dA=meanA-pA[k]; val dB=meanB-pB[k]
                k to (dL*dL + dA*dA + dB*dB)
            }.sortedBy { it.second }.take(M).map { it.first }.toIntArray()
            return if (tmp.size == M) tmp else IntArray(M){ tmp.getOrElse(it){0} }
        }
        val out = Array(nbx*nby) { IntArray(M){0} }
        var by=0
        while (by<nby) {
            var bx=0
            while (bx<nbx) {
                val x0 = bx*block; val y0 = by*block
                val x1 = min(w, x0+block); val y1 = min(h, y0+block)
                var sL=0f; var sA=0f; var sB=0f; var cnt=0
                var y=y0
                while (y<y1) { var x=x0
                    while (x<x1) {
                        val i = y*w + x
                        sL+=lab[i*3]; sA+=lab[i*3+1]; sB+=lab[i*3+2]; cnt++
                        x++
                    }
                    y++
                }
                val meanL=sL/max(1,cnt).toFloat(); val meanA=sA/max(1,cnt).toFloat(); val meanB=sB/max(1,cnt).toFloat()
                out[by*nbx+bx] = nearestM(meanL,meanA,meanB)
                bx++
            }
            by++
        }
        return out
    }

    // ---- Merge chroma twins: слияние близких цветов палитры (ΔE* < ~3) ----
    private fun mergeChromaTwins(palette: IntArray, thr: Float = 0.03f): IntArray {
        val keep = ArrayList<Int>(palette.size)
        val kL = ArrayList<Float>(); val kA = ArrayList<Float>(); val kB = ArrayList<Float>()
        loop@ for (c in palette) {
            val lab = rgbToOkLab((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            for (i in keep.indices) {
                val dL = lab[0]-kL[i]; val dA=lab[1]-kA[i]; val dB=lab[2]-kB[i]
                if ((dL*dL + dA*dA + dB*dB) < thr*thr) continue@loop
            }
            keep += c; kL += lab[0]; kA += lab[1]; kB += lab[2]
        }
        return keep.toIntArray()
    }

    // ---- Оценка энергии дитера на сабсемпле (доля «вторых» выборов) ----
    private fun ditherEnergyProbe(bmp: Bitmap, palette: IntArray, edge: BooleanArray): Float {
        val (lab,w,h) = toLab(bmp)
        val pL = FloatArray(palette.size); val pA = FloatArray(palette.size); val pB = FloatArray(palette.size)
        for (i in palette.indices) {
            val c=palette[i]; val t=rgbToOkLab((c shr 16) and 0xFF,(c shr 8) and 0xFF,c and 0xFF)
            pL[i]=t[0]; pA[i]=t[1]; pB[i]=t[2]
        }
        var num=0; var den=0
        var y=0
        while (y<h) {
            var x=0
            while (x<w) {
                val i=y*w+x
                if (edge[i]) { x++; continue } // на краях не считаем
                val L=lab[i*3]; val A=lab[i*3+1]; val B=lab[i*3+2]
                var d1=Float.POSITIVE_INFINITY; var d2=Float.POSITIVE_INFINITY; var k=0
                while (k<palette.size) {
                    val dl=L-pL[k]; val da=A-pA[k]; val db=B-pB[k]
                    val d=dl*dl+da*da+db*db
                    if (d<d1){ d2=d1; d1=d } else if (d<d2){ d2=d }
                    k++
                }
                val probSecond = (d2 / (d1 + d2 + 1e-9f))
                num += (probSecond*1000f).toInt(); den += 1000
                x++
            }
            y++
        }
        return if (den==0) 0f else num.toFloat()/den.toFloat()
    }

    // ---- ROI: нормированная величина градиента (0..1) ----
    private fun edgeMagnitude(bmp: Bitmap): FloatArray {
        val w=bmp.width; val h=bmp.height
        val px=IntArray(w*h); bmp.getPixels(px,0,w,0,0,w,h)
        val Y=IntArray(w*h); var i=0
        while (i<px.size){
            val c=px[i]; Y[i]=(54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256; i++
        }
        val mag=FloatArray(w*h)
        var y=1; while (y<h-1){
            var x=1; while (x<w-1){
                val p=y*w+x
                val gx=-Y[p-w-1]-2*Y[p-1]-Y[p+w-1] + Y[p-w+1]+2*Y[p+1]+Y[p+w+1]
                val gy= Y[p-w-1]+2*Y[p-w]+Y[p-w+1] - Y[p+w-1]-2*Y[p+w]-Y[p+w+1]
                mag[p] = (abs(gx)+abs(gy)).toFloat(); x++
            }
            y++
        }
        // нормировка к 95‑му перцентилю
        val copy = mag.copyOf(); Arrays.sort(copy)
        val m95 = copy[((copy.size-1)*0.95f).roundToInt().coerceIn(0,copy.size-1)]
        if (m95 <= 0f) return FloatArray(w*h){0f}
        val out = FloatArray(w*h){ i2 -> (mag[i2]/m95).coerceIn(0f,1f) }
        return out
    }

    // ---- Regularizer: равномерная «лестница» по L* (внутри центроидов) ----
    private fun enforceLLadder(centers: FloatArray, counts: FloatArray, lambda: Float) {
        val K = centers.size/3
        val idx = (0 until K).sortedBy { centers[it*3] }
        for (p in 1 until K-1) {
            val i = idx[p-1]; val j = idx[p]; val k = idx[p+1]
            val li = centers[i*3]; val lj = centers[j*3]; val lk = centers[k*3]
            val target = (li + lk) * 0.5f
            val w = (counts[j] / (counts.sum().coerceAtLeast(1e-6f))).coerceIn(0.05f, 0.3f)
            centers[j*3] = (lj + lambda * w * (target - lj)).coerceIn(0f,1f)
        }
    }

    // ---- Regularizer: анти‑конфетти прокси (на сабсемпле) ----
    private fun confettiProxyNudge(lab: FloatArray, w:Int, h:Int, centers: FloatArray, lambda: Float) {
        // Быстрый лейбл‑мап на 1/4 разрешения
        val sx = max(1, w/4); val sy = max(1, h/4)
        val labels = IntArray(sx*sy)
        var y=0
        while (y<sy){
            var x=0
            while (x<sx){
                val ix = (y*4)*w + (x*4)
                val L=lab[ix*3]; val A=lab[ix*3+1]; val B=lab[ix*3+2]
                var best=0; var dBest=Float.POSITIVE_INFINITY; var k=0
                while (k<centers.size){
                    val dL=L-centers[k]; val dA=A-centers[k+1]; val dB=B-centers[k+2]
                    val d=dL*dL+dA*dA+dB*dB
                    if (d<dBest){ dBest=d; best=k/3 }
                    k+=3
                }
                labels[y*sx+x]=best; x++
            }
            y++
        }
        // Оценка изолятов по лейблу
        val cc = IntArray(centers.size/3)
        val iso = IntArray(centers.size/3)
        var yy=1
        while (yy<sy-1){
            var xx=1
            while (xx<sx-1){
                val i=yy*sx+xx; val c=labels[i]
                cc[c]++
                if (labels[i-1]!=c && labels[i+1]!=c && labels[i-sx]!=c && labels[i+sx]!=c) iso[c]++
                xx++
            }
            yy++
        }
        for (k in 0 until cc.size){
            val size=max(1,cc[k]); val ratio=iso[k].toFloat()/size.toFloat()
            if (ratio>0.20f && size < (sx*sy*0.01f)) {
                // Нудж: подтянуть центр к ближайшему «крупному» соседу
                var bestN=-1; var bestD=Float.POSITIVE_INFINITY
                var j=0
                while (j<centers.size){
                    if (j/3==k){ j+=3; continue }
                    val dL=centers[k*3]-centers[j]; val dA=centers[k*3+1]-centers[j+1]; val dB=centers[k*3+2]-centers[j+2]
                    val d=dL*dL+dA*dA+dB*dB
                    if (d<bestD){ bestD=d; bestN=j/3 }
                    j+=3
                }
                if (bestN>=0){
                    centers[k*3  ] = centers[k*3  ] + lambda*0.15f*(centers[bestN*3  ]-centers[k*3  ])
                    centers[k*3+1] = centers[k*3+1] + lambda*0.15f*(centers[bestN*3+1]-centers[k*3+1])
                    centers[k*3+2] = centers[k*3+2] + lambda*0.15f*(centers[bestN*3+2]-centers[k*3+2])
                }
            }
        }
    }

    // ---- Thread look‑ahead: проекция центроидов к ближайшей нитке с cap ----
    private fun projectTowardsThreads(centers: FloatArray, threads: List<Swatch>, cap: Float) {
        val tL = FloatArray(threads.size); val tA = FloatArray(threads.size); val tB = FloatArray(threads.size)
        for (i in threads.indices) {
            val c=threads[i].argb; val lab=rgbToOkLab((c shr 16) and 0xFF,(c shr 8) and 0xFF,c and 0xFF)
            tL[i]=lab[0]; tA[i]=lab[1]; tB[i]=lab[2]
        }
        var k=0
        while (k<centers.size){
            val L=centers[k]; val A=centers[k+1]; val B=centers[k+2]
            var best=0; var dBest=Float.POSITIVE_INFINITY
            for (i in threads.indices){
                val dL=L-tL[i]; val dA=A-tA[i]; val dB=B-tB[i]
                val d=dL*dL+dA*dA+dB*dB
                if (d<dBest){ dBest=d; best=i }
            }
            val d = sqrt(dBest)
            if (d>1e-6f){
                val step = min(1f, cap/d)
                centers[k  ] = (L + step*(tL[best]-L)).coerceIn(0f,1f)
                centers[k+1] = A + step*(tA[best]-A)
                centers[k+2] = B + step*(tB[best]-B)
            }
            k+=3
        }
    }

    private fun bandingIndexFromLQuant(lab: FloatArray, centers: FloatArray, w:Int, h:Int): Float {
        // Для Kneedle: грубо оцениваем бэндинг как долю длинных плоских пробегов L* после квантизации в центры
        val qL = FloatArray(w*h)
        var i=0
        while (i<w*h){
            val L=lab[i*3]; val A=lab[i*3+1]; val B=lab[i*3+2]
            var best=Float.POSITIVE_INFINITY; var bestL=0f; var j=0
            while (j<centers.size){
                val dL=L-centers[j]; val dA=A-centers[j+1]; val dB=B-centers[j+2]
                val dist=dL*dL + dA*dA + dB*dB
                if (dist<best){ best=dist; bestL=centers[j] }
                j+=3
            }
            qL[i]=bestL; i++
        }
        // как в bandingIndex()
        var bad=0; var all=0
        var y=0
        while (y<h){
            var x=0
            while (x<w-3){
                val i0=y*w+x
                if (abs(qL[i0]-qL[i0+1])<0.005f && abs(qL[i0+1]-qL[i0+2])<0.005f && abs(qL[i0+2]-qL[i0+3])<0.005f) bad++
                x++
            }
            all += w; y++
        }
        return (bad.toFloat()/max(1,all).toFloat()).coerceIn(0f,1f)
    }

    // OKLab utils (совместимы с дискретной веткой)
    private fun rgbToOkLab(r8: Int, g8: Int, b8: Int): FloatArray {
        fun srgbToLinear(c: Int): Double { val s=c/255.0; return if (s<=0.04045) s/12.92 else ((s+0.055)/1.055).pow(2.4) }
        val r=srgbToLinear(r8); val g=srgbToLinear(g8); val b=srgbToLinear(b8)
        val l=0.4122214708*r + 0.5363325363*g + 0.0514459929*b
        val m=0.2119034982*r + 0.6806995451*g + 0.1073969566*b
        val s=0.0883024619*r + 0.2817188376*g + 0.6299787005*b
        val l_ = cbrt(l); val m_ = cbrt(m); val s_ = cbrt(s)
        val L=(0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_).toFloat()
        val A=(1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_).toFloat()
        val B=(0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_).toFloat()
        return floatArrayOf(L,A,B)
    }

    private fun fmt(x: Float) = String.format(Locale.US, "%.3f", x)
}
