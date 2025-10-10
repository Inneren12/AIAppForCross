package com.appforcross.editor.auto.discrete

import android.graphics.Bitmap
import kotlin.math.*
import com.appforcross.core.palette.Swatch
import com.appforcross.editor.auto.detect.SmartSceneDetector // для Toggles

/**
 * Дискретная ветка V2:
 * - Перебор S из сетки, ресэмпл NN (без сглаживания).
 * - Опц. cel-collapse для градиентов (без дитера).
 * - Квантование к нитям активной палитры (OKLab NN).
 * - Метрики: EdgeSSIM (по контурной карте), ΔE95 (среднее по исходнику -> recolor),
 *   TinyDetailLoss, ConfettiRatio, HolesRetained (если TEXT=ON).
 * - Пороговые профили (WORDMARK/TEXT/AA/PIXEL_GRID через тумблеры).
 * - Выбор минимального S, прошедшего пороги и Score(S) ≥ T.
 */
object DiscretePipeline {

    // --- Публичные типы ---

    data class Config(
        val sizes: IntArray = intArrayOf(120, 160, 180, 200, 240, 300, 360, 400, 440),
        val targetScore: Float = 0.75f,
        val enableCelCollapse: Boolean = true
    )

    data class Metrics(
        val edgeSSIM: Float,
        val deltaE95: Float,
        val tinyDetailLoss: Float,
        val confettiRatio: Float,
        val holesRetained: Float?,   // null, если TEXT=false
        val score: Float,
        val passed: Boolean,
        val reasons: List<String>
    )

    data class Result(
        val image: Bitmap,            // итоговая сетка S (клетка=пиксель)
        val gridWidth: Int,
        val gridHeight: Int,
        val gridSizeS: Int,           // S (ширина в стежках)
        val usedSwatches: List<Swatch>,
        val metrics: Metrics
    )

    // --- Публичный API ---

    fun run(
        source: Bitmap,
        palette: List<Swatch>,
        toggles: SmartSceneDetector.Toggles,
        cfg: Config = Config()
    ): Result {
        require(source.width > 0 && source.height > 0) { "Empty bitmap" }
        require(palette.isNotEmpty()) { "Palette is empty" }

        // Предрасчёт палитры в OKLab
        val swL = FloatArray(palette.size)
        val swA = FloatArray(palette.size)
        val swB = FloatArray(palette.size)
        for (i in palette.indices) {
            val c = palette[i].argb
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            val lab = rgbToOkLab(r, g, b)
            swL[i] = lab[0]; swA[i] = lab[1]; swB[i] = lab[2]
        }

        // Эталон для EdgeSSIM / TinyDetail: downscale исходника до каждого S (BL)
        // и карта краёв для него
        var best: Result? = null
        var bestScore = -1f

        // Для PIXEL_GRID — наказываем S, у которых нецелое снижение масштаба
        fun integerPenalty(S: Int): Float {
            if (!toggles.pixelGrid) return 0f
            val ratio = source.width.toFloat() / S.toFloat()
            val frac = abs(ratio - ratio.roundToInt())
            return if (frac <= 0.02f) 0f else (frac * 2f).coerceAtMost(0.25f) // до -0.25 к score
        }

        // Подозрение на билинейное растяжение (сглаженные пиксели)
        val useBlockMajority = toggles.pixelGrid && suspectBilinearUpscale(source)
        for (S in cfg.sizes) {
            if (S < 8) continue
            val H = max(1, (source.height * (S.toFloat() / source.width)).roundToInt())
            // Оригинал → «эталон для краёв» (BL, мягко)
            val ref = Bitmap.createScaledBitmap(source, S, H, true)
            val refEdges = edgeMap(ref)

            // ── Жёсткий gate для PIXEL_GRID: только целочисленный масштаб
            if (toggles.pixelGrid) {
                val ratio = source.width.toFloat() / S.toFloat()
                val frac = abs(ratio - ratio.roundToInt())
                if (frac > 0.03f) {
                    // Сразу отклоняем кандидат
                    continue
                }
            }

            // Ресэмпл → сетка S
            var grid = if (useBlockMajority)
                blockMajorityResample(source, S, H)
            else
                Bitmap.createScaledBitmap(source, S, H, false)
            // Лёгкая пред-очистка конфетти перед квантованием (mode-filter 3×3 в пределах ΔE≈2)
            grid = precleanMode3x3WithinDE(grid)
            if (cfg.enableCelCollapse && !toggles.pixelGrid) {
                grid = celCollapse(grid) // 2–3 ступени по L* в областях с постоянным hue
            }

            // Квантование к нитям палитры (без дитера)
            val q = quantizeToPalette(grid, palette, swL, swA, swB)
            val usedIdx = q.usedSwatchIdx.toMutableSet()

            // (Опц.) упрощение роли/слияния близких цветов (ΔE≤4)
            roleMergeGreedy(q.pixels, S, H, swL, swA, swB, usedIdx)

            // Мини‑топология: срезать “одиночки” (если не образуют дырку/контур)
            cullIsolatesToMajority(q.pixels, S, H)
            // Мини‑топология: запрет пробегов длиной 1 (min‑run=2) по горизонтали/вертикали
            enforceMinRun2(q.pixels, S, H)

            // Метрики
            val edge = edgeSSIM(refEdges, edgeMap(q.bitmap))
            val dE = deltaE95AvgAgainstSubset(source, palette, usedIdx)
            val tiny = tinyDetailLoss(refEdges, edgeMap(q.bitmap))
            val confetti = confettiRatio(q.pixels, S, H)
            val holes = if (toggles.text) holesRetained(ref, q.bitmap) else null
            // 10×10 лимит: считаем превышение и долю “плохих” блоков
                        val (blkExcess, blkShare) = blockColorsStats(q.pixels, S, H, block = 10, limit = if (toggles.text) 3 else 4)

            // EC‑law(TEXT): ΔL* ≥ 25, ΔE ≥ 18 (OKLab‑proxy → пороги ~0.25 и ~0.18)
            val ecOk: Boolean
            val ecMsg: String?
            if (toggles.text) {
                val (dL, dEec) = textContrastStats(q.bitmap)
                val okL = dL >= 0.25f
                val okE = dEec >= 0.18f
                ecOk = okL && okE
                ecMsg = when {
                    !okL && !okE -> "EC‑law: ΔL<0.25 & ΔE<0.18"
                    !okL -> "EC‑law: ΔL<0.25"
                    !okE -> "EC‑law: ΔE<0.18"
                    else -> null
                }
            } else { ecOk = true; ecMsg = null }

            var score = scoreAggregate(
                edge = edge,
                dE = dE,
                confetti = confetti,
                tiny = tiny,
                holes = holes,
                toggles = toggles,
                blockPenalty = blkShare.coerceIn(0f,1f) * (blkExcess.coerceIn(0f,1f))
            ) - integerPenalty(S)

            // Доп. вес за EC‑law(TEXT): чем выше ΔL и ΔE относительно порогов, тем больше бонус
            if (toggles.text) {
                val (dL, dEec) = textContrastStats(q.bitmap)
                val sL = (dL / 0.25f).coerceIn(0f, 1f)   // 1.0 == порог ΔL*25
                val sE = (dEec / 0.18f).coerceIn(0f, 1f) // 1.0 == порог ΔE≈18
                val ecBoost = 0.10f * (0.5f*sL + 0.5f*sE)
                score += ecBoost
            }

            // Пороговые профили
            val (edgeMin, confMax, dEMax, holesMin) = when {
                toggles.text -> TripleEdge(edgeMin = 0.86f, confMax = 0.03f, dEMax = 12f, holesMin = 0.80f)
                toggles.aa ->  TripleEdge(edgeMin = 0.84f, confMax = 0.035f, dEMax = 15f, holesMin = null)
                else ->       TripleEdge(edgeMin = 0.85f, confMax = 0.035f, dEMax = 14f, holesMin = null)
            }

            val reasons = mutableListOf<String>()
            if (edge < edgeMin) reasons += "EdgeSSIM<${fmt(edgeMin)}"
            if (confetti > confMax) reasons += "Confetti>${fmt(confMax)}"
            if (dE > dEMax) reasons += "ΔE95>${fmt(dEMax)}"
            if (holesMin != null && (holes ?: 0f) < holesMin) reasons += "Holes<${fmt(holesMin)}"
            // 10×10: если много “плохих” блоков — FAIL
            if (blkShare > 0.35f && blkExcess > 0.30f) reasons += "Block10×10 limit exceeded"
            // EC‑law(TEXT): обязателен
            if (!ecOk) reasons += ecMsg!!
            val passed = reasons.isEmpty() && score >= cfg.targetScore
            val used = usedIdx.map { palette[it] }
            val res = Result(q.bitmap, S, H, S, used, Metrics(edge, dE, tiny, confetti, holes, score, passed, reasons))

            // Выбор минимального S c PASS; иначе — лучший score
            if (passed) {
                best = res
                break
            } else if (score > bestScore) {
                best = res; bestScore = score
            }
        }

        return requireNotNull(best) { "DiscretePipeline: no candidate produced" }
    }

    // --- Метрики и вспомогательные алгоритмы ---

    // 1) EdgeSSIM на бинарных картах краёв (Sobel->magnitude->threshold)
    private fun edgeSSIM(a: BooleanArray, b: BooleanArray): Float {
        // Преобразуем в 0/1
        val n = min(a.size, b.size)
        var muA = 0.0; var muB = 0.0
        for (i in 0 until n) { if (a[i]) muA += 1.0; if (b[i]) muB += 1.0 }
        muA /= n; muB /= n
        var va = 0.0; var vb = 0.0; var cab = 0.0
        for (i in 0 until n) {
            val xa = if (a[i]) 1.0 else 0.0
            val xb = if (b[i]) 1.0 else 0.0
            va += (xa - muA) * (xa - muA)
            vb += (xb - muB) * (xb - muB)
            cab += (xa - muA) * (xb - muB)
        }
        va /= n; vb /= n; cab /= n
        // SSIM на бинарном сигнале (C1,C2 маленькие, чтобы избежать деления на 0)
        val C1 = 0.01 * 0.01
        val C2 = 0.03 * 0.03
        val num = (2*muA*muB + C1) * (2*cab + C2)
        val den = (muA*muA + muB*muB + C1) * (va + vb + C2)
        return (num / den).toFloat().coerceIn(0f, 1f)
    }

    // 2) ΔE95 (усреднение по исходнику -> ближайший цвет из выбранного подмножества палитры)
    private fun deltaE95AvgAgainstSubset(src: Bitmap, palette: List<Swatch>, subset: Set<Int>): Float {
        if (subset.isEmpty()) return 0f
        val idx = subset.sorted()
        val L = FloatArray(idx.size)
        val A = FloatArray(idx.size)
        val B = FloatArray(idx.size)
        for (i in idx.indices) {
            val c = palette[idx[i]].argb
            val lab = rgbToOkLab((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            L[i]=lab[0]; A[i]=lab[1]; B[i]=lab[2]
        }
        val n = src.width * src.height
        val px = IntArray(n)
        src.getPixels(px, 0, src.width, 0, 0, src.width, src.height)
        var sum = 0.0
        var i = 0
        while (i < n) {
            val c = px[i]
            val lab = rgbToOkLab((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            var best = 0
            var bestD = Float.POSITIVE_INFINITY
            var k = 0
            while (k < L.size) {
                val dl = lab[0]-L[k]; val da = lab[1]-A[k]; val db = lab[2]-B[k]
                val d = sqrt((dl*dl + da*da + db*db).toDouble()) // OKLab euclid ~ ΔE95 proxy
                if (d < bestD) { bestD = d.toFloat(); best = k }
                k++
            }
            // Суммируем ΔE (OKLab-euclid как быстрый прокси; можно заменить на СIEDE2000)
            sum += bestD
            i++
        }
        return (sum / n).toFloat()
    }

    // 3) TinyDetailLoss: доля «крошечных» компонент в refEdges, отсутствующих в qEdges
    private fun tinyDetailLoss(refEdges: BooleanArray, qEdges: BooleanArray): Float {
        val w = sqrt(refEdges.size.toFloat()).roundToInt()
        val h = if (w == 0) 0 else refEdges.size / w
        fun ccCount(edges: BooleanArray, tinyMax: Int): Int {
            val seen = BooleanArray(edges.size)
            var cnt = 0
            val qx = IntArray(edges.size)
            val qy = IntArray(edges.size)
            var qs=0; var qe=0
            var i=0
            while (i < edges.size) {
                if (seen[i] || !edges[i]) { i++; continue }
                var area=0
                qs=0; qe=0
                qx[qe]=i%w; qy[qe]=i/w; qe++
                seen[i]=true
                while (qs<qe) {
                    val x=qx[qs]; val y=qy[qs]; qs++
                    area++
                    val nbs = intArrayOf((y*w+x-1),(y*w+x+1),((y-1)*w+x),((y+1)*w+x))
                    for (nb in nbs) {
                        if (nb<0 || nb>=edges.size || seen[nb]) continue
                        val ny=nb/w; val nx=nb-ny*w
                        if (abs(nx-x)+abs(ny-y)!=1) continue
                        if (edges[nb]) { seen[nb]=true; qx[qe]=nx; qy[qe]=ny; qe++ }
                    }
                }
                if (area <= tinyMax) cnt++
                i++
            }
            return cnt
        }
        val tinyRef = ccCount(refEdges, tinyMax = 4)
        if (tinyRef == 0) return 0f
        val tinyQ = ccCount(qEdges, tinyMax = 4)
        val lost = (tinyRef - min(tinyRef, tinyQ)).coerceAtLeast(0)
        return (lost.toFloat() / tinyRef.toFloat()).coerceIn(0f, 1f)
    }

    // 4) ConfettiRatio: доля клеток, отличных от всех 4 соседей
    private fun confettiRatio(px: IntArray, w: Int, h: Int): Float {
        var bad = 0; var all = 0
        var y = 1
        while (y < h - 1) {
            var x = 1
            while (x < w - 1) {
                val i = y*w + x
                val c = px[i]
                all++
                if (px[i-1] != c && px[i+1] != c && px[i-w] != c && px[i+w] != c) bad++
                x++
            }
            y++
        }
        return if (all == 0) 0f else (bad.toFloat() / all.toFloat())
    }

    // 5) HolesRetained: отношение числа «дыр» (Otsu-бинаризация) до/после
    private fun holesRetained(ref: Bitmap, q: Bitmap): Float {
        val (binA, w, h) = otsuBinary(ref)
        val (binB, _, _) = otsuBinary(q)
        fun holes(bin: BooleanArray): Int {
            // считаем CC по фону и такие, что не касаются рамки → это «дырки»
            val seen = BooleanArray(bin.size)
            val qx = IntArray(bin.size); val qy = IntArray(bin.size)
            var qs=0; var qe=0; var holes=0
            var i=0
            while (i < bin.size) {
                if (seen[i] || bin[i]) { i++; continue }
                val x0 = i % w; val y0 = i / w
                qs=0; qe=0
                qx[qe]=x0; qy[qe]=y0; qe++
                seen[i]=true
                var touchesBorder = false
                while (qs<qe) {
                    val x=qx[qs]; val y=qy[qs]; qs++
                    if (x==0 || x==w-1 || y==0 || y==h-1) touchesBorder = true
                    val nbs = arrayOf(x-1 to y, x+1 to y, x to y-1, x to y+1)
                    for ((nx,ny) in nbs) {
                        if (nx<0 || nx>=w || ny<0 || ny>=h) continue
                        val j = ny*w + nx
                        if (seen[j] || bin[j]) continue
                        seen[j]=true; qx[qe]=nx; qy[qe]=ny; qe++
                    }
                }
                if (!touchesBorder) holes++
                i++
            }
            return holes
        }
        val a = holes(binA); val b = holes(binB)
        return if (a==0) 1f else (b.toFloat()/a.toFloat()).coerceIn(0f,1f)
    }

    // --- Оценка: агрегированный Score ---
    private fun scoreAggregate(
        edge: Float,
        dE: Float,
        confetti: Float,
        tiny: Float,
        holes: Float?,
        toggles: SmartSceneDetector.Toggles,
        blockPenalty: Float
    ): Float {
        // Нормировка dE: 0..15 → 1..0
        fun dENorm(x: Float): Float = (1f - (x/15f)).coerceIn(0f, 1f)
        val base =
            0.42f * edge +
                    0.22f * dENorm(dE) +
                    0.18f * (1f - confetti) +
                    0.18f * (1f - tiny)
                    0.18f * (1f - tiny) -
                    0.16f * blockPenalty
        return if (toggles.text) base * 0.8f + 0.2f * (holes ?: 0f) else base
    }

    // --- Квантование к палитре ---
    private data class QuantOut(val bitmap: Bitmap, val pixels: IntArray, val usedSwatchIdx: IntArray)

    private fun quantizeToPalette(
        src: Bitmap,
        palette: List<Swatch>,
        swL: FloatArray, swA: FloatArray, swB: FloatArray
    ): QuantOut {
        val w = src.width; val h = src.height
        val n = w * h
        val inPx = IntArray(n)
        src.getPixels(inPx, 0, w, 0, 0, w, h)
        val outPx = IntArray(n)
        val used = HashSet<Int>(16)
        var i = 0
        while (i < n) {
            val c = inPx[i]
            val lab = rgbToOkLab((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            var best = 0
            var bestD = Float.POSITIVE_INFINITY
            var k = 0
            while (k < swL.size) {
                val dl = lab[0]-swL[k]; val da = lab[1]-swA[k]; val db = lab[2]-swB[k]
                val d = dl*dl + da*da + db*db
                if (d < bestD) { bestD = d; best = k }
                k++
            }
            val argb = palette[best].argb or (0xFF shl 24)
            outPx[i] = argb
            used += best
            i++
        }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(outPx, 0, w, 0, 0, w, h)
        return QuantOut(out, outPx, used.toIntArray())
    }

    // --- Greedy role-merge (ΔE<=4; плюс снятие конфетти) ---
    private fun roleMergeGreedy(
        px: IntArray, w: Int, h: Int,
        L: FloatArray, A: FloatArray, B: FloatArray,
        used: MutableSet<Int>
    ) {
        if (used.size <= 2) return
        // Частоты и соседства
        val freq = HashMap<Int, Int>()
        val adj = HashMap<Long, Int>() // key=min<<20|max, value=edge length
        fun key(a:Int,b:Int): Long { val x=min(a,b); val y=max(a,b); return (x.toLong() shl 20) or y.toLong() }
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val i = y*w + x
                val c = px[i]
                val idxTmp = usedIndexOfColor(c, L, A, B)
                if (idxTmp == null) { x++; continue }
                val idx = idxTmp
                freq[idx] = (freq[idx] ?: 0) + 1
                if (x+1 < w) {
                    val j = i+1
                    val idx2Tmp = usedIndexOfColor(px[j], L, A, B)
                    if (idx2Tmp == null) { x++; continue }
                    val idx2 = idx2Tmp
                    if (idx != idx2) adj[key(idx, idx2)] = (adj[key(idx, idx2)] ?: 0) + 1
                }
                if (y+1 < h) {
                    val j = i+w
                    val idx2Tmp = usedIndexOfColor(px[j], L, A, B)
                    if (idx2Tmp == null) { x++; continue }
                    val idx2 = idx2Tmp
                    if (idx != idx2) adj[key(idx, idx2)] = (adj[key(idx, idx2)] ?: 0) + 1
                }
                x++
            }
            y++
        }
        // Кандидаты на merge: ΔE<=4
        data class Cand(val a:Int, val b:Int, val edge:Int, val dE:Float)
        val cands = mutableListOf<Cand>()
        for ((k, e) in adj) {
            val a = (k shr 20).toInt()
            val b = (k and ((1 shl 20)-1).toLong()).toInt()
            val d = sqrt((L[a]-L[b]).pow(2) + (A[a]-A[b]).pow(2) + (B[a]-B[b]).pow(2))
            if (d <= 4f) cands += Cand(a,b,e,d)
        }
        // Слияние «маленького» в «большой»
        cands.sortedBy { it.dE }.forEach { c ->
            if (!used.contains(c.a) || !used.contains(c.b)) return@forEach
            val small = if ((freq[c.a] ?: 0) <= (freq[c.b] ?: 0)) c.a else c.b
            val big   = if (small == c.a) c.b else c.a
            // перекрашиваем small -> big
            var i=0
            while (i < px.size) {
                val idx = usedIndexOfColor(px[i], L, A, B)
                if (idx == small) {
                    val argb = labToArgb(L[big], A[big], B[big])
                    px[i] = argb
                }
                i++
            }
            used.remove(small)
            freq[big] = (freq[big] ?: 0) + (freq[small] ?: 0)
            freq.remove(small)
        }
    }
    private fun usedIndexOfColor(argb: Int, L: FloatArray, A: FloatArray, B: FloatArray): Int? {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = argb and 0xFF
        val lab = rgbToOkLab(r, g, b)
        var best = -1; var bestD = Float.POSITIVE_INFINITY
        var i = 0
        while (i < L.size) {
            val dl = lab[0]-L[i]; val da = lab[1]-A[i]; val db = lab[2]-B[i]
            val d = dl*dl + da*da + db*db
            if (d < bestD) { bestD = d; best = i }
            i++
        }
        return if (best >= 0) best else null
    }

    // --- Cel-collapse: 2–3 ступени L* при высокой константности hue (локально) ---
    private fun celCollapse(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height; val n = w*h
        val px = IntArray(n)
        src.getPixels(px, 0, w, 0, 0, w, h)
        val L = FloatArray(n); val A = FloatArray(n); val B = FloatArray(n)
        var i=0
        while (i<n) {
            val c = px[i]
            val lab = rgbToOkLab((c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF)
            L[i]=lab[0]; A[i]=lab[1]; B[i]=lab[2]; i++
        }
        // Простая локальная проверка «HueConstancy»: |∇A|+|∇B| мало → можно дробить L на 2–3 кванта
        val out = IntArray(n)
        fun quantL(l: Float, l0: Float, l1: Float, l2: Float): Float {
            return when {
                l < l0 -> l0
                l > l2 -> l2
                else -> if (abs(l - l1) < abs(l - (if (l < l1) l0 else l2))) l1 else if (l < l1) l0 else l2
            }
        }
        var y=1
        while (y<h-1) {
            var x=1
            while (x<w-1) {
                val i0 = y*w + x
                val gA = abs(A[i0-1]-A[i0+1]) + abs(A[i0-w]-A[i0+w])
                val gB = abs(B[i0-1]-B[i0+1]) + abs(B[i0-w]-B[i0+w])
                val flatHue = (gA + gB) < 0.05f
                val Lq = if (flatHue) {
                    // 3 уровня вокруг локального медианного
                    val lN = floatArrayOf(L[i0], L[i0-1], L[i0+1], L[i0-w], L[i0+w]).sorted()
                    val m = lN[2]
                    val l0 = (m - 0.10f).coerceIn(0f,1f)
                    val l1 = m
                    val l2 = (m + 0.10f).coerceIn(0f,1f)
                    quantL(L[i0], l0, l1, l2)
                } else L[i0]
                out[i0] = labToArgb(Lq, A[i0], B[i0])
                x++
            }
            y++
        }
        // рамку копируем как есть
        for (x in 0 until w) { out[x] = px[x]; out[(h-1)*w+x] = px[(h-1)*w+x] }
        for (y2 in 0 until h) { out[y2*w] = px[y2*w]; out[y2*w + (w-1)] = px[y2*w + (w-1)] }
        val dst = Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888)
        dst.setPixels(out, 0, w, 0, 0, w, h)
        return dst
    }

    // --- Edge map (Sobel + глобальный порог 90‑й перцентиль величины градиента) ---
    private fun edgeMap(bmp: Bitmap): BooleanArray {
        val w = bmp.width; val h = bmp.height
        val Y = IntArray(w*h)
        val px = IntArray(w*h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        var i=0
        while (i<px.size) {
            val c=px[i]
            Y[i] = (54*((c shr 16) and 0xFF) + 183*((c shr 8) and 0xFF) + 19*(c and 0xFF)) / 256
            i++
        }
        val mag = IntArray(w*h)
        var y=1
        while (y<h-1) {
            var x=1
            while (x<w-1) {
                val p = y*w + x
                val gx = -Y[p-w-1]-2*Y[p-1]-Y[p+w-1] + Y[p-w+1]+2*Y[p+1]+Y[p+w+1]
                val gy =  Y[p-w-1]+2*Y[p-w]+Y[p-w+1] - Y[p+w-1]-2*Y[p+w]-Y[p+w+1]
                mag[p] = abs(gx) + abs(gy)
                x++
            }
            y++
        }
        val nz = mag.filter { it>0 }.sorted()
        val thr = if (nz.isEmpty()) 255 else nz[(nz.size * 0.90f).toInt().coerceAtMost(nz.size-1)]
        val out = BooleanArray(w*h) { idx -> mag[idx] >= thr }
        return out
    }

    // --- Otsu-бинаризация для HolesRetained ---
    private data class BinOut(val bin: BooleanArray, val w:Int, val h:Int)
    private fun otsuBinary(b: Bitmap): BinOut {
        val w=b.width; val h=b.height
        val hist = IntArray(256)
        val px = IntArray(w*h); b.getPixels(px,0,w,0,0,w,h)
        var i=0
        while (i<px.size) {
            val c=px[i]
            val yv = (54*((c shr 16) and 0xFF)+183*((c shr 8) and 0xFF)+19*(c and 0xFF))/256
            hist[yv]++; i++
        }
        var sum=0.0; for (t in 0..255) sum += t*hist[t]
        var sumB=0.0; var wB=0.0; var maxVar=-1.0; var thr=127
        var t=0
        while (t<=255) {
            wB += hist[t]; if (wB==0.0) { t++; continue }
            val wF = w*h - wB; if (wF==0.0) break
            sumB += t*hist[t]
            val mB = sumB/wB; val mF = (sum - sumB)/wF
            val v = wB*wF*(mB-mF)*(mB-mF)
            if (v > maxVar) { maxVar=v; thr=t }
            t++
        }
        val bin = BooleanArray(w*h)
        i=0
        while (i<px.size) {
            val c=px[i]
            val yv = (54*((c shr 16) and 0xFF)+183*((c shr 8) and 0xFF)+19*(c and 0xFF))/256
            bin[i] = yv < thr
            i++
        }
        return BinOut(bin, w, h)
    }

    // --- OKLab utils ---
    private fun rgbToOkLab(r8: Int, g8: Int, b8: Int): FloatArray {
        fun srgbToLinear(c: Int): Double {
            val s = c/255.0
            return if (s <= 0.04045) s/12.92 else ((s + 0.055)/1.055).pow(2.4)
        }
        val r = srgbToLinear(r8); val g = srgbToLinear(g8); val b = srgbToLinear(b8)
        val l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
        val m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
        val s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
        val l_ = cbrt(l); val m_ = cbrt(m); val s_ = cbrt(s)
        val L = (0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_).toFloat()
        val A = (1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_).toFloat()
        val B = (0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_).toFloat()
        return floatArrayOf(L, A, B)
    }
    private fun labToArgb(L: Float, a: Float, b: Float): Int {
        val l = L + 0.3963377774f * a + 0.2158037573f * b
        val m = L - 0.1055613458f * a - 0.0638541728f * b
        val s = L - 0.0894841775f * a - 1.2914855480f * b
        val l3 = l*l*l; val m3 = m*m*m; val s3 = s*s*s
        var rLin = +4.0767416621f * l3 - 3.3077115913f * m3 + 0.2309699292f * s3
        var gLin = -1.2684380046f * l3 + 2.6097574011f * m3 - 0.3413193965f * s3
        var bLin = -0.0041960863f * l3 - 0.7034186147f * m3 + 1.7076147010f * s3
        fun gamma(xIn: Float): Float { val x=max(0f,xIn); return if (x<=0.0031308f) 12.92f*x else (1.055*x.toDouble().pow(1.0/2.4)-0.055).toFloat() }
        val r=(gamma(rLin).coerceIn(0f,1f)*255f+0.5f).toInt()
        val g=(gamma(gLin).coerceIn(0f,1f)*255f+0.5f).toInt()
        val b8=(gamma(bLin).coerceIn(0f,1f)*255f+0.5f).toInt()
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b8
    }

    // Удобства
    private data class TripleEdge(val edgeMin: Float, val confMax: Float, val dEMax: Float, val holesMin: Float?)
    private fun fmt(x: Float) = String.format(java.util.Locale.US, "%.3f", x)

    // ─────────────────────────────────────────────────────────────
    // Новые хелперы: пред‑очистка, EC‑law(TEXT), лимит 10×10, изоляты
    // ─────────────────────────────────────────────────────────────

    // Запрет пробегов длиной 1: одиночные «палки» в ряд/колонку переокрашиваем в ближайшее большинство
    private fun enforceMinRun2(px: IntArray, w: Int, h: Int) {
        // Горизонтальные «единички»
        var y = 0
        while (y < h) {
            var x = 1
            while (x < w - 1) {
                val i = y*w + x
                val c = px[i]
                if (px[i-1] == px[i+1] && px[i-1] != c) {
                    // проверим вертикальный контекст
                    val up = if (y > 0) px[(y-1)*w + x] else c
                    val dn = if (y+1 < h) px[(y+1)*w + x] else c
                    px[i] = if (up == px[i-1] || dn == px[i-1]) px[i-1] else px[i-1]
                    x += 2; continue
                }
                x++
            }
            y++
        }
        // Вертикальные «единички»
        var x = 0
        while (x < w) {
            var yy = 1
            while (yy < h - 1) {
                val i = yy*w + x
                val c = px[i]
                if (px[i-w] == px[i+w] && px[i-w] != c) {
                    val lf = if (x > 0) px[yy*w + x - 1] else c
                    val rt = if (x+1 < w) px[yy*w + x + 1] else c
                    px[i] = if (lf == px[i-w] || rt == px[i-w]) px[i-w] else px[i-w]
                    yy += 2; continue
                }
                yy++
            }
            x++
        }
    }

    // Подозрение на билинейное растяжение исходника:
    // сравниваем "down-then-up через BL" и "down-then-up через NN" — если BL ближе к src по PSNR, то src похож на BL‑растянутый.
    private fun suspectBilinearUpscale(src: Bitmap): Boolean {
        val w = src.width; val h = src.height
        val dw = max(1, w / 2); val dh = max(1, h / 2)
        val px = IntArray(w*h); src.getPixels(px,0,w,0,0,w,h)
        fun down(): IntArray {
            val out = IntArray(dw*dh)
            var y=0; while (y<dh) { var x=0; while (x<dw) {
                out[y*dw+x] = px[(y*2).coerceAtMost(h-1)*w + (x*2).coerceAtMost(w-1)]
                x++ }; y++ }
            return out
        }
        fun upNN(small: IntArray): IntArray {
            val out = IntArray(w*h)
            var y=0; while (y<h) { var x=0; while (x<w) {
                out[y*w+x] = small[(y/2).coerceAtMost(dh-1)*dw + (x/2).coerceAtMost(dw-1)]
                x++ }; y++ }
            return out
        }
        fun upBL(small: IntArray): IntArray {
            val out = IntArray(w*h)
            var y=0
            while (y<h) {
                val sy = y/2.0; val y0 = floor(sy).toInt().coerceIn(0, dh-1); val y1 = min(dh-1, y0+1); val ty = sy - y0
                var x=0
                while (x<w) {
                    val sx = x/2.0; val x0 = floor(sx).toInt().coerceIn(0, dw-1); val x1 = min(dw-1, x0+1); val tx = sx - x0
                    val c00 = small[y0*dw + x0]; val c01 = small[y0*dw + x1]
                    val c10 = small[y1*dw + x0]; val c11 = small[y1*dw + x1]
                    fun lerp(a:Int,b:Int,t:Double)= (a + (b-a)*t).toInt()
                    val r = lerp((c00 ushr 16) and 0xFF, (c01 ushr 16) and 0xFF, tx)
                    val g = lerp((c00 ushr 8)  and 0xFF, (c01 ushr 8)  and 0xFF, tx)
                    val b = lerp( c00 and 0xFF,           c01 and 0xFF,           tx)
                    val r2 = lerp((c10 ushr 16) and 0xFF, (c11 ushr 16) and 0xFF, tx)
                    val g2 = lerp((c10 ushr 8)  and 0xFF, (c11 ushr 8)  and 0xFF, tx)
                    val b2 = lerp( c10 and 0xFF,           c11 and 0xFF,           tx)
                    val rr = lerp(r, r2, ty); val gg = lerp(g, g2, ty); val bb = lerp(b, b2, ty)
                    out[y*w+x] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
                    x++
                }
                y++
            }
            return out
        }
        fun psnr(a: IntArray, b: IntArray): Double {
            var mse = 0.0; var i=0
            while (i<a.size) {
                val ar=(a[i] ushr 16) and 0xFF; val ag=(a[i] ushr 8) and 0xFF; val ab=a[i] and 0xFF
                val br=(b[i] ushr 16) and 0xFF; val bg=(b[i] ushr 8) and 0xFF; val bb=b[i] and 0xFF
                val dr=ar-br; val dg=ag-bg; val db=ab-bb
                mse += (dr*dr + dg*dg + db*db)/3.0; i++
            }
            mse /= a.size
            return if (mse <= 1e-9) 99.0 else 10.0 * kotlin.math.log10(255.0*255.0 / mse)
        }
        val small = down()
        val srcArr = px
        val nn = upNN(small)
        val bl = upBL(small)
        val gain = psnr(srcArr, bl) - psnr(srcArr, nn)
        return gain >= 1.5 // BL ощутимо ближе к исходнику → подозреваем BL‑растяжение
    }

    // Ресэмпл «block‑majority»: для каждой целевой клетки берём модальный цвет источника в покрывающем прямоугольнике
    private fun blockMajorityResample(src: Bitmap, W: Int, H: Int): Bitmap {
        val sw = src.width; val sh = src.height
        val inPx = IntArray(sw*sh); src.getPixels(inPx,0,sw,0,0,sw,sh)
        val outPx = IntArray(W*H)
        var y = 0
        while (y < H) {
            val sy0 = (y * sh) / H
            val sy1 = ((y + 1) * sh + H - 1) / H
            var x = 0
            while (x < W) {
                val sx0 = (x * sw) / W
                val sx1 = ((x + 1) * sw + W - 1) / W
                // аккуратный подсчёт моды цвета в блоке [sx0..sx1), [sy0..sy1)
                val hist = HashMap<Int, Int>(16)
                var yy = sy0
                while (yy < sy1) {
                    var xx = sx0
                    while (xx < sx1) {
                        val c = inPx[yy*sw + xx] or (0xFF shl 24)
                        hist[c] = (hist[c] ?: 0) + 1
                        xx++
                    }
                    yy++
                }
                val best = hist.entries.maxByOrNull { it.value }?.key ?: inPx[sy0*sw + sx0] or (0xFF shl 24)
                outPx[y*W + x] = best
                x++
            }
            y++
        }
        return Bitmap.createBitmap(outPx, W, H, Bitmap.Config.ARGB_8888)
    }

    // Лёгкая пред‑очистка "mode‑filter 3×3" в пределах ΔE≈2 (OKLab proxy)
    private fun precleanMode3x3WithinDE(src: Bitmap): Bitmap {
        val w = src.width; val h = src.height; val n = w*h
        val px = IntArray(n); src.getPixels(px,0,w,0,0,w,h)
        val out = IntArray(n)
        fun de2(c1:Int, c2:Int): Float {
            val r1=(c1 ushr 16) and 0xFF; val g1=(c1 ushr 8) and 0xFF; val b1=c1 and 0xFF
            val r2=(c2 ushr 16) and 0xFF; val g2=(c2 ushr 8) and 0xFF; val b2=c2 and 0xFF
            val a1=rgbToOkLab(r1,g1,b1); val a2=rgbToOkLab(r2,g2,b2)
            val dl=a1[0]-a2[0]; val da=a1[1]-a2[1]; val db=a1[2]-a2[2]
            return kotlin.math.sqrt(dl*dl+da*da+db*db)
        }
        val thr = 0.02f // ≈2 ΔE*
        var y=0
        while (y<h) {
            var x=0
            while (x<w) {
                val i=y*w+x
                var best=px[i]; var bestCnt=0
                var yy=max(0,y-1)
                while (yy<=min(h-1,y+1)) {
                    var xx=max(0,x-1)
                    while (xx<=min(w-1,x+1)) {
                        val c=px[yy*w+xx]
                        // считаем "таким же", если ΔE маленькая → голоса в пользу моды
                        if (de2(px[i], c) <= thr) {
                            // грубый подсчёт голосов по совпадениям цвета
                            var cnt=0
                            var y2=max(0,y-1)
                            while (y2<=min(h-1,y+1)) {
                                var x2=max(0,x-1)
                                while (x2<=min(w-1,x+1)) {
                                    if (de2(c, px[y2*w+x2]) <= thr) cnt++
                                    x2++
                                }
                                y2++
                            }
                            if (cnt>bestCnt) { best=c; bestCnt=cnt }
                        }
                        xx++
                    }
                    yy++
                }
                out[i]=best
                x++
            }
            y++
        }
        val dst=Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888)
        dst.setPixels(out,0,w,0,0,w,h)
        return dst
    }

    // EC‑law(TEXT): средний контраст текста к подложке в OKLab‑proxy
    // Возвращает (ΔL≈0..1, ΔE≈0..2)
    private fun textContrastStats(bmp: Bitmap): Pair<Float, Float> {
        val (bin,w,h)=otsuBinary(bmp) // тёмное → "текст" (без OCR)
        val px=IntArray(w*h); bmp.getPixels(px,0,w,0,0,w,h)
        var tL=0f; var tA=0f; var tB=0f; var tc=0
        var bL=0f; var bA=0f; var bB=0f; var bc=0
        var i=0
        while (i<px.size) {
            val c=px[i]; val lab=rgbToOkLab((c ushr 16) and 0xFF,(c ushr 8) and 0xFF,c and 0xFF)
            if (bin[i]) { tL+=lab[0]; tA+=lab[1]; tB+=lab[2]; tc++ } else { bL+=lab[0]; bA+=lab[1]; bB+=lab[2]; bc++ }
            i++
        }
        if (tc==0 || bc==0) return 0f to 0f
        val mTL=tL/tc; val mTA=tA/tc; val mTB=tB/tc
        val mBL=bL/bc; val mBA=bA/bc; val mBB=bB/bc
        val dL=kotlin.math.abs(mTL-mBL)
        val dA=mTA-mBA; val dB=mTB-mBB
        val dE=kotlin.math.sqrt(dL*dL + dA*dA + dB*dB)
        return dL to dE
    }

    // Лимит “цветов на 10×10”: (avgExcessNormalized, shareExceed)
    private fun blockColorsStats(px: IntArray, w: Int, h: Int, block: Int, limit: Int): Pair<Float, Float> {
        var blocks=0; var exceedBlocks=0; var sumExcess=0
        var y=0
        while (y<h) {
            var x=0
            while (x<w) {
                val x1=min(w, x+block); val y1=min(h, y+block)
                val set=HashSet<Int>(16)
                var yy=y; while (yy<y1) { var xx=x; while (xx<x1) { set += px[yy*w+xx]; xx++ }; yy++ }
                val k=set.size
                if (k>limit) { exceedBlocks++; sumExcess += (k-limit) }
                blocks++
                x+=block
            }
            y+=block
        }
        if (blocks==0) return 0f to 0f
        val avgExcess = (sumExcess.toFloat()/blocks.toFloat())
        val avgExcessNorm = (avgExcess / limit.toFloat()).coerceIn(0f,1f)
        val share = (exceedBlocks.toFloat()/blocks.toFloat()).coerceIn(0f,1f)
        return avgExcessNorm to share
    }

    // Срезать изолированные "одиночные" стежки (4‑соседство) к цвету большинства окрестности
    private fun cullIsolatesToMajority(px: IntArray, w: Int, h: Int) {
        val copy = px.copyOf()
        var y = 1
        while (y < h - 1) {
            var x = 1
            while (x < w - 1) {
                val i = y * w + x
                val c = copy[i]
                if (copy[i - 1] != c && copy[i + 1] != c && copy[i - w] != c && copy[i + w] != c) {
                    val n1 = copy[i - 1]
                    val n2 = copy[i + 1]
                    val n3 = copy[i - w]
                    val n4 = copy[i + w]
                    // простое большинство из 4-х
                    val a = n1
                    val b = n2
                    val d = n3
                    val e = n4
                    val repl = when {
                        a == b || a == d || a == e -> a
                        b == d || b == e -> b
                        d == e -> d
                        else -> a // фоллбэк
                    }
                    px[i] = repl
                }
                x++
            }
            y++
        }
    }

}
