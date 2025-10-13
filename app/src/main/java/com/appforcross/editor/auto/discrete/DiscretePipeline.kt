// app/src/main/java/com/appforcross/editor/auto/discrete/DiscretePipeline.kt
package com.appforcross.editor.auto.discrete

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.appforcross.core.metrics.DeltaE
import com.appforcross.core.palette.Swatch
import com.appforcross.editor.auto.detect.SmartSceneDetector
import java.util.Locale
import kotlin.math.*

// ───────────────────────────────────────────────────────────────────────────────
// DISCRETE V2.5 — Quant-once + ID downscale
// - ОДНА квантизация исходника к палитре ниток -> ID-карта (индексы 0..K-1).
// - Для каждого S: быстрый downscale ID-карты (MODE для целых коэффициентов,
//   drop‑schedule с thin‑guard + phase search для рациональных).
// - Ранний 2‑уровневый отбор по дешёвым метрикам (EdgeIoU/SSIM proxy, Confetti,
//   HolesRetained для TEXT). «Тяжёлые» (ΔE95, TinyLoss, Score) считаем только
//   для 1–2 лучших кандидатов.
// - Минимум цветов страхуется Color‑Floor, плюс безопасные локальные правки.
// - Возвращаем минимальный S, прошедший пороги, иначе — лучший по Score.
// ───────────────────────────────────────────────────────────────────────────────
object DiscretePipeline {

    // ── Публичные типы ─────────────────────────────────────────────────────────
    data class Config(
        val sizes: IntArray = intArrayOf(120, 160, 180, 200, 240, 300, 360, 400, 440),
        val targetScore: Float = 0.75f,
        val enableCelCollapse: Boolean = true,
        // Минимум цветов после всех упрощений; ≤0 — авто по гистограмме
        val minKeepColors: Int = -1
    )

    data class Metrics(
        val edgeSSIM: Float,
        val deltaE95: Float,
        val tinyDetailLoss: Float,
        val confettiRatio: Float,
        val holesRetained: Float?, // null если TEXT=false
        val score: Float,
        val passed: Boolean,
        val reasons: List<String>
    )

    data class Result(
        val image: Bitmap,                  // итоговое изображение (клетка=пиксель)
        val gridWidth: Int,
        val gridHeight: Int,
        val gridSizeS: Int,                 // S (ширина в «стежках»)
        val usedSwatches: List<Swatch>,
        val metrics: Metrics
    )

    // ── Внутренние структуры ───────────────────────────────────────────────────
    private data class TripleEdge(
        val edgeMin: Float,
        val confMax: Float,
        val dEMax: Float,
        val holesMin: Float?
    )

    // ── Публичный API ──────────────────────────────────────────────────────────
    fun run(
        source: Bitmap,
        palette: List<Swatch>,
        toggles: SmartSceneDetector.Toggles,
        cfg: Config = Config()
    ): Result {
        require(source.width > 0 && source.height > 0) { "Empty bitmap" }
        require(palette.isNotEmpty()) { "Palette is empty" }

        val tAll = SystemClock.elapsedRealtimeNanos()

        // Палитра в OKLab (один раз)
        val swL = FloatArray(palette.size)
        val swA = FloatArray(palette.size)
        val swB = FloatArray(palette.size)
        for (i in palette.indices) {
            val c = palette[i].argb
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            val lab = rgbToOkLab(r, g, b)
            swL[i] = lab[0]; swA[i] = lab[1]; swB[i] = lab[2]
        }

        // ── 1) Quant‑once: исходник -> ID‑карта палитры ───────────────────────
        val tQ = SystemClock.elapsedRealtimeNanos()
        val (idsFull, usedFull) = quantizeIdsOnce(source, swL, swA, swB)
        Log.i("DiscretePipeline", "QuantOnce: ids=${idsFull.size} used=${usedFull.size} pal=${palette.size} → ${(SystemClock.elapsedRealtimeNanos()-tQ)/1e6} ms")

        // ── 2) Кандидаты S (учёт PIXEL_GRID) ──────────────────────────────────
        val gridInf = if (toggles.pixelGrid) inferBasePixelPitchG(source) else null
        val candidateSizes: IntArray = cfg.sizes
            .filter { s ->
                if (!toggles.pixelGrid) true
                else if (gridInf != null) isIntegerScaleForG(s, source.width, gridInf.g)
                else {
                    val ratio = source.width.toFloat() / s
                    abs(ratio - round(ratio)) <= 0.05f
                }
            }
            .distinct().sorted().toIntArray()
        require(candidateSizes.isNotEmpty()) { "No candidate S after filtering" }

        // Профили порогов
        val prof: TripleEdge = when {
            toggles.text -> TripleEdge(edgeMin = 0.86f, confMax = 0.030f, dEMax = 12f, holesMin = 0.80f)
            toggles.aa   -> TripleEdge(edgeMin = 0.84f, confMax = 0.035f, dEMax = 15f, holesMin = null)
            else         -> TripleEdge(edgeMin = 0.85f, confMax = 0.035f, dEMax = 14f, holesMin = null)
        }

        // Минимум цветов на S (авто, если не задано)
        fun estimateDominantColors12(bmp: Bitmap, cover: Float = 0.98f): Int {
            val w = bmp.width; val h = bmp.height
            val n = w * h
            val px = IntArray(n); bmp.getPixels(px, 0, w, 0, 0, w, h)
            val hist = IntArray(4096)
            var i = 0
            while (i < n) {
                val c = px[i]
                val key = (((c ushr 16) and 0xFF) ushr 4 shl 8) or
                        (((c ushr 8 ) and 0xFF) ushr 4 shl 4) or
                        (( c          and 0xFF) ushr 4)
                hist[key]++; i++
            }
            val nz = hist.filter { it > 0 }.sortedByDescending { it }
            if (nz.isEmpty()) return 2
            var sum = 0; var k = 0; val need = (n * cover).toInt()
            while (k < nz.size && sum < need) { sum += nz[k]; k++ }
            return k.coerceIn(2, 8)
        }

        // ── 3) Дешёвый отбор по метрикам на ID‑карте (уровень A) ───────────────
        data class CheapCand(
            val S: Int,
            val W: Int,
            val H: Int,
            val ids: IntArray,
            val edge: Float,
            val confetti: Float,
            val holes: Float?,  // только если TEXT
            val passA: Boolean
        )

        fun integerPenalty(S: Int): Float {
            if (!toggles.pixelGrid) return 0f
            val ratio = source.width.toFloat() / S
            val frac = abs(ratio - round(ratio))
            return if (frac <= 0.02f) 0f else (frac * 2f).coerceAtMost(0.25f)
        }

        val cheap = ArrayList<CheapCand>(candidateSizes.size)
        for (S in candidateSizes) {
            val H = max(1, (source.height * (S.toFloat() / source.width)).roundToInt())

            // BL‑эталон только для краёв (дёшево), больших матем. расходов нет
            val ref = Bitmap.createScaledBitmap(source, S, H, true)
            val refEdges = edgeMap(ref)

            // ↓ быстрый даунскейл ID‑карты с phase‑search (до 4 фаз по осям)
            val idsS = downscaleIdsSmart(idsFull, source.width, source.height, S, H)

            // Лёгкие локальные компенсации
            enforceMinRun2IDs(idsS, S, H)  // убираем 1‑длины
            cullIsolatesToMajorityIDs(idsS, S, H)

            // Ранние метрики (дёшево)
            val qEdges = edgeMapFromIds(idsS, S, H)
            val edge = edgeSSIM(refEdges, qEdges)            // SSIM proxy на бинаре
            val conf = confettiRatioIds(idsS, S, H)
            val holes = if (toggles.text) {
                // один раз пересобрать картинку для Otsu — это дёшево
                val tmpBmp = idsToBitmap(idsS, S, H, palette)
                holesRetained(ref, tmpBmp)
            } else null

            val passA = edge >= prof.edgeMin && conf <= prof.confMax && (prof.holesMin == null || (holes ?: 1f) >= prof.holesMin)
            cheap += CheapCand(S, S, H, idsS, edge, conf, holes, passA)
            Log.i("DiscretePipeline", "A: S=$S edge=${fmt(edge)} conf=${fmt(conf)} holes=${fmt(holes ?: -1f)} pass=$passA")
        }

        // Минимальный S, прошедший уровень A
        val firstPass = cheap.firstOrNull { it.passA }

        // ── 4) Точная оценка только для 1–2 лучших (уровень B) ────────────────
        val finals = ArrayList<Result>(2)
        val candidatesForB = buildList {
            if (firstPass != null) {
                // берём его и соседний побольше (на случай разрыва)
                add(firstPass)
                val idx = cheap.indexOf(firstPass)
                if (idx + 1 in cheap.indices) add(cheap[idx + 1])
            } else {
                // никто не прошёл — возьмём 1–2 лучших по edge/confetti
                add(cheap.maxByOrNull { it.edge - 0.35f * it.confetti } ?: cheap.last())
                val alt = cheap.sortedByDescending { it.edge - 0.35f * it.confetti }.getOrNull(1)
                if (alt != null) add(alt)
            }
        }.distinctBy { it.S }.take(2)

        val tB = SystemClock.elapsedRealtimeNanos()

        for (c in candidatesForB) {
            val S = c.S; val H = c.H
            val minKeepAuto = if (toggles.text) 2 else estimateDominantColors12(Bitmap.createScaledBitmap(source, S, H, true))
            val minKeep = if (cfg.minKeepColors > 0) cfg.minKeepColors else minKeepAuto

            // Color‑floor: если после упрощений цветов слишком мало — fallback на «блочное большинство»
            val uniq = uniqueIdsCount(c.ids)
            val idsFinal = if (uniq >= minKeep) {
                c.ids
            } else {
                Log.w("DiscretePipeline", "ColorFloor: uniq=$uniq < minKeep=$minKeep on S=$S → block‑majority fallback")
                downscaleIdsBlockMajority(idsFull, source.width, source.height, S, H)
                    .also { enforceMinRun2IDs(it, S, H); cullIsolatesToMajorityIDs(it, S, H) }
            }

            // Финальный растровый вид + used set
            val bmp = idsToBitmap(idsFinal, S, H, palette)
            val usedIdx = usedIndicesFromIds(idsFinal)

            // Метрики «дорогие»
            val ref = Bitmap.createScaledBitmap(source, S, H, true)
            val edge = edgeSSIM(edgeMap(ref), edgeMap(bmp))
            val dE = DeltaE.de2000Perc95(ref, bmp)
            val tiny = tinyDetailLoss(edgeMap(ref), edgeMap(bmp))
            val conf = confettiRatioIds(idsFinal, S, H)
            val holes = if (toggles.text) holesRetained(ref, bmp) else null
            val (blkExcess, blkShare) =
                blockColorsStatsARGB(bmp, block = 10, limit = if (toggles.text) 3 else 4)

            var score = scoreAggregate(
                edge = edge,
                dE = dE,
                confetti = conf,
                tiny = tiny,
                holes = holes,
                toggles = toggles,
                blockPenalty = blkShare.coerceIn(0f, 1f) * (blkExcess.coerceIn(0f, 1f))
            ) - integerPenalty(S)

            if (toggles.text) {
                val (dL, dEec) = textContrastStats(bmp)
                val sL = (dL / 0.25f).coerceIn(0f, 1f)
                val sE = (dEec / 0.18f).coerceIn(0f, 1f)
                score += 0.10f * (0.5f * sL + 0.5f * sE)
            }

            val reasons = mutableListOf<String>()
            if (edge < prof.edgeMin) reasons += "EdgeSSIM<${fmt(prof.edgeMin)}"
            if (conf > prof.confMax) reasons += "Confetti>${fmt(prof.confMax)}"
            if (dE > prof.dEMax) reasons += "ΔE95>${fmt(prof.dEMax)}"
            if (prof.holesMin != null && (holes ?: 0f) < prof.holesMin) reasons += "Holes<${fmt(prof.holesMin)}"

            val passed = reasons.isEmpty() && score >= cfg.targetScore
            val usedSwatches = usedIdx.map { palette[it] }

            Log.i(
                "DiscretePipeline",
                "FINAL S=$S usedColors=${usedIdx.size} score=${fmt(score)} pass=$passed edge=${fmt(edge)} dE=${fmt(dE)} conf=${fmt(conf)}"
            )

            finals += Result(
                image = bmp,
                gridWidth = S,
                gridHeight = H,
                gridSizeS = S,
                usedSwatches = usedSwatches,
                metrics = Metrics(
                    edgeSSIM = edge,
                    deltaE95 = dE,
                    tinyDetailLoss = tiny,
                    confettiRatio = conf,
                    holesRetained = holes,
                    score = score,
                    passed = passed,
                    reasons = reasons
                )
            )
        }

        // Выбор ответа
        val best = finals.firstOrNull { it.metrics.passed }
            ?: finals.maxByOrNull { it.metrics.score }
            ?: run {
                // На всякий случай — если «B» не вычислился, возьмём лучший из A
                val c = cheap.maxByOrNull { it.edge - 0.35f * it.confetti }!!
                Result(
                    image = idsToBitmap(c.ids, c.W, c.H, palette),
                    gridWidth = c.W,
                    gridHeight = c.H,
                    gridSizeS = c.S,
                    usedSwatches = usedIndicesFromIds(c.ids).map { palette[it] },
                    metrics = Metrics(
                        edgeSSIM = c.edge,
                        deltaE95 = 99f,
                        tinyDetailLoss = 1f,
                        confettiRatio = c.confetti,
                        holesRetained = c.holes,
                        score = 0f,
                        passed = false,
                        reasons = listOf("fallback")
                    )
                )
            }

        Log.i(
            "DiscretePipeline",
            "TOTAL → ${(SystemClock.elapsedRealtimeNanos() - tAll) / 1e6} ms; chosen S=${best.gridSizeS}, used=${best.usedSwatches.size}"
        )
        return best
    }

    // ── Quant‑once: исходник -> ID‑карта палитры ───────────────────────────────
    private fun quantizeIdsOnce(
        src: Bitmap,
        swL: FloatArray, swA: FloatArray, swB: FloatArray
    ): Pair<IntArray, Set<Int>> {
        val w = src.width; val h = src.height
        val n = w * h
        val px = IntArray(n)
        src.getPixels(px, 0, w, 0, 0, w, h)
        val ids = IntArray(n)
        val used = HashSet<Int>(16)
        var i = 0
        while (i < n) {
            val c = px[i]
            val r = (c ushr 16) and 0xFF
            val g = (c ushr 8) and 0xFF
            val b = c and 0xFF
            val lab = rgbToOkLab(r, g, b)
            var best = 0; var bestD = Float.POSITIVE_INFINITY; var k = 0
            while (k < swL.size) {
                val dl = lab[0] - swL[k]; val da = lab[1] - swA[k]; val db = lab[2] - swB[k]
                val d2 = dl*dl + da*da + db*db
                if (d2 < bestD) { bestD = d2; best = k }
                k++
            }
            ids[i] = best
            used += best
            i++
        }
        return ids to used
    }

    // ── Downscale ID‑карты: MODE (целые) / Drop‑Schedule + phase search (рациональные)
    private fun downscaleIdsSmart(ids: IntArray, w: Int, h: Int, W: Int, H: Int): IntArray {
        if (w % W == 0 && h % H == 0) {
            val bx = w / W; val by = h / H
            return downscaleIdsBlockMode(ids, w, h, bx, by)
        }
        // Фаза-поиск до 4*4
        fun gcd(a0: Int, b0: Int): Int { var a=a0; var b=b0; while (b!=0){ val t=a%b; a=b; b=t }; return abs(a) }
        val gX = gcd(W, w); val qX = w / gX; val px = min(4, qX)
        val phasesX = IntArray(px) { (it * qX) / max(1, px) }
        val gY = gcd(H, h); val qY = h / gY; val py = min(4, qY)
        val phasesY = IntArray(py) { (it * qY) / max(1, py) }

        val thinCols = thinColumnsMaskIds(ids, w, h)
        val thinRows = thinRowsMaskIds(ids, w, h)

        var best: IntArray? = null
        var bestConf = Float.POSITIVE_INFINITY

        for (phY in phasesY) {
            val keepY = scheduleKeepWithThinGuard(H, h, phY, thinRows)
            for (phX in phasesX) {
                val keepX = scheduleKeepWithThinGuard(W, w, phX, thinCols)
                val out = downscaleIdsByKeeps(ids, w, h, keepX, keepY)
                val conf = confettiRatioIds(out, W, H)
                if (conf < bestConf) { bestConf = conf; best = out }
            }
        }
        return best!!
    }

    private fun downscaleIdsBlockMode(src: IntArray, w: Int, h: Int, bx: Int, by: Int): IntArray {
        val W = w / bx; val H = h / by
        val out = IntArray(W * H)
        var ty = 0
        while (ty < H) {
            val y0 = ty * by
            var tx = 0
            while (tx < W) {
                val x0 = tx * bx
                // Модальный ID цвета в блоке bx×by (очень дёшево)
                var bestId = -1; var bestCnt = -1
                val hist = HashMap<Int, Int>(16)
                var yy = 0
                while (yy < by) {
                    val row = (y0 + yy) * w + x0
                    var xx = 0
                    while (xx < bx) {
                        val id = src[row + xx]
                        val c = (hist[id] ?: 0) + 1
                        hist[id] = c
                        if (c > bestCnt) { bestCnt = c; bestId = id }
                        xx++
                    }
                    yy++
                }
                out[ty * W + tx] = bestId
                tx++
            }
            ty++
        }
        return out
    }

    private fun downscaleIdsBlockMajority(src: IntArray, w: Int, h: Int, W: Int, H: Int): IntArray {
        val out = IntArray(W * H)
        var y = 0
        while (y < H) {
            val sy0 = (y * h) / H
            val sy1 = ((y + 1) * h + H - 1) / H
            var x = 0
            while (x < W) {
                val sx0 = (x * w) / W
                val sx1 = ((x + 1) * w + W - 1) / W
                var bestId = -1; var bestCnt = -1
                val hist = HashMap<Int, Int>(16)
                var yy = sy0
                while (yy < sy1) {
                    var xx = sx0
                    while (xx < sx1) {
                        val id = src[yy * w + xx]
                        val c = (hist[id] ?: 0) + 1
                        hist[id] = c
                        if (c > bestCnt) { bestCnt = c; bestId = id }
                        xx++
                    }
                    yy++
                }
                out[y * W + x] = bestId
                x++
            }
            y++
        }
        return out
    }

    private fun downscaleIdsByKeeps(src: IntArray, w: Int, h: Int, keepX: IntArray, keepY: IntArray): IntArray {
        val W = keepX.size; val H = keepY.size
        val out = IntArray(W * H)
        var ty = 0
        while (ty < H) {
            val sy = keepY[ty]
            var tx = 0
            while (tx < W) {
                val sx = keepX[tx]
                // Небольшое «окно большинства» 3×3 вокруг выбранной точки — устойчиво к фазе
                var bestId = -1; var bestCnt = -1
                val x0 = max(0, sx - 1); val x1 = min(w - 1, sx + 1)
                val y0 = max(0, sy - 1); val y1 = min(h - 1, sy + 1)
                val hist = HashMap<Int, Int>(9)
                var yy = y0
                while (yy <= y1) {
                    var xx = x0
                    while (xx <= x1) {
                        val id = src[yy * w + xx]
                        val c = (hist[id] ?: 0) + 1
                        hist[id] = c
                        if (c > bestCnt) { bestCnt = c; bestId = id }
                        xx++
                    }
                    yy++
                }
                out[ty * W + tx] = bestId
                tx++
            }
            ty++
        }
        return out
    }

    private fun scheduleKeepWithThinGuard(target: Int, size: Int, phase: Int, thin: BooleanArray): IntArray {
        val keep = BooleanArray(size)
        var acc = phase.coerceIn(0, max(0, size - 1))
        var outCount = 0
        val p = target; val q = size
        for (i in 0 until size) {
            acc += p
            val take = if (acc >= q) { acc -= q; true } else false
            if (take) { keep[i] = true; outCount++ }
        }
        // thin‑guard: если тонкий пропущен — меняем местами с ближайшим «не‑тонким» взятым
        var i = 0
        while (i < size) {
            if (!keep[i] && thin[i]) {
                var jL = i - 1; var jR = i + 1; var swapped = false
                while (jL >= 0 || jR < size) {
                    if (jL >= 0 && keep[jL] && !thin[jL]) { keep[jL] = false; keep[i] = true; swapped = true; break }
                    if (jR < size && keep[jR] && !thin[jR]) { keep[jR] = false; keep[i] = true; swapped = true; break }
                    jL--; jR++
                }
                if (!swapped) keep[i] = true
            }
            i++
        }
        val out = IntArray(target)
        var k = 0; var idx = 0
        while (idx < size && k < target) { if (keep[idx]) out[k++] = idx; idx++ }
        while (k < target) { out[k] = min(size - 1, out[max(0, k - 1)] + 1); k++ }
        return out
    }

    private fun thinColumnsMaskIds(ids: IntArray, w: Int, h: Int): BooleanArray {
        val thin = BooleanArray(w)
        var y = 0
        while (y < h) {
            val row = y * w
            var x = 0
            while (x < w) {
                val c = ids[row + x]
                val l = if (x > 0) ids[row + x - 1] else c
                val r = if (x < w - 1) ids[row + x + 1] else c
                if (c != l && c != r) thin[x] = true
                x++
            }
            y++
        }
        return thin
    }

    private fun thinRowsMaskIds(ids: IntArray, w: Int, h: Int): BooleanArray {
        val thin = BooleanArray(h)
        var x = 0
        while (x < w) {
            var y = 0
            while (y < h) {
                val c = ids[y * w + x]
                val t = if (y > 0) ids[(y - 1) * w + x] else c
                val b = if (y < h - 1) ids[(y + 1) * w + x] else c
                if (c != t && c != b) thin[y] = true
                y++
            }
            x++
        }
        return thin
    }

    // ── Лёгкие правки на ID‑карте ─────────────────────────────────────────────
    private fun enforceMinRun2IDs(ids: IntArray, w: Int, h: Int) {
        // Горизонтальные «единицы»
        var y = 0
        while (y < h) {
            var x = 1
            while (x < w - 1) {
                val i = y * w + x
                val c = ids[i]
                if (ids[i - 1] == ids[i + 1] && ids[i - 1] != c) {
                    ids[i] = ids[i - 1]
                    x += 2; continue
                }
                x++
            }
            y++
        }
        // Вертикальные «единицы»
        var x = 0
        while (x < w) {
            var yy = 1
            while (yy < h - 1) {
                val i = yy * w + x
                val c = ids[i]
                if (ids[i - w] == ids[i + w] && ids[i - w] != c) {
                    ids[i] = ids[i - w]
                    yy += 2; continue
                }
                yy++
            }
            x++
        }
    }

    private fun cullIsolatesToMajorityIDs(ids: IntArray, w: Int, h: Int) {
        val src = ids.copyOf()
        var y = 1
        while (y < h - 1) {
            var x = 1
            while (x < w - 1) {
                val i = y * w + x
                val c = src[i]
                if (src[i - 1] != c && src[i + 1] != c && src[i - w] != c && src[i + w] != c) {
                    val a = src[i - 1]; val b = src[i + 1]; val d = src[i - w]; val e = src[i + w]
                    val repl = when {
                        a == b || a == d || a == e -> a
                        b == d || b == e -> b
                        d == e -> d
                        else -> a
                    }
                    ids[i] = repl
                }
                x++
            }
            y++
        }
    }

    // ── Метрики на ID‑карте ───────────────────────────────────────────────────
    private fun confettiRatioIds(ids: IntArray, w: Int, h: Int): Float {
        var bad = 0; var all = 0
        var y = 1
        while (y < h - 1) {
            var x = 1
            while (x < w - 1) {
                val i = y * w + x
                val c = ids[i]
                all++
                if (ids[i - 1] != c && ids[i + 1] != c && ids[i - w] != c && ids[i + w] != c) bad++
                x++
            }
            y++
        }
        return if (all == 0) 0f else (bad.toFloat() / all)
    }

    private fun edgeMapFromIds(ids: IntArray, w: Int, h: Int): BooleanArray {
        val out = BooleanArray(w * h)
        var y = 1
        while (y < h - 1) {
            var x = 1
            while (x < w - 1) {
                val i = y * w + x
                val c = ids[i]
                out[i] = (ids[i - 1] != c || ids[i + 1] != c || ids[i - w] != c || ids[i + w] != c)
                x++
            }
            y++
        }
        return out
    }

    private fun idsToBitmap(ids: IntArray, w: Int, h: Int, palette: List<Swatch>): Bitmap {
        val out = IntArray(w * h)
        var i = 0
        while (i < out.size) {
            val idx = ids[i].coerceIn(0, palette.lastIndex)
            out[i] = (0xFF shl 24) or (palette[idx].argb and 0x00FFFFFF)
            i++
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun usedIndicesFromIds(ids: IntArray): Set<Int> {
        val s = HashSet<Int>(16)
        var i = 0
        while (i < ids.size) { s += ids[i]; i++ }
        return s
    }

    private fun uniqueIdsCount(ids: IntArray): Int {
        val s = HashSet<Int>(ids.size / 4 + 1)
        for (v in ids) s += v
        return s.size
    }

    // ── Метрики и утилиты ARGB (взято/адаптировано из прежней версии) ─────────
    private fun edgeSSIM(a: BooleanArray, b: BooleanArray): Float {
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
        val C1 = 0.01 * 0.01
        val C2 = 0.03 * 0.03
        val num = (2*muA*muB + C1) * (2*cab + C2)
        val den = (muA*muA + muB*muB + C1) * (va + vb + C2)
        return (num / den).toFloat().coerceIn(0f, 1f)
    }

    private fun tinyDetailLoss(refEdges: BooleanArray, qEdges: BooleanArray): Float {
        val w = sqrt(refEdges.size.toFloat()).roundToInt()
        val h = if (w == 0) 0 else refEdges.size / w
        fun ccCount(edges: BooleanArray, tinyMax: Int): Int {
            val seen = BooleanArray(edges.size)
            val qx = IntArray(edges.size); val qy = IntArray(edges.size)
            var qs = 0; var qe = 0; var cnt = 0
            var i = 0
            while (i < edges.size) {
                if (seen[i] || !edges[i]) { i++; continue }
                var area = 0
                qs = 0; qe = 0
                qx[qe] = i % w; qy[qe] = i / w; qe++
                seen[i] = true
                while (qs < qe) {
                    val x = qx[qs]; val y = qy[qs]; qs++
                    area++
                    val nbs = intArrayOf(y*w + x - 1, y*w + x + 1, (y - 1)*w + x, (y + 1)*w + x)
                    for (nb in nbs) {
                        if (nb < 0 || nb >= edges.size || seen[nb]) continue
                        val ny = nb / w; val nx = nb - ny * w
                        if (abs(nx - x) + abs(ny - y) != 1) continue
                        if (edges[nb]) { seen[nb] = true; qx[qe] = nx; qy[qe] = ny; qe++ }
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
        return (lost.toFloat() / tinyRef).coerceIn(0f, 1f)
    }

    private fun edgeMap(bmp: Bitmap): BooleanArray {
        val w = bmp.width; val h = bmp.height
        val Y = IntArray(w * h)
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            Y[i] = (54 * ((c ushr 16) and 0xFF) + 183 * ((c ushr 8) and 0xFF) + 19 * (c and 0xFF)) / 256
            i++
        }
        val mag = IntArray(w * h)
        var y = 1
        while (y < h - 1) {
            var x = 1
            while (x < w - 1) {
                val p = y * w + x
                val gx = -Y[p - w - 1] - 2 * Y[p - 1] - Y[p + w - 1] + Y[p - w + 1] + 2 * Y[p + 1] + Y[p + w + 1]
                val gy =  Y[p - w - 1] + 2 * Y[p - w] + Y[p - w + 1] - Y[p + w - 1] - 2 * Y[p + w] - Y[p + w + 1]
                mag[p] = abs(gx) + abs(gy)
                x++
            }
            y++
        }
        val nz = mag.filter { it > 0 }.sorted()
        val thr = if (nz.isEmpty()) 255 else nz[(nz.size * 0.90f).toInt().coerceAtMost(nz.size - 1)]
        val out = BooleanArray(w * h) { idx -> mag[idx] >= thr }
        return out
    }

    private fun holesRetained(ref: Bitmap, q: Bitmap): Float {
        val (binA, w, h) = otsuBinary(ref)
        val (binB, _, _) = otsuBinary(q)
        fun holes(bin: BooleanArray): Int {
            val seen = BooleanArray(bin.size)
            val qx = IntArray(bin.size); val qy = IntArray(bin.size)
            var qs = 0; var qe = 0; var holes = 0
            var i = 0
            while (i < bin.size) {
                if (seen[i] || bin[i]) { i++; continue }
                val x0 = i % w; val y0 = i / w
                qs = 0; qe = 0
                qx[qe] = x0; qy[qe] = y0; qe++
                seen[i] = true
                var touches = false
                while (qs < qe) {
                    val x = qx[qs]; val y = qy[qs]; qs++
                    if (x == 0 || x == w - 1 || y == 0 || y == h - 1) touches = true
                    val nbs = arrayOf(x - 1 to y, x + 1 to y, x to y - 1, x to y + 1)
                    for ((nx, ny) in nbs) {
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
                        val j = ny * w + nx
                        if (seen[j] || bin[j]) continue
                        seen[j] = true; qx[qe] = nx; qy[qe] = ny; qe++
                    }
                }
                if (!touches) holes++
                i++
            }
            return holes
        }
        val a = holes(binA); val b = holes(binB)
        return if (a == 0) 1f else (b.toFloat() / a.toFloat()).coerceIn(0f, 1f)
    }

    private fun blockColorsStatsARGB(bmp: Bitmap, block: Int, limit: Int): Pair<Float, Float> {
        val w = bmp.width; val h = bmp.height
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        var blocks = 0; var exceed = 0; var sumExcess = 0
        var y = 0
        while (y < h) {
            var x = 0
            while (x < w) {
                val x1 = min(w, x + block); val y1 = min(h, y + block)
                val set = HashSet<Int>(16)
                var yy = y
                while (yy < y1) {
                    var xx = x
                    while (xx < x1) { set += px[yy * w + xx]; xx++ }
                    yy++
                }
                val k = set.size
                if (k > limit) { exceed++; sumExcess += (k - limit) }
                blocks++
                x += block
            }
            y += block
        }
        if (blocks == 0) return 0f to 0f
        val avgExcess = (sumExcess.toFloat() / blocks.toFloat())
        val avgExcessNorm = (avgExcess / limit.toFloat()).coerceIn(0f, 1f)
        val share = (exceed.toFloat() / blocks.toFloat()).coerceIn(0f, 1f)
        return avgExcessNorm to share
    }

    private data class BinOut(val bin: BooleanArray, val w: Int, val h: Int)
    private fun otsuBinary(b: Bitmap): BinOut {
        val w = b.width; val h = b.height
        val hist = IntArray(256)
        val px = IntArray(w * h); b.getPixels(px, 0, w, 0, 0, w, h)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            val yv = (54 * ((c ushr 16) and 0xFF) + 183 * ((c ushr 8) and 0xFF) + 19 * (c and 0xFF)) / 256
            hist[yv]++; i++
        }
        var sum = 0.0; for (t in 0..255) sum += t * hist[t]
        var sumB = 0.0; var wB = 0.0; var maxVar = -1.0; var thr = 127
        var t = 0
        while (t <= 255) {
            wB += hist[t]; if (wB == 0.0) { t++; continue }
            val wF = w * h - wB; if (wF == 0.0) break
            sumB += t * hist[t]
            val mB = sumB / wB; val mF = (sum - sumB) / wF
            val v = wB * wF * (mB - mF) * (mB - mF)
            if (v > maxVar) { maxVar = v; thr = t }
            t++
        }
        val bin = BooleanArray(w * h)
        i = 0
        while (i < px.size) {
            val c = px[i]
            val yv = (54 * ((c ushr 16) and 0xFF) + 183 * ((c ushr 8) and 0xFF) + 19 * (c and 0xFF)) / 256
            bin[i] = yv < thr
            i++
        }
        return BinOut(bin, w, h)
    }

    private fun textContrastStats(bmp: Bitmap): Pair<Float, Float> {
        val (bin, w, h) = otsuBinary(bmp)
        val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
        var tL = 0f; var tA = 0f; var tB = 0f; var tc = 0
        var bL = 0f; var bA = 0f; var bB = 0f; var bc = 0
        var i = 0
        while (i < px.size) {
            val c = px[i]
            val lab = rgbToOkLab((c ushr 16) and 0xFF, (c ushr 8) and 0xFF, c and 0xFF)
            if (bin[i]) { tL += lab[0]; tA += lab[1]; tB += lab[2]; tc++ }
            else { bL += lab[0]; bA += lab[1]; bB += lab[2]; bc++ }
            i++
        }
        if (tc == 0 || bc == 0) return 0f to 0f
        val mTL = tL / tc; val mTA = tA / tc; val mTB = tB / tc
        val mBL = bL / bc; val mBA = bA / bc; val mBB = bB / bc
        val dL = abs(mTL - mBL)
        val dA = mTA - mBA; val dB = mTB - mBB
        val dE = sqrt(dL * dL + dA * dA + dB * dB)
        return dL to dE
    }

    private fun scoreAggregate(
        edge: Float,
        dE: Float,
        confetti: Float,
        tiny: Float,
        holes: Float?,
        toggles: SmartSceneDetector.Toggles,
        blockPenalty: Float
    ): Float {
        fun dENorm(x: Float): Float = (1f - (x / 15f)).coerceIn(0f, 1f)
        val base =
            0.42f * edge +
                    0.22f * dENorm(dE) +
                    0.18f * (1f - confetti) +
                    0.18f * (1f - tiny) -
                    0.16f * blockPenalty
        return if (toggles.text) base * 0.8f + 0.2f * (holes ?: 0f) else base
    }

    // ── Детект периода «логического пикселя» и integer‑scale фильтр ───────────
    private data class GridInference(val g: Int, val score: Float)
    private fun inferBasePixelPitchG(src: Bitmap, maxG: Int = 12): GridInference? {
        val w = src.width; val h = src.height
        if (w < 24 || h < 24) return null
        val n = w * h
        val px = IntArray(n); src.getPixels(px, 0, w, 0, 0, w, h)
        val colEnergy = FloatArray(w)
        var y = 1
        while (y < h - 1) {
            var x = 1
            while (x < w - 1) {
                val i = y * w + x
                fun lum(c: Int): Int {
                    val r = (c shr 16) and 0xFF; val g = (c shr 8) and 0xFF; val b = c and 0xFF
                    return (r * 299 + g * 587 + b * 114 + 500) / 1000
                }
                val gx = lum(px[i + 1]) - lum(px[i - 1])
                val gy = lum(px[i + w]) - lum(px[i - w])
                val g2 = (gx * gx + gy * gy).toFloat()
                colEnergy[x] += g2
                x++
            }
            y++
        }
        val peaks = ArrayList<Int>()
        for (x in 2 until w - 2) {
            val v = colEnergy[x]
            if (v > colEnergy[x - 1] && v > colEnergy[x + 1] && v > 0.25f * (colEnergy.maxOrNull() ?: 0f)) {
                peaks += x
            }
        }
        if (peaks.size < 4) return null
        val hist = IntArray(maxG + 1)
        for (i in 1 until peaks.size) {
            val d = peaks[i] - peaks[i - 1]
            if (d in 2..maxG) hist[d]++
        }
        var bestG = 0; var bestC = 0
        for (g in 2..maxG) if (hist[g] > bestC) { bestC = hist[g]; bestG = g }
        if (bestG == 0) return null
        val score = bestC / max(1f, peaks.size - 1f)
        return GridInference(bestG, score)
    }

    private fun isIntegerScaleForG(S: Int, srcW: Int, g: Int, tol: Float = 0.05f): Boolean {
        val base = srcW.toFloat() / g.toFloat()
        if (base <= 0f) return false
        val k = S / base
        return abs(k - round(k)) <= tol
    }

    // ── OKLab utils ────────────────────────────────────────────────────────────
    private fun rgbToOkLab(r8: Int, g8: Int, b8: Int): FloatArray {
        fun srgbToLinear(c: Int): Double {
            val s = c / 255.0
            return if (s <= 0.04045) s / 12.92 else ((s + 0.055) / 1.055).pow(2.4)
        }
        val r = srgbToLinear(r8); val g = srgbToLinear(g8); val b = srgbToLinear(b8)
        val l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        val m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        val s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        val l_ = cbrt(l); val m_ = cbrt(m); val s_ = cbrt(s)
        val L = (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_).toFloat()
        val A = (1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_).toFloat()
        val B = (0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_).toFloat()
        return floatArrayOf(L, A, B)
    }

    private fun fmt(x: Float) = String.format(Locale.US, "%.3f", x)
}
