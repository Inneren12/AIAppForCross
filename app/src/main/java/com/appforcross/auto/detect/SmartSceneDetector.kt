package com.appforcross.editor.auto.detect

import android.graphics.Bitmap
import android.util.Log
import kotlin.math.*

// SmartSceneDetector (Android, :core)
// 1) Грубое решение: DISCRETE ↔ PHOTO
// 2) Для DISCRETE: три тумблера TEXT / PIXEL_GRID / AA

object SmartSceneDetector {

    // ---------------- API ----------------

    enum class Mode { DISCRETE, PHOTO }

    data class Toggles(
        val text: Boolean,
        val pixelGrid: Boolean,
        val aa: Boolean
    )

    data class Features(
        val top2Cover: Float,     // покрытие Top2 12‑бит корзин
        val top8Cover: Float,     // покрытие Top8 12‑бит корзин
        val c16: Int,             // ~кол-во «укрупнённых» центров OKLab
        val rarePct: Float,       // доля редких корзин
        val backgroundPct: Float, // грубая доля «фона» по L*

        val gradScore: Float,     // средняя сила градиента
        val edgeSharp: Float,     // доля «сильных» краёв (90‑й перцентиль)
        val halftone: Float,      // энергия «кольца» в FFT (растр)
        val alphaHalo: Float,     // «ореолы» AA около сильных краёв

        val textLike: Float,      // текстоподобие (Оцу + CC + регулярность)
        val strokeCV: Float,      // вариация ширины штриха
        val outlineRatio: Float,  // доля тёмной кромки вокруг заливок
        val orthEdge: Float,      // доля ортогональных направлений
        val blockiness: Float,    // «ступенчатость» + dPSNR
        val largestRegion: Float, // доля крупнейшей области (12‑бит ключ)
        val tileScore: Float,     // автокорреляция тайла
        val tileRelPeriod: Float, // относительный период (0..1)
    )

    data class Options(
        val maxSide: Int = 256,
        val tau: Float = 0.18f,         // температура softmax (DISCRETE vs PHOTO)
        // гейты/пороговые эвристики
        val halftoneGate: Float = 0.50f,
        val pixelBlockinessGate: Float = 0.65f,
        val orthEdgeGate: Float = 0.55f,
        val textLikeGate: Float = 0.55f,
        val top8DiscreteMin: Float = 0.88f,
        val c16DiscreteMax: Int = 22,
        val tileGate: Float = 0.55f,
        val tileRelSmallMax: Float = 0.25f,
        // гистерезис
        val hysteresisKeepPrevAbove: Float = 0.70f,
        val hysteresisDeltaToSwitch: Float = 0.08f
    )

    data class Decision(
        val mode: Mode,
        val confidence: Float,       // softmax(DISC,PHOTO) для выбранного mode
        val toggles: Toggles?,       // только для DISCRETE; для PHOTO = null
        val scores: Map<Mode, Float>,
        val features: Features
    )

    fun detect(src: Bitmap, opts: Options = Options()): Decision {
        val bm = scaleToMaxSide(src, opts.maxSide)
        val f = computeFeatures(bm)

        // --- SCORE_DISCRETE ---
        fun nC16() = ((f.c16 - 2f) / 30f).coerceIn(0f, 1f)
        fun inv(x: Float) = (1f - x).coerceIn(0f, 1f)

        var scoreDiscrete =
            0.28f * f.top8Cover +
                    0.16f * inv(nC16()) +
                    0.16f * f.edgeSharp +
                    0.10f * inv(f.rarePct) +
                    0.10f * inv(f.halftone) +
                    0.08f * f.largestRegion +
                    0.06f * f.orthEdge +
                    0.06f * inv(f.gradScore)

        // усиление «флагов»/простых логотипов
        if (f.top8Cover >= 0.92f && f.c16 <= 14 && (f.gradScore <= 0.33f || f.orthEdge >= 0.60f)) {
            scoreDiscrete = min(1f, scoreDiscrete * 1.06f)
        }

        // при сильной «пиксельности» ослабим альтернативу
        if (f.blockiness >= opts.pixelBlockinessGate && f.orthEdge >= 0.60f) {
            scoreDiscrete = min(1f, scoreDiscrete * 1.05f)
        }

        // --- SCORE_PHOTO ---
        var scorePhoto =
            0.35f * nC16() +
                    0.25f * f.gradScore +
                    0.15f * inv(f.edgeSharp) +
                    0.15f * inv(f.top8Cover) +
                    0.10f * f.halftone

        // мелкий тайл/растр → ещё фото
        if (f.tileScore >= opts.tileGate && f.tileRelPeriod in 0f..opts.tileRelSmallMax) {
            scorePhoto = min(1f, scorePhoto * 1.08f)
        }
        // «soft»
        if (f.edgeSharp <= 0.30f && f.gradScore in 0.10f..0.30f) {
            scorePhoto = min(1f, scorePhoto * 1.06f)
        }
        // фото не любит растр>gate (но halftone уже в составе scorePhoto)
        if (f.halftone >= opts.halftoneGate) {
            scorePhoto = min(1f, scorePhoto * 1.04f)
        }

        // --- первичные гейты для устойчивости ---
        val gateDiscrete = (f.top8Cover >= opts.top8DiscreteMin && f.c16 <= opts.c16DiscreteMax)
        val gatePhoto    = (f.halftone >= opts.halftoneGate) ||
                (f.tileScore >= opts.tileGate && f.tileRelPeriod in 0f..opts.tileRelSmallMax) ||
                ((f.c16 >= 24) || (f.gradScore >= 0.25f))

        val scoresRaw = mapOf(
            Mode.DISCRETE to if (gateDiscrete) scoreDiscrete else 0.55f * scoreDiscrete,
            Mode.PHOTO    to if (gatePhoto)    scorePhoto    else 0.55f * scorePhoto
        )
        val scores = softmax(scoresRaw, opts.tau)

        // --- winner + гистерезис ---
        var (mode, conf) = scores.maxByOrNull { it.value }?.let { it.key to it.value }
            ?: (Mode.PHOTO to 0f)

        val sig = signature(bm, f)
        lastBySignature[sig]?.let { last ->
            if (last.p >= opts.hysteresisKeepPrevAbove) {
                val need = last.p + opts.hysteresisDeltaToSwitch
                if (mode != last.mode && conf <= need) {
                    mode = last.mode; conf = last.p
                }
            }
        }
        lastBySignature[sig] = Last(mode, conf)

        // --- toggles (только для DISCRETE) ---
        val toggles = if (mode == Mode.DISCRETE) {
            val text = f.textLike >= opts.textLikeGate && f.strokeCV <= 0.40f && f.c16 <= 12
            val pixel = (f.blockiness >= opts.pixelBlockinessGate) && (f.orthEdge >= 0.60f) && (f.alphaHalo <= 0.15f)
            val aa = (f.alphaHalo >= 0.18f) && (f.edgeSharp >= 0.35f) && !pixel
            Toggles(text = text, pixelGrid = pixel, aa = aa)
        } else null

        logDecision(mode, conf, scores, f, toggles)
        return Decision(mode, conf, toggles, scores, f)
    }

    fun clearCache() = lastBySignature.clear()

    // --------------- ВНУТРЕННЕЕ: признаки ----------------

    private fun computeFeatures(bm: Bitmap): Features {
        val w = bm.width; val h = bm.height; val n = w * h
        val px = IntArray(n); bm.getPixels(px, 0, w, 0, 0, w, h)

        // 12‑бит гистограмма
        val hist12 = IntArray(4096)
        var i = 0
        while (i < n) {
            val c = px[i]
            val k = (((c ushr 16) and 0xFF) ushr 4 shl 8) or
                    (((c ushr 8 ) and 0xFF) ushr 4 shl 4) or
                    (( c         and 0xFF) ushr 4)
            hist12[k]++; i++
        }
        val nz = (0 until 4096).filter { hist12[it] > 0 }.sortedByDescending { hist12[it] }
        val top2Cover = nz.take(2).sumOf { hist12[it] }.toFloat() / n
        val top8Cover = nz.take(8).sumOf { hist12[it] }.toFloat() / n
        var rare = 0
        for (k in nz) if (hist12[k] / n.toFloat() < 0.005f) rare += hist12[k]
        val rarePct = (rare / n.toFloat()).coerceIn(0f, 1f)

        // кластеры OKLab (по 12‑бит корзинам)
        val cl = okLabClustersFrom12Bins(hist12, maxCenters = 48, tau = 0.08f)
        val c16 = cl.centers.size
        val top2Labs = cl.topK(2)

        // перекраска в центры — стабилизирует крайовые признаки
        val centersLab = FloatArray(c16 * 3).also { arr ->
            var j = 0; for (lab in cl.centers) { arr[j++] = lab[0]; arr[j++] = lab[1]; arr[j++] = lab[2] }
        }
        val pxQ = repaintToCenters(px, w, h, centersLab)

        // градиенты/края
        val (gradScore, edgeSharp, mags, thrStrong) = sobelGradStats(pxQ, w, h)
        // AA‑ореолы
        val alphaHalo = alphaHaloScore(pxQ, w, h, mags, thrStrong, top2Labs)
        // растр (FFT‑кольцо)
        val halftone = halftoneRingFFT(pxQ, w, h)
        // SWT‑лайт
        val (_, strokeCV) = strokeWidthStats(pxQ, w, h)
        val outlineRatio = outlineRatioMorph(pxQ, w, h, hist12)
        val textLike = textLikeIndex(pxQ, w, h)
        val blockiness = blockinessScore(pxQ, w, h)
        val orthEdge = orthogonalEdgeShare(pxQ, w, h)
        val largestRegion = largestRegionShare(pxQ, w, h)
        val backgroundPct = backgroundShare(pxQ, w, h)
        val tileScore = tileAutoCorrScore(pxQ, w, h)
        val tileRelPeriod = estimateRelPeriod(pxQ, w, h)

        return Features(
            top2Cover, top8Cover, c16, rarePct, backgroundPct,
            gradScore, edgeSharp, halftone, alphaHalo,
            textLike, strokeCV, outlineRatio, orthEdge, blockiness, largestRegion, tileScore, tileRelPeriod
        )
    }

    // --------- Реализации признаков (самодостаточные) ---------

    private data class GradPack(val gradAvg: Float, val edgeStrongShare: Float, val mags: IntArray, val thrStrong: Int)

    private fun sobelGradStats(px: IntArray, w: Int, h: Int): GradPack {
        val Y = IntArray(w*h)
        var i = 0
        while (i < px.size) {
            val c = px[i]
            Y[i] = ((54*((c ushr 16) and 0xFF) + 183*((c ushr 8) and 0xFF) + 19*(c and 0xFF)) / 256)
            i++
        }
        val mags = IntArray(w*h)
        var sum = 0L; var cnt = 0
        var y = 1
        while (y < h-1) {
            var x = 1
            while (x < w-1) {
                val i0 = y*w + x
                val gx = -Y[i0-w-1]-2*Y[i0-1]-Y[i0+w-1] + Y[i0-w+1]+2*Y[i0+1]+Y[i0+w+1]
                val gy =  Y[i0-w-1]+2*Y[i0-w]+Y[i0-w+1] - Y[i0+w-1]-2*Y[i0+w]-Y[i0+w+1]
                val m = abs(gx)+abs(gy)
                mags[i0] = m
                sum += m; cnt++
                x++
            }
            y++
        }
        val gradAvg = ((sum.toFloat()/max(1, cnt)) / 255f).coerceIn(0f, 1f)
        val nonZero = mags.filter { it > 0 }.sorted()
        val thr = if (nonZero.isEmpty()) 255 else nonZero[(nonZero.size * 0.90f).toInt().coerceAtMost(nonZero.size - 1)]
        val strong = mags.count { it >= thr }.toFloat() / max(1, cnt).toFloat()
        return GradPack(gradAvg, strong.coerceIn(0f,1f), mags, thr)
    }

    private fun alphaHaloScore(
        px: IntArray, w: Int, h: Int,
        mags: IntArray, thrStrong: Int,
        labsTop2: List<FloatArray>
    ): Float {
        if (labsTop2.size < 2) return 0f
        val c1 = labsTop2[0]; val c2 = labsTop2[1]
        val v0 = c2[0]-c1[0]; val v1 = c2[1]-c1[1]; val v2 = c2[2]-c1[2]
        val v2len = v0*v0 + v1*v1 + v2*v2
        if (v2len < 1e-6) return 0f
        val len = sqrt(v2len)
        val eps = 0.12f * len
        var halo = 0; var total = 0
        var y = 1
        while (y < h - 1) {
            var x = 1
            while (x < w - 1) {
                val i = y*w + x
                if (mags[i] < thrStrong) { x++; continue }
                val c = px[i]
                val lab = rgbToOkLab((c ushr 16) and 0xFF, (c ushr 8) and 0xFF, c and 0xFF)
                val pc0 = lab[0] - c1[0]; val pc1 = lab[1] - c1[1]; val pc2 = lab[2] - c1[2]
                val t = ((pc0*v0 + pc1*v1 + pc2*v2) / v2len).toFloat()
                if (t in 0f..1f) {
                    val pr0 = c1[0] + t*v0; val pr1 = c1[1] + t*v1; val pr2 = c1[2] + t*v2
                    val d0 = lab[0]-pr0; val d1 = lab[1]-pr1; val d2 = lab[2]-pr2
                    val d = sqrt(d0*d0 + d1*d1 + d2*d2)
                    if (d <= eps) halo++
                }
                total++; x++
            }
            y++
        }
        return if (total == 0) 0f else (halo.toFloat()/total.toFloat()).coerceIn(0f,1f)
    }

    private fun halftoneRingFFT(px: IntArray, w: Int, h: Int): Float {
        val N = min(128, min(w, h)).takeHighestOneBit().coerceAtLeast(32)
        val g = Array(N) { DoubleArray(N) }
        val hann = DoubleArray(N) { i -> 0.5 - 0.5*cos(2.0*PI*i/(N-1)) }
        var y = 0
        while (y < N) {
            val sy = (y*(h-1).toDouble()/(N-1)).toInt()
            var x = 0
            while (x < N) {
                val sx = (x*(w-1).toDouble()/(N-1)).toInt()
                val c = px[sy*w + sx]
                val lum = (54*((c ushr 16) and 0xFF) + 183*((c ushr 8) and 0xFF) + 19*(c and 0xFF)) / 256.0
                g[y][x] = lum * hann[x] * hann[y]
                x++
            }
            y++
        }
        val re = Array(N) { g[it].clone() }
        val im = Array(N) { DoubleArray(N) }
        // FFT по строкам
        y = 0; while (y < N) { fft1d(re[y], im[y], false); y++ }
        // FFT по столбцам
        val colRe = DoubleArray(N); val colIm = DoubleArray(N)
        var x = 0
        while (x < N) {
            var k = 0
            while (k < N) { colRe[k]=re[k][x]; colIm[k]=im[k][x]; k++ }
            fft1d(colRe, colIm, false)
            k = 0
            while (k < N) { re[k][x]=colRe[k]; im[k][x]=colIm[k]; k++ }
            x++
        }
        val half = N/2.0
        var ringEnergy = 0.0; var totalEnergy = 0.0
        y = 0
        while (y < N) {
            val dy = min(y, N - y).toDouble()
            x = 0
            while (x < N) {
                val dx = min(x, N - x).toDouble()
                val r = sqrt(dx*dx + dy*dy) / half
                val e = re[y][x]*re[y][x] + im[y][x]*im[y][x]
                if (r >= 0.02) totalEnergy += e
                if (r >= 0.08 && r <= 0.25) ringEnergy += e
                x++
            }
            y++
        }
        return if (totalEnergy <= 0.0) 0f else (ringEnergy/totalEnergy).toFloat().coerceIn(0f,1f)
    }

    private fun fft1d(re: DoubleArray, im: DoubleArray, invert: Boolean) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var len = 2
        while (len <= n) {
            val ang = 2.0 * Math.PI / len * if (invert) -1 else 1
            val wlenRe = cos(ang); val wlenIm = sin(ang)
            var i = 0
            while (i < n) {
                var wr = 1.0; var wi = 0.0
                var j2 = 0
                while (j2 < len/2) {
                    val uRe = re[i+j2]; val uIm = im[i+j2]
                    val vRe = re[i+j2+len/2]*wr - im[i+j2+len/2]*wi
                    val vIm = re[i+j2+len/2]*wi + im[i+j2+len/2]*wr
                    re[i+j2] = uRe + vRe; im[i+j2] = uIm + vIm
                    re[i+j2+len/2] = uRe - vRe; im[i+j2+len/2] = uIm - vIm
                    val nwr = wr*wlenRe - wi*wlenIm
                    val nwi = wr*wlenIm + wi*wlenRe
                    wr = nwr; wi = nwi
                    j2++
                }
                i += len
            }
            len = len shl 1
        }
        if (invert) for (i in 0 until n) { re[i] /= n; im[i] /= n }
    }

    private fun strokeWidthStats(px: IntArray, w: Int, h: Int): Pair<Float, Float> {
        val Y = IntArray(w*h)
        var i = 0; while (i < px.size) { val c=px[i]; Y[i]=((54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256); i++ }
        val gxA = IntArray(w*h); val gyA = IntArray(w*h); val mag = IntArray(w*h)
        var y = 1
        while (y < h-1) {
            var x = 1
            while (x < w-1) {
                val i0 = y*w + x
                val gx = -Y[i0-w-1]-2*Y[i0-1]-Y[i0+w-1] + Y[i0-w+1]+2*Y[i0+1]+Y[i0+w+1]
                val gy =  Y[i0-w-1]+2*Y[i0-w]+Y[i0-w+1] - Y[i0+w-1]-2*Y[i0+w]-Y[i0+w+1]
                gxA[i0]=gx; gyA[i0]=gy; mag[i0]=abs(gx)+abs(gy)
                x++
            }
            y++
        }
        val nz = mag.filter { it>0 }.sorted()
        val thr = if (nz.isEmpty()) 255 else nz[(nz.size*0.82f).toInt().coerceAtMost(nz.size-1)]
        val widths = ArrayList<Float>(512)
        var samples = 0
        y = 2
        while (y < h-2 && samples < 1024) {
            var x = 2
            while (x < w-2 && samples < 1024) {
                val i0 = y*w + x
                if (mag[i0] < thr) { x++; continue }
                val gx = gxA[i0].toFloat(); val gy = gyA[i0].toFloat()
                val len = hypot(gx, gy); if (len == 0f) { x++; continue }
                val nx = gx/len; val ny = gy/len
                fun march(sign: Int): Int {
                    var step = 0
                    var fx = x.toFloat(); var fy = y.toFloat()
                    while (step < 10) {
                        fx += sign*nx; fy += sign*ny
                        val xi = fx.roundToInt(); val yi = fy.roundToInt()
                        if (xi !in 1 until (w-1) || yi !in 1 until (h-1)) break
                        val idx = yi*w + xi
                        if (mag[idx] < 50) break
                        step++
                    }
                    return step
                }
                val left = march(-1); val right = march(+1)
                widths += (left + right + 1).toFloat()
                samples++; x += 2
            }
            y += 2
        }
        if (widths.isEmpty()) return Float.NaN to Float.NaN
        val mean = widths.average().toFloat()
        val sd = sqrt(widths.fold(0.0){ acc, v -> acc + (v-mean)*(v-mean) } / widths.size).toFloat()
        return mean to (if (mean==0f) Float.NaN else (sd/mean).coerceIn(0f,1f))
    }

    private fun outlineRatioMorph(px: IntArray, w: Int, h: Int, hist12: IntArray): Float {
        val sorted = (0 until 4096).filter { hist12[it] > 0 }.sortedByDescending { hist12[it] }
        var acc = 0; val total = w*h
        val keep = HashSet<Int>()
        for (k in sorted) { if (acc.toFloat()/total < 0.70f) { keep += k; acc += hist12[k] } else break }
        fun key12(c: Int) = (((c ushr 16) and 0xFF) ushr 4 shl 8) or (((c ushr 8) and 0xFF) ushr 4 shl 4) or ((c and 0xFF) ushr 4)
        val mask = BooleanArray(total) { i -> keep.contains(key12(px[i])) }
        val dark = BooleanArray(total)
        var i = 0; while (i < total) { val c=px[i]; val yv=(54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256; dark[i]= yv < 64; i++ }
        var edgeCnt = 0; var darkOnEdge = 0
        var y = 1
        while (y < h-1) {
            var x = 1
            while (x < w-1) {
                val i0 = y*w + x
                var hasOn=false; var hasOff=false
                var yy=y-1
                while (yy<=y+1) {
                    var xx=x-1
                    while (xx<=x+1) {
                        val v=mask[yy*w+xx]
                        hasOn = hasOn or v; hasOff = hasOff or !v
                        xx++
                    }
                    yy++
                }
                val e = hasOn && hasOff
                if (e) { edgeCnt++; if (dark[i0]) darkOnEdge++ }
                x++
            }
            y++
        }
        return if (edgeCnt==0) 0f else (darkOnEdge.toFloat()/edgeCnt.toFloat()).coerceIn(0f,1f)
    }

    private fun textLikeIndex(px: IntArray, w: Int, h: Int): Float {
        val Y = IntArray(w*h)
        val hist = IntArray(256)
        var i = 0; while (i < px.size) { val c=px[i]; val yv=(54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256; Y[i]=yv; hist[yv]++; i++ }
        val thr = otsu(hist, w*h)
        val bin = BooleanArray(w*h) { k -> Y[k] < thr }
        val seen = BooleanArray(w*h)
        data class CC(val x0:Int, val y0:Int, val x1:Int, val y1:Int, val area:Int, val holes:Int)
        val comps = ArrayList<CC>()
        val q = IntArray(w*h)
        var qs = 0; var qe = 0
        i = 0
        while (i < w*h) {
            if (seen[i] || !bin[i]) { i++; continue }
            var x0 = i%w; var x1 = x0; var y0 = i/w; var y1 = y0; var area = 0
            qs = 0; qe = 0; q[qe++] = i; seen[i]=true
            while (qs < qe) {
                val p = q[qs++]
                area++
                val py = p/w; val px0 = p - py*w
                x0 = min(x0, px0); x1 = max(x1, px0)
                y0 = min(y0, py);  y1 = max(y1, py)
                val nbs = intArrayOf(p-1,p+1,p-w,p+w)
                for (nb in nbs) {
                    if (nb<0 || nb>=w*h || seen[nb]) continue
                    val ny = nb/w; val nx = nb - ny*w
                    if (abs(nx-px0)+abs(ny-py)!=1) continue
                    if (bin[nb]) { seen[nb]=true; q[qe++]=nb }
                }
            }
            val holes = countHoles(bin, w, h, x0, y0, x1, y1)
            comps += CC(x0,y0,x1,y1,area,holes)
            i++
        }
        if (comps.isEmpty()) return 0f
        val filtered = comps.filter { cc ->
            val ww = (cc.x1-cc.x0+1); val hh = (cc.y1-cc.y0+1)
            val ar = ww.toFloat()/max(1,hh).toFloat()
            val fill = cc.area.toFloat()/max(1, ww*hh).toFloat()
            ar in 0.2f..5f && fill in 0.15f..0.95f
        }
        if (filtered.isEmpty()) return 0f
        val centers = filtered.map { (it.x0+it.x1)/2f to (it.y0+it.y1)/2f }
        val byRow = centers.groupBy { (it.second/max(8f, h/24f)).roundToInt() }
        var regRows = 0; var rows = 0
        for ((_, list) in byRow) {
            if (list.size < 3) { rows++; continue }
            val xs = list.map { it.first }.sorted()
            val ds = xs.zip(xs.drop(1)) { a,b -> b-a }
            if (ds.isEmpty()) { rows++; continue }
            val mean = ds.average().toFloat(); if (mean <= 0f) { rows++; continue }
            val sd = sqrt(ds.fold(0.0){acc,v-> acc+(v-mean)*(v-mean) } / ds.size).toFloat()
            val cv = (sd/mean).coerceAtMost(1f)
            if (cv <= 0.55f) regRows++
            rows++
        }
        val sAspect = (filtered.size.toFloat()/comps.size.toFloat()).coerceIn(0f,1f)
        val sHoles  = (filtered.count { it.holes > 0 }.toFloat()/ filtered.size.toFloat()).coerceIn(0f,1f)
        val sReg    = if (rows==0) 0f else (regRows.toFloat()/rows.toFloat())
        return (0.45f*sAspect + 0.35f*sReg + 0.20f*sHoles).coerceIn(0f,1f)
    }

    private fun countHoles(bin: BooleanArray, w:Int, h:Int, x0:Int, y0:Int, x1:Int, y1:Int): Int {
        val bw = x1-x0+1; val bh = y1-y0+1
        val vis = BooleanArray(bw*bh)
        fun idx(x:Int,y:Int)= (y-y0)*bw + (x-x0)
        var holes = 0
        var y = y0
        while (y <= y1) {
            var x = x0
            while (x <= x1) {
                val i = idx(x,y)
                if (vis[i]) { x++; continue }
                val fg = bin[y*w + x]
                if (fg) { vis[i]=true; x++; continue }
                var touchesBorder = false
                val qx = IntArray(bw*bh); val qy = IntArray(bw*bh); var qs=0; var qe=0
                qx[qe]=x; qy[qe]=y; qe++; vis[i]=true
                while (qs<qe) {
                    val cx=qx[qs]; val cy=qy[qs]; qs++
                    if (cx==x0 || cx==x1 || cy==y0 || cy==y1) touchesBorder = true
                    val nbs = arrayOf(cx-1 to cy, cx+1 to cy, cx to cy-1, cx to cy+1)
                    for ((nx,ny) in nbs) {
                        if (nx<x0 || nx>x1 || ny<y0 || ny>y1) continue
                        val ii = idx(nx,ny)
                        if (vis[ii]) continue
                        if (bin[ny*w + nx]) { vis[ii]=true; continue }
                        vis[ii]=true; qx[qe]=nx; qy[qe]=ny; qe++
                    }
                }
                if (!touchesBorder) holes++
                x++
            }
            y++
        }
        return holes
    }

    private fun blockinessScore(px: IntArray, w: Int, h: Int): Float {
        val Y = IntArray(w*h)
        var i = 0; while (i < px.size) { val c=px[i]; Y[i]=((54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256); i++ }
        var steps = 0; var all = 0
        var y = 0
        while (y < h) {
            var x = 1
            while (x < w) {
                val d = abs(Y[y*w + x] - Y[y*w + x - 1])
                if (d in 1..8) steps++
                all++; x++
            }
            y++
        }
        var x = 0
        while (x < w) {
            y = 1
            while (y < h) {
                val d = abs(Y[y*w + x] - Y[(y-1)*w + x])
                if (d in 1..8) steps++
                all++; y++
            }
            x++
        }
        val sStep = if (all==0) 0f else (steps.toFloat()/all.toFloat())

        // NN vs BL «down→up» PSNR‑gap
        fun downUpNN(): IntArray {
            val dw = max(1, w/2); val dh = max(1, h/2)
            val small = IntArray(dw*dh)
            var yy=0; while (yy<dh) { var xx=0; while (xx<dw) {
                small[yy*dw+xx] = px[(yy*2).coerceAtMost(h-1)*w + (xx*2).coerceAtMost(w-1)]
                xx++ }; yy++ }
            val big = IntArray(w*h)
            yy=0; while (yy<h) { var xx=0; while (xx<w) {
                big[yy*w+xx] = small[(yy/2).coerceAtMost(dh-1)*dw + (xx/2).coerceAtMost(dw-1)]
                xx++ }; yy++ }
            return big
        }
        fun downUpBL(): IntArray {
            val dw = max(1, w/2); val dh = max(1, h/2)
            val small = IntArray(dw*dh)
            var yy=0; while (yy<dh) { var xx=0; while (xx<dw) {
                small[yy*dw+xx] = px[(yy*2).coerceAtMost(h-1)*w + (xx*2).coerceAtMost(w-1)]
                xx++ }; yy++ }
            val big = IntArray(w*h)
            var yy2=0
            while (yy2 < h) {
                val sy = (yy2/2.0); val y0 = floor(sy).toInt().coerceIn(0, dh-1); val y1 = min(dh-1, y0+1)
                val ty = sy - y0
                var xx2=0
                while (xx2 < w) {
                    val sx = (xx2/2.0); val x0 = floor(sx).toInt().coerceIn(0, dw-1); val x1 = min(dw-1, x0+1)
                    val tx = sx - x0
                    val c00 = small[y0*dw + x0]; val c01 = small[y0*dw + x1]
                    val c10 = small[y1*dw + x0]; val c11 = small[y1*dw + x1]
                    fun lerp(a:Int,b:Int,t:Double) = (a + (b-a)*t).toInt()
                    val r = lerp((c00 ushr 16) and 0xFF, (c01 ushr 16) and 0xFF, tx)
                    val g = lerp((c00 ushr 8)  and 0xFF, (c01 ushr 8)  and 0xFF, tx)
                    val b = lerp( c00 and 0xFF,           c01 and 0xFF,           tx)
                    val r2 = lerp((c10 ushr 16) and 0xFF, (c11 ushr 16) and 0xFF, tx)
                    val g2 = lerp((c10 ushr 8)  and 0xFF, (c11 ushr 8)  and 0xFF, tx)
                    val b2 = lerp( c10 and 0xFF,           c11 and 0xFF,           tx)
                    val rr = lerp(r, r2, ty)
                    val gg = lerp(g, g2, ty)
                    val bb = lerp(b, b2, ty)
                    big[yy2*w + xx2] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
                    xx2++
                }
                yy2++
            }
            return big
        }
        fun psnr(a: IntArray, b: IntArray): Double {
            var mse = 0.0
            var i2 = 0
            while (i2 < a.size) {
                val ar=(a[i2] ushr 16) and 0xFF; val ag=(a[i2] ushr 8) and 0xFF; val ab=a[i2] and 0xFF
                val br=(b[i2] ushr 16) and 0xFF; val bg=(b[i2] ushr 8) and 0xFF; val bb=b[i2] and 0xFF
                val dr=ar-br; val dg=ag-bg; val db=ab-bb
                mse += (dr*dr + dg*dg + db*db)/3.0
                i2++
            }
            mse /= a.size
            return if (mse<=1e-9) 99.0 else 20.0*ln(255.0/sqrt(mse))/ln(10.0)
        }
        val dP = (psnr(px, downUpBL()) - psnr(px, downUpNN())).coerceAtLeast(0.0)
        val sPsnr = ((dP / 6.0).toFloat()).coerceIn(0f, 1f)
        return (0.55f*sStep + 0.45f*sPsnr).coerceIn(0f,1f)
    }

    private fun orthogonalEdgeShare(px: IntArray, w: Int, h: Int): Float {
        val Y = IntArray(w*h)
        var i = 0; while (i < px.size) { val c=px[i]; Y[i]=((54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256); i++ }
        val bins = FloatArray(4) // 0°,45°,90°,135°
        var total = 0
        var y = 1
        while (y < h-1) {
            var x = 1
            while (x < w-1) {
                val i0 = y*w + x
                val gx = -Y[i0-w-1]-2*Y[i0-1]-Y[i0+w-1] + Y[i0-w+1]+2*Y[i0+1]+Y[i0+w+1]
                val gy =  Y[i0-w-1]+2*Y[i0-w]+Y[i0-w+1] - Y[i0+w-1]-2*Y[i0+w]-Y[i0+w+1]
                val mag = abs(gx)+abs(gy)
                if (mag < 60) { x++; continue }
                val ang = atan2(gy.toFloat(), gx.toFloat()) * (180f/PI.toFloat())
                val a = ((ang+180f) % 180f)
                val idx = when {
                    a < 22.5f || a >= 157.5f -> 0
                    a < 67.5f -> 1
                    a < 112.5f -> 2
                    else -> 3
                }
                bins[idx] += 1f; total++; x++
            }
            y++
        }
        return if (total==0) 0f else (bins.maxOrNull() ?: 0f) / total.toFloat()
    }

    private fun largestRegionShare(px: IntArray, w: Int, h: Int): Float {
        fun key12(c: Int) = (((c ushr 16) and 0xFF) ushr 4 shl 8) or (((c ushr 8) and 0xFF) ushr 4 shl 4) or ((c and 0xFF) ushr 4)
        val seen = BooleanArray(w*h)
        var best = 0
        val q = IntArray(w*h); var qs=0; var qe=0
        var i = 0
        while (i < w*h) {
            if (seen[i]) { i++; continue }
            val k = key12(px[i])
            qs=0; qe=0; q[qe++]=i; seen[i]=true
            var area = 0
            while (qs<qe) {
                val p = q[qs++]
                area++
                val y = p/w; val x = p - y*w
                val nbs = intArrayOf(p-1,p+1,p-w,p+w)
                for (nb in nbs) {
                    if (nb<0 || nb>=w*h || seen[nb]) continue
                    val ny = nb/w; val nx = nb - ny*w
                    if (abs(nx-x)+abs(ny-y)!=1) continue
                    if (key12(px[nb]) == k) { seen[nb]=true; q[qe++]=nb }
                }
            }
            if (area>best) best=area
            i++
        }
        return best.toFloat() / (w*h).toFloat()
    }

    private fun backgroundShare(px: IntArray, w: Int, h: Int): Float {
        val n = w*h
        var sumL = 0f
        val L = FloatArray(n)
        var i = 0
        while (i < n) {
            val c = px[i]
            val lab = rgbToOkLab((c ushr 16) and 0xFF, (c ushr 8) and 0xFF, c and 0xFF)
            L[i] = lab[0]; sumL += lab[0]; i++
        }
        val m = sumL / max(1, n)
        var c0 = 0; var c1 = 0
        for (v in L) if (v < m) c0++ else c1++
        return max(c0, c1).toFloat() / n.toFloat()
    }

    private fun tileAutoCorrScore(px: IntArray, w: Int, h: Int): Float {
        fun row(y:Int)= IntArray(w){x-> val c=px[y*w+x]; ((54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256) }
        fun col(x:Int)= IntArray(h){y-> val c=px[y*w+x]; ((54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256) }
        fun ac1d(vec:IntArray, p:Int): Float {
            val mu = vec.average()
            var num=0.0; var den=0.0
            var i=p
            while (i<vec.size){
                num += (vec[i]-mu)*(vec[i-p]-mu)
                den += (vec[i]-mu)*(vec[i]-mu)
                i++
            }
            return if (den==0.0) 0f else (num/den).toFloat().coerceIn(0f,1f)
        }
        val r=row(h/2); val c=col(w/2)
        var best=0f
        var p = 6
        while (p <= 60) {
            best = max(best, max(ac1d(r,p), ac1d(c,p)))
            p++
        }
        return best
    }

    private fun estimateRelPeriod(px: IntArray, w: Int, h: Int): Float {
        fun lumRow(y: Int): IntArray {
            val L = IntArray(w)
            var i = 0; while (i < w) { val c=px[y*w+i]; L[i]=(54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256; i++ }
            return L
        }
        fun lumCol(x: Int): IntArray {
            val L = IntArray(h)
            var i = 0; while (i < h) { val c=px[i*w+x]; L[i]=(54*((c ushr 16) and 0xFF)+183*((c ushr 8) and 0xFF)+19*(c and 0xFF))/256; i++ }
            return L
        }
        fun ac1(vec: IntArray, p: Int): Float {
            val mu = vec.average()
            var num=0.0; var den=0.0
            var i = p; while (i < vec.size) { val a=vec[i]-mu; val b=vec[i-p]-mu; num+=a*b; den+=a*a; i++ }
            return if (den == 0.0) 0f else (num/den).toFloat()
        }
        val r = lumRow(h/2); val c = lumCol(w/2)
        var best = 0f; var bestP = 0
        val pMax = (min(w, h)/2).coerceAtLeast(12)
        var p = 6
        while (p <= pMax) {
            val s = max(ac1(r,p), ac1(c,p))
            if (s > best) { best = s; bestP = p }
            p++
        }
        val rel = bestP / min(w, h).toFloat()
        return if (best >= 0.35f) rel else 0f
    }

    private data class ClusterInfo(val centers: List<FloatArray>, val weights: IntArray) {
        fun topK(k: Int): List<FloatArray> {
            val idx = weights.indices.sortedByDescending { weights[it] }.take(k)
            return idx.map { centers[it] }
        }
    }

    private fun okLabClustersFrom12Bins(hist12: IntArray, maxCenters: Int, tau: Float): ClusterInfo {
        val bins = hist12.indices.filter { hist12[it] > 0 }.sortedByDescending { hist12[it] }
        val centers = ArrayList<FloatArray>(min(maxCenters, bins.size))
        val weights = ArrayList<Int>(min(maxCenters, bins.size))
        val tau2 = tau * tau
        fun rgbOfKey(k: Int): IntArray {
            val r = ((k ushr 8) and 0xF) shl 4
            val g = ((k ushr 4) and 0xF) shl 4
            val b = (k and 0xF) shl 4
            return intArrayOf(r, g, b)
        }
        outer@ for (k in bins) {
            val c = rgbOfKey(k)
            val lab = rgbToOkLab(c[0], c[1], c[2])
            for (i in centers.indices) {
                val r = centers[i]
                val dl = lab[0]-r[0]; val da = lab[1]-r[1]; val db = lab[2]-r[2]
                if (dl*dl + da*da + db*db <= tau2) {
                    weights[i] = weights[i] + hist12[k]
                    continue@outer
                }
            }
            centers += lab; weights += hist12[k]
            if (centers.size >= maxCenters) break
        }
        return ClusterInfo(centers, weights.toIntArray())
    }

    private fun repaintToCenters(src: IntArray, w:Int, h:Int, centersLab: FloatArray): IntArray {
        val n = src.size
        val out = IntArray(n)
        var i = 0
        while (i < n) {
            val c = src[i]
            val lab = rgbToOkLab((c ushr 16) and 0xFF, (c ushr 8) and 0xFF, c and 0xFF)
            var best = 0; var bestD = Float.POSITIVE_INFINITY; var j = 0
            while (j < centersLab.size) {
                val dl = lab[0]-centersLab[j]; val da = lab[1]-centersLab[j+1]; val db = lab[2]-centersLab[j+2]
                val d = dl*dl + da*da + db*db
                if (d < bestD) { bestD = d; best = j }
                j += 3
            }
            out[i] = okLabToRgbInt(centersLab[best], centersLab[best+1], centersLab[best+2])
            i++
        }
        return out
    }

    private fun otsu(hist: IntArray, total: Int): Int {
        var sum = 0.0
        for (t in 0..255) sum += t * hist[t].toDouble()
        var sumB = 0.0; var wB = 0.0
        var maxVar = -1.0; var thr = 127
        var t = 0
        while (t <= 255) {
            wB += hist[t]; if (wB == 0.0) { t++; continue }
            val wF = total - wB; if (wF == 0.0) break
            sumB += t * hist[t]
            val mB = sumB / wB; val mF = (sum - sumB) / wF
            val varBetween = wB * wF * (mB - mF).pow(2.0)
            if (varBetween > maxVar) { maxVar = varBetween; thr = t }
            t++
        }
        return thr
    }

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

    private fun okLabToRgbInt(L: Float, a: Float, b: Float): Int {
        val l = L + 0.3963377774f * a + 0.2158037573f * b
        val m = L - 0.1055613458f * a - 0.0638541728f * b
        val s = L - 0.0894841775f * a - 1.2914855480f * b
        val l3 = l*l*l; val m3 = m*m*m; val s3 = s*s*s
        var rLin = +4.0767416621f * l3 - 3.3077115913f * m3 + 0.2309699292f * s3
        var gLin = -1.2684380046f * l3 + 2.6097574011f * m3 - 0.3413193965f * s3
        var bLin = -0.0041960863f * l3 - 0.7034186147f * m3 + 1.7076147010f * s3
        fun gamma(xIn: Float): Float {
            val x = max(0f, xIn)
            return if (x <= 0.0031308f) 12.92f * x else (1.055 * x.toDouble().pow(1.0/2.4) - 0.055).toFloat()
        }
        val r = (gamma(rLin).coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        val g = (gamma(gLin).coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        val b8 = (gamma(bLin).coerceIn(0f, 1f) * 255f + 0.5f).toInt()
        return (0xFF shl 24) or (r shl 16) or (g shl 8) or b8
    }

    private fun scaleToMaxSide(src: Bitmap, maxSide: Int): Bitmap {
        val side = max(src.width, src.height)
        if (side <= maxSide) return src
        val scale = maxSide.toFloat() / side.toFloat()
        val nw = max(1, (src.width * scale).roundToInt())
        val nh = max(1, (src.height * scale).roundToInt())
        return Bitmap.createScaledBitmap(src, nw, nh, true)
    }

    private fun softmax(scores: Map<Mode, Float>, tau: Float): Map<Mode, Float> {
        val maxS = scores.values.maxOrNull() ?: 0f
        var sum = 0.0
        val tmp = HashMap<Mode, Float>(2)
        for ((k, s) in scores) {
            val z = exp(((s - maxS) / max(1e-3f, tau)).toDouble()).toFloat()
            tmp[k] = z; sum += z
        }
        if (sum <= 0.0) return scores.mapValues { 0f }
        return tmp.mapValues { (it.value / sum.toFloat()).coerceIn(0f, 1f) }
    }

    private data class Last(val mode: Mode, val p: Float)
    private val lastBySignature = HashMap<Int, Last>()
    private fun signature(bm: Bitmap, f: Features): Int {
        var h = 17
        h = 31*h + bm.width
        h = 31*h + bm.height
        h = 31*h + (f.top8Cover * 1000f).roundToInt()
        h = 31*h + max(0, f.c16)
        return h
    }

    private fun logDecision(mode: Mode, conf: Float, scores: Map<Mode, Float>, f: Features, toggles: Toggles?) {
        val tag = "SmartScene"
        Log.d(tag, "MODE=$mode conf=${"%.2f".format(java.util.Locale.US, conf)} scores=$scores toggles=$toggles")
        Log.d(tag, "Top8=${fmt(f.top8Cover)} C16=${f.c16} Grad=${fmt(f.gradScore)} Edge=${fmt(f.edgeSharp)} Halo=${fmt(f.alphaHalo)} Half=${fmt(f.halftone)}")
        Log.d(tag, "Text=${fmt(f.textLike)} CV=${fmt(f.strokeCV)} Outline=${fmt(f.outlineRatio)} Orth=${fmt(f.orthEdge)} Block=${fmt(f.blockiness)}")
        Log.d(tag, "Largest=${fmt(f.largestRegion)} Tile=${fmt(f.tileScore)} Rel=${fmt(f.tileRelPeriod)} Back=${fmt(f.backgroundPct)} Rare=${fmt(f.rarePct)}")
    }
    private fun fmt(v: Float) = String.format(java.util.Locale.US, "%.3f", v)
}
