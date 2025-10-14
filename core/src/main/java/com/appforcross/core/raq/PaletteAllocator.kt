package com.appforcross.core.raq

import com.appforcross.core.zones.Zone
import com.appforcross.core.zones.ZoneMasks
import com.appforcross.core.zones.ZoneWeights
import kotlin.math.max
import kotlin.math.pow

data class RAQBounds(
    val skyMin: Int,   val skyMax: Int,
    val cloudMin: Int, val cloudMax: Int,
    val waterMin: Int, val waterMax: Int,
    val vegMin: Int,   val vegMax: Int,
    val groundMin: Int,val groundMax: Int,
    val builtMin: Int, val builtMax: Int,
    val skinMin: Int,  val skinMax: Int,
    )

/** Распределение глобального лимита цветов между зонами по площади × доле краёв. */
object PaletteAllocator {
    fun allocateCaps(
        wz: ZoneWeights,
        edgeMask: BooleanArray?, // может быть null
        total: Int,
        bounds: RAQBounds
    ): Map<Zone, Int> {
        val n = wz.sky.size
        val areas = FloatArray(7)
        var edgesW = FloatArray(7)
        var i = 0
        while (i < n) {
            val e = if (edgeMask != null && edgeMask[i]) 1f else 0f
            areas[0] += wz.sky[i];   if (e > 0f) edgesW[0] += wz.sky[i]
            areas[1] += wz.cloud[i]; if (e > 0f) edgesW[1] += wz.cloud[i]
            areas[2] += wz.water[i]; if (e > 0f) edgesW[2] += wz.water[i]
            areas[3] += wz.veg[i];   if (e > 0f) edgesW[3] += wz.veg[i]
            areas[4] += wz.ground[i];if (e > 0f) edgesW[4] += wz.ground[i]
            areas[5] += wz.built[i]; if (e > 0f) edgesW[5] += wz.built[i]
            areas[6] += wz.skin[i]; if (e > 0f) edgesW[6] += wz.skin[i]
            i++
        }
        val wArr = FloatArray(7)
        var sumW = 0f
        i = 0
        while (i < 7) {
            val area = areas[i]
            val edgeShare = if (area > 1e-3f) edgesW[i] / area else 0f
            val w = area.coerceAtLeast(1f).toDouble().pow(0.8).toFloat() * (1f + 0.5f * edgeShare)
            wArr[i] = w; sumW += w
            i++
        }
        val caps = IntArray(7)
        fun clamp(v: Int, min: Int, max: Int) = kotlin.math.max(min, kotlin.math.min(max, v))
        // черновое распределение
        val boundsArr = arrayOf(
            bounds.skyMin to bounds.skyMax,
            bounds.cloudMin to bounds.cloudMax,
            bounds.waterMin to bounds.waterMax,
            bounds.vegMin to bounds.vegMax,
            bounds.groundMin to bounds.groundMax,
            bounds.builtMin to bounds.builtMax,
            bounds.skinMin to bounds.skinMax,
            )
        if (sumW < 1e-6f) {
            // равномерно по минимумам
            for (k in 0 until 6) caps[k] = boundsArr[k].first
        } else {
            for (k in 0 until 6) {
                val v = ((total * (wArr[k] / sumW)) + 0.5f).toInt()
                caps[k] = clamp(v, boundsArr[k].first, boundsArr[k].second)
            }
        }
        // подгон под ровно total
        var diff = total - caps.sum()
        val order = (0 until 7).sortedByDescending { wArr[it] }
        var ptr = 0
        while (diff != 0 && ptr < order.size * 3) {
            val k = order[ptr % order.size]
            val (mn, mx) = boundsArr[k]
            val cur = caps[k]
            if (diff > 0 && cur < mx) { caps[k] = cur + 1; diff-- }
            else if (diff < 0 && cur > mn) { caps[k] = cur - 1; diff++ }
            ptr++
        }
        return mapOf(
            Zone.SKY to caps[0], Zone.CLOUD to caps[1], Zone.WATER to caps[2],
            Zone.VEG to caps[3], Zone.GROUND to caps[4], Zone.BUILT to caps[5],
            Zone.SKIN to caps[6]
        )
    }
}

