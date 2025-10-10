
package com.appforcross.core.apply

import com.appforcross.core.color.argbToOkLab
import com.appforcross.core.image.Raster

/**
 * Применяет LUT "центр -> разрешённый цвет" ко всему растру.
 * принимает исходный Raster.
 */
fun applyCentroidLut(
    full: Raster,
    centersLab: FloatArray,
    k2Allowed: IntArray,
    allowedArgb: IntArray
): Raster {
    require(full.width * full.height == full.argb.size) { "Raster size mismatch" }
    val w = full.width; val h = full.height
    val lab = argbToOkLab(full.argb)
    val out = IntArray(full.argb.size)
    val k = centersLab.size / 3
    var p3 = 0
    for (i in full.argb.indices) {
        val l = lab[p3 + 0]
        val a = lab[p3 + 1]
        val b = lab[p3 + 2]
        p3 += 3
        var bestK = 0
        var bestD = Float.POSITIVE_INFINITY
        var c = 0
        while (c < k) {
            val cl = centersLab[c * 3 + 0] - l
            val ca = centersLab[c * 3 + 1] - a
            val cb = centersLab[c * 3 + 2] - b
            val d = cl * cl + ca * ca + cb * cb
            if (d < bestD) { bestD = d; bestK = c }
            c++
        }
        val ai = k2Allowed[bestK]
        out[i] = allowedArgb[ai]
    }
    return Raster(w, h, out)
}
