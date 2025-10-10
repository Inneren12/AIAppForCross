
package com.appforcross.core.map
/**
 * Для каждого центра (OKLab) находит ближайший индекс в массиве allowedLab (OKLab).
 * Возвращает IntArray размера K: центр -> индекс допустимого цвета.
 */
fun mapCentersToAllowed(centersLab: FloatArray, allowedLab: FloatArray): IntArray {
    val k = centersLab.size / 3
    val m = allowedLab.size / 3
    require(k >= 1) { "centers must not be empty" }
    require(m >= 1) { "allowed must not be empty" }
    val out = IntArray(k)
    for (ci in 0 until k) {
        val c3 = ci * 3
        val cl = centersLab[c3 + 0]
        val ca = centersLab[c3 + 1]
        val cb = centersLab[c3 + 2]
        var best = 0
        var bestD = Float.POSITIVE_INFINITY
        for (ai in 0 until m) {
            val a3 = ai * 3
            val dl = allowedLab[a3 + 0] - cl
            val da = allowedLab[a3 + 1] - ca
            val db = allowedLab[a3 + 2] - cb
            val d = dl * dl + da * da + db * db
            if (d < bestD) { bestD = d; best = ai }
        }
        out[ci] = best
    }
    return out
}
