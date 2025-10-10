
package com.appforcross.core.quant

import java.util.Random
/**
 * K-means в OKLab-пространстве.
 * @param lab входные точки (интерлив L,a,b)
 * @return центры (FloatArray размером K*3)
 */
fun kmeansLab(lab: FloatArray, K: Int, iters: Int, seed: Int): FloatArray {
    require(K >= 1) { "K must be >=1" }
    require(lab.size % 3 == 0) { "lab must be multiple of 3" }
    val n = lab.size / 3
    val rnd = Random(seed.toLong())
    // Инициализация центров случайными точками
    val centers = FloatArray(K * 3)
    val used = HashSet<Int>()
    var idx = 0
    while (idx < K) {
        val i = rnd.nextInt(n)
        if (used.add(i)) {
            val j = i * 3
            centers[idx * 3 + 0] = lab[j + 0]
            centers[idx * 3 + 1] = lab[j + 1]
            centers[idx * 3 + 2] = lab[j + 2]
            idx++
        }
    }
    val counts = IntArray(K)
    val sums = FloatArray(K * 3)
    val assign = IntArray(n) { -1 }
    repeat(maxOf(1, iters)) {
        java.util.Arrays.fill(counts, 0)
        java.util.Arrays.fill(sums, 0f)
        // Присваивания
        for (i in 0 until n) {
            val j = i * 3
            val l = lab[j]; val a = lab[j + 1]; val b = lab[j + 2]
            var bestK = 0
            var bestD = Float.POSITIVE_INFINITY
            var c = 0
            while (c < K) {
                val cl = centers[c * 3 + 0] - l
                val ca = centers[c * 3 + 1] - a
                val cb = centers[c * 3 + 2] - b
                val d = cl * cl + ca * ca + cb * cb
                if (d < bestD) { bestD = d; bestK = c }
                c++
            }
            assign[i] = bestK
            counts[bestK]++
            val k3 = bestK * 3
            sums[k3 + 0] += l
            sums[k3 + 1] += a
            sums[k3 + 2] += b
        }
        // Обновление центров
        for (k in 0 until K) {
            val c = counts[k]
            if (c > 0) {
                val k3 = k * 3
                centers[k3 + 0] = sums[k3 + 0] / c
                centers[k3 + 1] = sums[k3 + 1] / c
                centers[k3 + 2] = sums[k3 + 2] / c
            } else {
                // Пустой кластер — реинициализируем случайной точкой
                val i = rnd.nextInt(n)
                val j = i * 3
                centers[k * 3 + 0] = lab[j + 0]
                centers[k * 3 + 1] = lab[j + 1]
                centers[k * 3 + 2] = lab[j + 2]
            }
        }
    }
    return centers
}
