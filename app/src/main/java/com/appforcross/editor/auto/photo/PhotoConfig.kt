package com.appforcross.editor.auto.photo

/**
 * Пресеты и базовые пороги для фото-ветки (Спринт 1).
 * Kneedle по ΔE95 + BandingIndex, без ROI и энергии дизера.
 */
object PhotoConfig {

    enum class Preset { PORTRAIT, LANDSCAPE, SOFT_ILLUSTRATION }

    data class Params(
        // --- Seeding / KMeans
        val wL: Float,              // вес L*: d^2 = wL*ΔL^2 + Δa^2 + Δb^2
        val rhoEdge: Float,         // бонус семенам на сильных краях
        val rhoRoi: Float = 0f,     // усиление ошибок/seed'ов в ROI (0..2)
        val kMin: Int,
        val kMax: Int,
        val kStep: Int = 4,
        val kneedleEps: Float = 0.02f, // порог «замедления» ошибки

        // --- Dither / анти-конфетти
        val orderedBias: Boolean = false,
        val antiConfetti: Boolean = true,
        // --- Regularizers (v1)
        val lambdaBand: Float = 0f,      // сила выравнивания L‑ступеней
        val lambdaConfetti: Float = 0f,  // сила анти‑конфетти прокси
        // --- Look‑ahead / shortlist
        val threadSnapCap: Float = 0.0f, // макс. шаг проекции в OKLab
        val shortlistM: Int = 0,         // 0 — выкл; 3..4 — вкл

        // --- PASS-профиль PHOTO
        val dE95Max: Float,
        val edgeSSIMMin: Float,
        val hfRetainMin: Float,
        val bandingMax: Float,
        val confettiMax: Float,

        // --- Score
        val scoreMin: Float = 0.75f
    )

    // Быстрые пресеты (из твоей таблицы)
    // Portrait: wL=1.2, ρroi=2, ρedge=0.5, λband=0.7, λconfetti=0.5, K до 50
    val Portrait = Params(
        wL = 1.2f, rhoEdge = 0.5f, rhoRoi = 2f,
        kMin = 24, kMax = 50,
        dE95Max = 18f, edgeSSIMMin = 0.70f, hfRetainMin = 0.75f,
        bandingMax = 0.09f, confettiMax = 0.05f,
        lambdaBand = 0.7f, lambdaConfetti = 0.5f,
        threadSnapCap = 0.02f, shortlistM = 4
    )

    // Landscape: wL=1.1, ρroi=1, λband=0.5, λconfetti=0.4, K до 60
    val Landscape = Params(
        wL = 1.1f, rhoEdge = 0.5f, rhoRoi = 1f,
        kMin = 24, kMax = 60,
        dE95Max = 20f, edgeSSIMMin = 0.70f, hfRetainMin = 0.70f,
        bandingMax = 0.10f, confettiMax = 0.055f,
        lambdaBand = 0.5f, lambdaConfetti = 0.4f,
        threadSnapCap = 0.02f, shortlistM = 4
    )

    // Soft illustration: wL=1.3, ordered-bias, K до 35, blends=OFF, shortlists M=3
    val SoftIllustration = Params(
        wL = 1.3f, rhoEdge = 0.5f,
        kMin = 20, kMax = 35,
        dE95Max = 16f, edgeSSIMMin = 0.70f, hfRetainMin = 0.70f,
        bandingMax = 0.08f, confettiMax = 0.05f,
        orderedBias = true,
        lambdaBand = 0.5f, lambdaConfetti = 0.4f,
        threadSnapCap = 0.015f, shortlistM = 3
    )

    val defaultSizes = intArrayOf(180, 200, 240, 300, 360, 400, 480, 600)
}
