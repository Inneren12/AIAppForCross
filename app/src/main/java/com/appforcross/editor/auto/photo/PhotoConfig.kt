package com.appforcross.editor.auto.photo

/**
 * Пресеты и базовые пороги для фото-ветки (Спринт 1).
 * Kneedle по ΔE95 + BandingIndex, без ROI и энергии дизера.
 */
object PhotoConfig {

    enum class Preset { PORTRAIT, LANDSCAPE, SOFT_ILLUSTRATION }

    enum class DitherMode { NONE, ORDERED, FS }

    data class PhotoParams(
        val paletteSize: Int,
        val preBlurSigma: Float,         // 0.0..1.2
        val ditherMode: DitherMode,      // NONE / ORDERED / FS
        val ditherStrength: Float,       // 0.2..0.5 for ORDERED, 0.6..0.9 for FS
        val edgeAwareQuant: Boolean,     // ΔE tie-break по контрасту
        val postCleanLevel: Int          // 0=off, 1=soft, 2=medium
    )

    /** Параметры (не меняют существующие пресеты/Params). */
    object B2 {
        /** Амплитуда ORDERED на превью в окнах бэндинга (0.0..0.5). */
        // более мягкий ordered, реже форсим его (FS-aniso по умолчанию)
        const val ORDERED_AMP: Float = 0.30f
        const val BAND_MASK_GATE: Float = 0.10f
        // аккуратный merge по OKLab^2
        const val MERGE_LAB_SQ: Float = 0.006f
    }

    /** Настройки Sprint B4 (анизотропный FS и protected-merge). */
    object B4 {
        /** Включить анизотропный FS (распространять ошибку вдоль касательной к границе). */
        const val ANISO_FS: Boolean = true
        /** Вес вдоль касательной (1.0 = базовый). */
        // анизотропный FS: вдоль края больше, поперёк меньше
        const val FS_ALONG: Float = 1.10f
        const val FS_ACROSS: Float = 0.30f
        /** Учитывать EdgeGuard при слиянии близких цветов. */
        const val PROTECTED_MERGE: Boolean = true
    }

    /** Настройки Sprint B5 (зоны, RAQ, SLIC-lite). */
    object B5 {
        // База (общие нейтрали) + зональные лимиты (в т.ч. SKIN)
        const val BASE_MIN: Int = 14
        const val SKIN_MIN: Int = 10; const val SKIN_MAX: Int = 14
        const val SKY_MIN: Int = 3;   const val SKY_MAX: Int = 12
        const val CLOUD_MIN: Int = 3; const val CLOUD_MAX: Int = 10
        const val WATER_MIN: Int = 3; const val WATER_MAX: Int = 12
        const val VEG_MIN: Int = 4;   const val VEG_MAX: Int = 16
        const val GROUND_MIN: Int = 2;const val GROUND_MAX: Int = 8
        const val BUILT_MIN: Int = 2; const val BUILT_MAX: Int = 8

        // Выборка для зонального k-means
        const val ZONE_SAMPLE_MAX: Int = 20000
        const val ZONE_KEEP_FRAC: Float = 0.35f

        // SLIC-lite
        const val SLIC_ON: Boolean = false
        const val SLIC_REGIONS: Int = 1500
        const val SLIC_MAJORITY_THRESHOLD: Float = 0.65f

        // Плоские зоны и пост-чистка
        const val FLAT_GRAD_T: Float = 0.012f
        const val NO_DITHER_IN_FLAT: Boolean = true
        const val CLEAN_SINGLETONS: Boolean = true

        // Портрет: skin/order/dark
        const val SKIN_ORDERED_AMP: Float = 0.24f
        const val DARK_L_T: Float = 0.10f

        // ===== Sprint ABC: ускорение без потери качества =====
        // A) Узкий перебор K вокруг «разумной» цели (для фото по умолчанию)
        const val AUTOTUNE_NARROW: Boolean = true
        const val CAP_GOAL_PHOTO: Int = 26       // целевой бюджет цветов для фото
        const val K_NEAR: Int = 2                // {goal-2, goal, goal+2, goal+6}
        const val K_FAR: Int = 6
        const val TRY_ORDERED_IF_SKY_FRAC: Float = 0.25f // ORDERED только если неба много
        // (blur в автотюне — только 0.0 для скорости)

        // B) Заморозить палитру (RAQ) до прохода по S
        const val FREEZE_PALETTE_AFTER_RAQ: Boolean = true

        // C) Ранний выход по S, если первый масштаб проходит ключевые пороги
        const val EARLY_EXIT_S: Boolean = true
        const val EARLY_EDGE_SSIM_MIN: Float = 0.70f
        const val EARLY_BANDING_MAX:   Float = 0.04f

        // Минимально допустимое число цветов после merge для фото (QA-гарантия)
        const val MIN_AFTER_MERGE: Int = 24
        // Порог мерджа в Lab^2: для портрета используем более строгий (меньше сливаний)
        const val MERGE_LAB_SQ_DEFAULT: Float = 9.0f    // ~ΔE≈3.0
        const val MERGE_LAB_SQ_PORTRAIT_STRICT: Float = 2.25f // ~ΔE≈1.5
    }


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
        kMin = 10, kMax = 50,
        dE95Max = 18f, edgeSSIMMin = 0.70f, hfRetainMin = 0.75f,
        bandingMax = 0.09f, confettiMax = 0.05f,
        lambdaBand = 0.7f, lambdaConfetti = 0.5f,
        threadSnapCap = 0.02f, shortlistM = 4
    )

    // Landscape: wL=1.1, ρroi=1, λband=0.5, λconfetti=0.4, K до 60
    val Landscape = Params(
        wL = 1.1f, rhoEdge = 0.5f, rhoRoi = 1f,
        kMin = 10, kMax = 60, kStep = 2,
        dE95Max = 20f, edgeSSIMMin = 0.70f, hfRetainMin = 0.70f,
        bandingMax = 0.10f, confettiMax = 0.055f,
        lambdaBand = 0.5f, lambdaConfetti = 0.4f,
        threadSnapCap = 0.02f, shortlistM = 4
    )

    // Soft illustration: wL=1.3, ordered-bias, K до 35, blends=OFF, shortlists M=3
    val SoftIllustration = Params(
        wL = 1.3f, rhoEdge = 0.5f,
        kMin = 10, kMax = 35,
        dE95Max = 16f, edgeSSIMMin = 0.70f, hfRetainMin = 0.70f,
        bandingMax = 0.08f, confettiMax = 0.05f,
        orderedBias = true,
        lambdaBand = 0.5f, lambdaConfetti = 0.4f,
        threadSnapCap = 0.015f, shortlistM = 3
    )

    val defaultSizes = intArrayOf(180, 200, 240, 300, 360, 400, 480, 600, 720)
}
